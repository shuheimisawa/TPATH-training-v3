#!/usr/bin/env python
# convert_qupath_geojson.py - Convert QuPath GeoJSON annotations to training masks

import os
import json
import argparse
import numpy as np
import cv2
from shapely.geometry import shape, Polygon
from tqdm import tqdm
import openslide
from PIL import Image
import matplotlib.pyplot as plt
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Disable PIL's size limit
Image.MAX_IMAGE_PIXELS = None


class QuPathGeoJSONConverter:
    """Convert QuPath GeoJSON annotations to masks for training."""
    
    def __init__(self, 
                 wsi_path: str, 
                 geojson_path: str, 
                 output_dir: str,
                 class_mapping: dict = None,
                 tile_size: int = 512,
                 overlap: int = 64,
                 level: int = 0,
                 background_threshold: int = 240,
                 min_tissue_percentage: float = 0.08,
                 visualize: bool = False):
        """
        Initialize the converter.
        
        Args:
            wsi_path: Path to the whole slide image
            geojson_path: Path to the QuPath GeoJSON annotations
            output_dir: Directory to save the output masks
            class_mapping: Dictionary mapping QuPath classification names to class indices
            tile_size: Size of tiles to extract
            overlap: Overlap between adjacent tiles
            level: Magnification level to process
            background_threshold: Threshold for background filtering
            min_tissue_percentage: Minimum percentage of tissue required in a tile
            visualize: Whether to create visualization images
        """
        self.wsi_path = wsi_path
        self.geojson_path = geojson_path
        self.output_dir = output_dir
        
        # Default class mapping if none provided
        self.class_mapping = class_mapping or {
            "Normal": 1,
            "Sclerotic": 2,
            "Partially_sclerotic": 3,
            "Uncertain": 4,
            # Default to background class (0) for any other classification
            "default": 0
        }
        
        self.tile_size = tile_size
        self.overlap = overlap
        self.level = level
        self.background_threshold = background_threshold
        self.min_tissue_percentage = min_tissue_percentage
        self.visualize = visualize
        
        # Create output directories
        self.images_dir = os.path.join(output_dir, 'images')
        self.masks_dir = os.path.join(output_dir, 'masks')
        self.vis_dir = os.path.join(output_dir, 'visualizations')
        
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.masks_dir, exist_ok=True)
        if visualize:
            os.makedirs(self.vis_dir, exist_ok=True)
            
        # Set up slide and annotations
        self.slide = None
        self.annotations = None
        self.masks = {}
        
        # Color map for visualizations
        self.color_map = {
            0: (0, 0, 0),      # Background - Black
            1: (0, 255, 0),    # Normal - Green
            2: (255, 0, 0),    # Sclerotic - Red
            3: (255, 255, 0),  # Partially_sclerotic - Yellow
            4: (0, 0, 255)     # Uncertain - Blue
        }
        
        # Initialize slide and load annotations
        self._initialize()
        
    def _initialize(self):
        """Initialize slide and load annotations."""
        try:
            # Open slide
            self.slide = openslide.OpenSlide(self.wsi_path)
            logger.info(f"Opened slide: {os.path.basename(self.wsi_path)}")
            logger.info(f"Slide dimensions: {self.slide.dimensions}")
            logger.info(f"Slide levels: {self.slide.level_count}")
            
            # Load GeoJSON annotations
            with open(self.geojson_path, 'r') as f:
                self.geojson_data = json.load(f)
            
            # Parse annotations to extract polygons and classes
            self._parse_annotations()
            logger.info(f"Loaded {len(self.annotations)} annotations from GeoJSON")
            
        except Exception as e:
            logger.error(f"Error initializing converter: {e}")
            raise
    
    def _parse_annotations(self):
        """Parse QuPath GeoJSON annotations."""
        self.annotations = []
        
        # Check if it's a QuPath GeoJSON
        if 'type' in self.geojson_data and self.geojson_data['type'] == 'FeatureCollection':
            # Process features
            for feature in self.geojson_data['features']:
                try:
                    # Extract geometry
                    geometry = feature['geometry']
                    
                    # Extract classification
                    classification = "default"
                    if 'properties' in feature and 'classification' in feature['properties']:
                        classification = feature['properties']['classification']['name']
                    
                    # Get class index
                    class_index = self.class_mapping.get(classification, self.class_mapping['default'])
                    
                    # *** Add Warning for Unmapped Classifications ***
                    if class_index == 0 and classification != "default": # Check if it defaulted
                        logger.warning(f"Unmapped classification found: '{classification}'. Assigning to background (0).")
                    # *** End Addition ***
                    
                    # Create Shapely polygon
                    polygon = shape(geometry)
                    
                    # Add to annotations
                    self.annotations.append({
                        'polygon': polygon,
                        'class': class_index,
                        'classification': classification
                    })
                    
                except Exception as e:
                    logger.warning(f"Error parsing annotation: {e}")
        else:
            logger.warning("GeoJSON file does not appear to be a QuPath export")
    
    def _create_tile_mask(self, tile_x_l0, tile_y_l0, width, height, level):
        """
        Create a mask for a tile by rasterizing annotations.
        Args:
            tile_x_l0: Level 0 X-coordinate of tile top-left
            tile_y_l0: Level 0 Y-coordinate of tile top-left
            width: Width of tile mask (pixels at target level)
            height: Height of tile mask (pixels at target level)
            level: Magnification level
        Returns: Mask with class indices
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        downsample_factor = self.slide.level_downsamples[level] if level < self.slide.level_count else 1.0

        for annotation in self.annotations:
            try:
                polygon = annotation['polygon']
                class_index = annotation['class']
                if class_index == 0: continue

                if hasattr(polygon, 'exterior'):
                    # Single polygon
                    coords_level0 = np.array(polygon.exterior.coords)
                    coords_rel_level0 = coords_level0.copy()
                    coords_rel_level0[:, 0] -= tile_x_l0
                    coords_rel_level0[:, 1] -= tile_y_l0
                    coords_final = (coords_rel_level0 / downsample_factor).astype(np.int32)

                    # *** Simple Check before drawing ***
                    min_c, max_c = coords_final.min(axis=0), coords_final.max(axis=0)
                    if max_c[0] < 0 or max_c[1] < 0 or min_c[0] >= width or min_c[1] >= height:
                         # logger.debug(f"Tile L0({tile_x_l0},{tile_y_l0}) L{level} Dim({width}x{height}): Polygon class {class_index} entirely outside tile bounds. Skipping draw.")
                         continue # Skip if entirely outside

                    # logger.debug(f"Tile L0({tile_x_l0},{tile_y_l0}) L{level} Dim({width}x{height}): Drawing class {class_index} with final coords head: {coords_final[:2]}")
                    cv2.fillPoly(mask, [coords_final], class_index)

                    for interior in polygon.interiors:
                        interior_coords_level0 = np.array(interior.coords)
                        interior_coords_rel_level0 = interior_coords_level0.copy()
                        interior_coords_rel_level0[:, 0] -= tile_x_l0
                        interior_coords_rel_level0[:, 1] -= tile_y_l0
                        interior_coords_final = (interior_coords_rel_level0 / downsample_factor).astype(np.int32)
                        # logger.debug(f"Tile L0({tile_x_l0},{tile_y_l0}): Drawing HOLE.")
                        cv2.fillPoly(mask, [interior_coords_final], 0)

                elif hasattr(polygon, 'geoms'):
                    # MultiPolygon
                    for geom in polygon.geoms:
                        if hasattr(geom, 'exterior'):
                            coords_level0 = np.array(geom.exterior.coords)
                            coords_rel_level0 = coords_level0.copy()
                            coords_rel_level0[:, 0] -= tile_x_l0
                            coords_rel_level0[:, 1] -= tile_y_l0
                            coords_final = (coords_rel_level0 / downsample_factor).astype(np.int32)
                            
                            min_c, max_c = coords_final.min(axis=0), coords_final.max(axis=0)
                            if max_c[0] < 0 or max_c[1] < 0 or min_c[0] >= width or min_c[1] >= height:
                                # logger.debug(f"Tile L0({tile_x_l0},{tile_y_l0}) L{level} Dim({width}x{height}): MultiPolygon geom class {class_index} entirely outside tile bounds. Skipping draw.")
                                continue # Skip if entirely outside
                                
                            # logger.debug(f"Tile L0({tile_x_l0},{tile_y_l0}) L{level} Dim({width}x{height}): Drawing MultiPolygon geom class {class_index} with final coords head: {coords_final[:2]}")
                            cv2.fillPoly(mask, [coords_final], class_index)

                            for interior in geom.interiors:
                                interior_coords_level0 = np.array(interior.coords)
                                interior_coords_rel_level0 = interior_coords_level0.copy()
                                interior_coords_rel_level0[:, 0] -= tile_x_l0
                                interior_coords_rel_level0[:, 1] -= tile_y_l0
                                interior_coords_final = (interior_coords_rel_level0 / downsample_factor).astype(np.int32)
                                # logger.debug(f"Tile L0({tile_x_l0},{tile_y_l0}): Drawing MultiPolygon HOLE.")
                                cv2.fillPoly(mask, [interior_coords_final], 0)
            except Exception as e:
                logger.warning(f"Error creating mask for annotation: {e}")
        # Add final check of mask value
        if np.max(mask) == 0:
             # logger.debug(f"Tile L0({tile_x_l0},{tile_y_l0}): Mask is still empty after processing all annotations for this tile.")
             pass # Keep mask empty if nothing drawn
        return mask
    
    def _has_enough_tissue(self, image):
        """
        Check if an image has enough tissue (non-background pixels).
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Boolean indicating if image has enough tissue
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Create tissue mask (non-white pixels)
        tissue_mask = gray < self.background_threshold
        
        # Calculate tissue percentage
        tissue_percentage = np.mean(tissue_mask)
        
        # Check if enough tissue is present
        has_enough_tissue = tissue_percentage >= self.min_tissue_percentage
        
        return has_enough_tissue, tissue_percentage
    
    def process_tile(self, tile_idx, tile_x_l0, tile_y_l0, effective_tile_size_l0):
        """ Process a single tile. """
        try:
            # Level 0 coordinates for reading region (top-left), considering overlap
            actual_x_l0 = max(0, tile_x_l0 - self.overlap)
            actual_y_l0 = max(0, tile_y_l0 - self.overlap)

            # Dimensions at the target level
            level_width_total, level_height_total = self.slide.level_dimensions[self.level]

            # Calculate the size (width, height) of the tile to read *at the target level*
            # It should be tile_size, unless it extends beyond the level boundaries

            # Calculate top-left coordinates AT THE TARGET LEVEL to check boundaries
            downsample_factor = self.slide.level_downsamples[self.level]
            actual_x_level = actual_x_l0 / downsample_factor
            actual_y_level = actual_y_l0 / downsample_factor

            # Calculate width/height at the target level
            width_level = min(self.tile_size, level_width_total - int(actual_x_level))
            height_level = min(self.tile_size, level_height_total - int(actual_y_level))

            # Ensure width/height are positive
            if width_level <= 0 or height_level <= 0:
                 logger.debug(f"Tile {tile_idx}: Calculated dimensions L{self.level} ({width_level}x{height_level}) invalid. Skipping.")
                 return None

            logger.debug(f"Tile {tile_idx}: Reading L{self.level} region at L0 ({actual_x_l0},{actual_y_l0}) with size ({width_level}x{height_level})")

            # Read tile from slide using L0 coordinates but target level and target level size
            tile = self.slide.read_region((actual_x_l0, actual_y_l0), self.level, (width_level, height_level))
            tile = tile.convert('RGB')
            tile_array = np.array(tile)

            # Check tissue
            has_tissue, tissue_percentage = self._has_enough_tissue(tile_array)
            if not has_tissue:
                 logger.debug(f"Tile {tile_idx}: Skipped due to low tissue percentage ({tissue_percentage:.2f} < {self.min_tissue_percentage:.2f})")
                 return None

            # Create mask for this tile using L0 coordinates but level size
            mask = self._create_tile_mask(actual_x_l0, actual_y_l0, width_level, height_level, self.level)

            # Check if mask has any annotations AFTER processing
            max_mask_val = np.max(mask)
            if max_mask_val == 0:
                 logger.debug(f"Tile {tile_idx}: Skipped saving as mask is empty after processing annotations.")
                 # Decide whether to skip empty masks or save them
                 # For now, let's skip saving empty masks to avoid confusion
                 return None

            # Save tile image (size: width_level x height_level)
            image_filename = f"tile_{tile_idx:06d}.png"
            image_path = os.path.join(self.images_dir, image_filename)
            logger.debug(f"Tile {tile_idx}: Saving image to {image_path}")
            tile.save(image_path)

            # Save mask (size: width_level x height_level)
            mask_filename = f"mask_{tile_idx:06d}.png"
            mask_path = os.path.join(self.masks_dir, mask_filename)
            logger.debug(f"Tile {tile_idx}: Saving mask with max value {max_mask_val} to {mask_path}")
            # Ensure mask is saved correctly (e.g., no unintended scaling/type change)
            cv2.imwrite(mask_path, mask.astype(np.uint8))

            # Create visualization if requested
            if self.visualize:
                self._create_visualization(tile_array, mask, tile_idx)

            # Return tile information
            return {
                'index': tile_idx,
                'x': actual_x_l0, # Keep L0 coordinates for metadata consistency? Or use L-level?
                'y': actual_y_l0,
                'width': width_level, # Width at target level
                'height': height_level, # Height at target level
                'level': self.level,
                'image_path': image_path,
                'mask_path': mask_path,
                'tissue_percentage': tissue_percentage
            }

        except openslide.OpenSlideError as ose:
             logger.error(f"OpenSlideError processing tile {tile_idx} at L0({actual_x_l0},{actual_y_l0}) Size({width_level}x{height_level}) L{self.level}: {ose}")
             return None
        except Exception as e:
            # Log detailed error including coordinates and size for debugging
            log_msg = f"Error processing tile {tile_idx} at L0({actual_x_l0},{actual_y_l0}) Size({width_level}x{height_level}) L{self.level}: {e}"
            logger.error(log_msg, exc_info=True) # Add traceback
            return None
    
    def _create_visualization(self, image, mask, tile_idx):
        """
        Create a visualization of the mask overlaid on the image.
        
        Args:
            image: RGB image as numpy array
            mask: Mask with class indices
            tile_idx: Tile index
        """
        # Create a copy of the image
        vis = image.copy()
        
        # Create mask overlay
        overlay = np.zeros_like(vis)
        
        # Fill overlay with class colors
        for class_idx, color in self.color_map.items():
            if class_idx == 0:  # Skip background
                continue
                
            class_mask = (mask == class_idx)
            if np.any(class_mask):
                overlay[class_mask] = color
        
        # Blend overlay with image
        alpha = 0.5  # Transparency factor
        cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0, vis)
        
        # Draw class boundaries
        for class_idx, color in self.color_map.items():
            if class_idx == 0:  # Skip background
                continue
                
            # Find contours for this class
            class_mask = (mask == class_idx).astype(np.uint8) * 255
            contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw contours
            cv2.drawContours(vis, contours, -1, color, 2)
        
        # Save visualization
        vis_filename = f"vis_{tile_idx:06d}.png"
        vis_path = os.path.join(self.vis_dir, vis_filename)
        cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    
    def process_all_tiles(self, num_workers=4):
        """ Process all tiles in the slide. """
        # Calculate effective tile size AT LEVEL 0 for the grid
        # Overlap is defined at target level, scale it to L0 for grid step
        downsample_factor = self.slide.level_downsamples[self.level]
        effective_tile_size_level = self.tile_size - 2 * self.overlap # Step size at target level
        effective_tile_size_l0 = int(effective_tile_size_level * downsample_factor) # Step size at L0

        if effective_tile_size_l0 <= 0:
             logger.error(f"Effective tile size at Level 0 ({effective_tile_size_l0}) is not positive. Check tile_size ({self.tile_size}), overlap ({self.overlap}), and level ({self.level}).")
             return []

        # Generate tile coordinates based on L0 dimensions using L0 step size
        slide_width_l0, slide_height_l0 = self.slide.dimensions # These are L0 dimensions
        tile_coordinates_l0 = []

        for y_l0 in range(0, slide_height_l0, effective_tile_size_l0):
            for x_l0 in range(0, slide_width_l0, effective_tile_size_l0):
                tile_coordinates_l0.append((x_l0, y_l0))

        logger.info(f"Processing {len(tile_coordinates_l0)} tiles based on L0 grid (step={effective_tile_size_l0}) for target L{self.level} with {num_workers} workers")

        # Process tiles with parallel workers
        results = []

        # Ensure slide object can be pickled if using ProcessPoolExecutor
        # If slide object causes issues, consider reading slide path and opening in worker
        # For simplicity now, sticking to num_workers=1 if issues arise

        if num_workers > 1:
            # Need to ensure self.slide is not passed directly if it's not pickleable
            # Pass wsi_path instead and open slide within process_tile_worker
            logger.warning("Parallel processing might fail if OpenSlide object cannot be pickled. Consider num_workers=1 if errors occur.")
            # Placeholder for parallel execution logic if needed later
            # For now, defaulting to sequential to ensure debugging
            num_workers = 1 # Force sequential for now

        # Sequential processing (num_workers=1)
        processed_count = 0
        for tile_idx, (tile_x_l0, tile_y_l0) in enumerate(tqdm(tile_coordinates_l0)):
            result = self.process_tile(tile_idx, tile_x_l0, tile_y_l0, effective_tile_size_l0)
            if result is not None:
                results.append(result)
                processed_count += 1
                logger.debug(f"Successfully processed tile {tile_idx}. Total processed: {processed_count}")


        # Save metadata
        metadata = {
            'slide_path': self.wsi_path,
            'geojson_path': self.geojson_path,
            'tile_size': self.tile_size,
            'overlap': self.overlap,
            'level': self.level,
            'class_mapping': self.class_mapping,
            'tiles': results
        }
        
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Processed {len(results)} tiles with annotations")
        logger.info(f"Saved metadata to {metadata_path}")

        return results
    
    def create_dataset_split(self, train_ratio=0.7, val_ratio=0.15):
        """
        Create train/val/test split.
        
        Args:
            train_ratio: Ratio of training data
            val_ratio: Ratio of validation data
            
        Returns:
            Dictionary with split information
        """
        # Load metadata
        metadata_path = os.path.join(self.output_dir, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get all tile information
        tiles = metadata['tiles']
        
        # Shuffle tiles
        np.random.seed(42)
        np.random.shuffle(tiles)
        
        # Split tiles
        n_tiles = len(tiles)
        n_train = int(n_tiles * train_ratio)
        n_val = int(n_tiles * val_ratio)
        
        train_tiles = tiles[:n_train]
        val_tiles = tiles[n_train:n_train + n_val]
        test_tiles = tiles[n_train + n_val:]
        
        # Create split directories
        train_dir = os.path.join(self.output_dir, 'train')
        val_dir = os.path.join(self.output_dir, 'val')
        test_dir = os.path.join(self.output_dir, 'test')
        
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Create split metadata
        split_info = {
            'train': [tile['index'] for tile in train_tiles],
            'val': [tile['index'] for tile in val_tiles],
            'test': [tile['index'] for tile in test_tiles]
        }
        
        # Copy files to split directories
        for split, tiles_list in [('train', train_tiles), ('val', val_tiles), ('test', test_tiles)]:
            split_images_dir = os.path.join(self.output_dir, split, 'images')
            split_masks_dir = os.path.join(self.output_dir, split, 'masks')
            
            os.makedirs(split_images_dir, exist_ok=True)
            os.makedirs(split_masks_dir, exist_ok=True)
            
            for tile in tqdm(tiles_list, desc=f"Copying {split} tiles"):
                # Get filenames
                image_filename = os.path.basename(tile['image_path'])
                mask_filename = os.path.basename(tile['mask_path'])
                
                # Create symlinks or copy files
                os.symlink(
                    os.path.abspath(tile['image_path']),
                    os.path.join(split_images_dir, image_filename)
                )
                
                os.symlink(
                    os.path.abspath(tile['mask_path']),
                    os.path.join(split_masks_dir, mask_filename)
                )
        
        # Save split info
        split_path = os.path.join(self.output_dir, 'split_info.json')
        with open(split_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        logger.info(f"Created dataset split: {len(train_tiles)} train, {len(val_tiles)} val, {len(test_tiles)} test")
        logger.info(f"Saved split information to {split_path}")
        
        return split_info
    
    def close(self):
        """Close slide and release resources."""
        if self.slide is not None:
            self.slide.close()
            logger.info("Closed slide")


def process_slide(args):
    """
    Process a single slide with its annotations.
    
    Args:
        args: Command-line arguments
    """
    # Create converter
    converter = QuPathGeoJSONConverter(
        wsi_path=args.wsi_path,
        geojson_path=args.geojson_path,
        output_dir=args.output_dir,
        class_mapping=args.class_mapping,
        tile_size=args.tile_size,
        overlap=args.overlap,
        level=args.level,
        background_threshold=args.background_threshold,
        min_tissue_percentage=args.min_tissue_percentage,
        visualize=args.visualize
    )
    
    # Process all tiles
    converter.process_all_tiles(num_workers=args.num_workers)
    
    # Create dataset split if requested
    if args.create_split:
        converter.create_dataset_split(
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio
        )
    
    # Close converter
    converter.close()


def process_batch(args):
    """
    Process multiple slides in batch mode.
    
    Args:
        args: Command-line arguments
    """
    # Get all slide files
    slide_paths = []
    for ext in args.slide_extensions:
        slide_paths.extend(glob.glob(os.path.join(args.input_dir, f"*{ext}")))
    
    logger.info(f"Found {len(slide_paths)} slide files")
    
    if len(slide_paths) == 0:
        logger.error(f"No slide files found in {args.input_dir}")
        return
    
    # Process each slide
    for slide_path in slide_paths:
        try:
            # Get slide name
            slide_name = os.path.splitext(os.path.basename(slide_path))[0]
            logger.info(f"Processing slide: {slide_name}")
            
            # Find matching GeoJSON file
            geojson_path = os.path.join(args.annotations_dir, f"{slide_name}.geojson")
            if not os.path.exists(geojson_path):
                logger.warning(f"No annotations found for {slide_name}, skipping")
                continue
            
            # Create output directory for this slide
            slide_output_dir = os.path.join(args.output_dir, slide_name)
            os.makedirs(slide_output_dir, exist_ok=True)
            
            # Create converter
            converter = QuPathGeoJSONConverter(
                wsi_path=slide_path,
                geojson_path=geojson_path,
                output_dir=slide_output_dir,
                class_mapping=args.class_mapping,
                tile_size=args.tile_size,
                overlap=args.overlap,
                level=args.level,
                background_threshold=args.background_threshold,
                min_tissue_percentage=args.min_tissue_percentage,
                visualize=args.visualize
            )
            
            # Process all tiles
            converter.process_all_tiles(num_workers=args.num_workers)
            
            # Create dataset split if requested
            if args.create_split:
                converter.create_dataset_split(
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio
                )
            
            # Close converter
            converter.close()
            
        except Exception as e:
            logger.error(f"Error processing slide {slide_path}: {e}")
            continue


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Convert QuPath GeoJSON annotations to masks for training')
    
    parser.add_argument('--wsi-path', type=str,
                        help='Path to whole slide image')
    parser.add_argument('--geojson-path', type=str,
                        help='Path to QuPath GeoJSON annotations')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Path to output directory')
    
    # Batch processing arguments
    parser.add_argument('--batch-mode', action='store_true',
                        help='Process multiple slides in batch mode')
    parser.add_argument('--input-dir', type=str,
                        help='Directory containing slide files (for batch mode)')
    parser.add_argument('--annotations-dir', type=str,
                        help='Directory containing annotation files (for batch mode)')
    parser.add_argument('--slide-extensions', type=str, nargs='+', default=['.svs', '.tif', '.tiff', '.ndpi'],
                        help='Slide file extensions to look for (for batch mode)')
    
    # Processing parameters
    parser.add_argument('--tile-size', type=int, default=512,
                        help='Size of tiles to extract')
    parser.add_argument('--overlap', type=int, default=64,
                        help='Overlap between adjacent tiles')
    parser.add_argument('--level', type=int, default=0,
                        help='Magnification level to process')
    parser.add_argument('--background-threshold', type=int, default=240,
                        help='Threshold for background filtering (0-255)')
    parser.add_argument('--min-tissue-percentage', type=float, default=0.08,
                        help='Minimum percentage of tissue required (0-1)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of worker processes for parallel processing')
    
    # Dataset split parameters
    parser.add_argument('--create-split', action='store_true',
                        help='Create train/val/test split')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Ratio of training data')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Ratio of validation data')
    
    # Visualization parameters
    parser.add_argument('--visualize', action='store_true',
                        help='Create visualization images')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Define class mapping - modify this to match your QuPath classifications
    args.class_mapping = {
        "Normal": 1,
        "Sclerotic": 2,
        "Partially_sclerotic": 3,
        "Uncertain": 4,
        # Default to background class (0) for any other classification
        "default": 0
    }
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process in batch mode or single slide mode
    if args.batch_mode:
        if not args.input_dir or not args.annotations_dir:
            parser.error("--input-dir and --annotations-dir are required for batch mode")
        
        process_batch(args)
    else:
        if not args.wsi_path or not args.geojson_path:
            parser.error("--wsi-path and --geojson-path are required for single slide mode")
        
        process_slide(args)


if __name__ == '__main__':
    main()