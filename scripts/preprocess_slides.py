#!/usr/bin/env python
"""
preprocess_slides.py

Extracts patches from whole slide images and creates corresponding segmentation masks
from QuPath annotations. Designed specifically for glomeruli segmentation training.
"""

import os
import json
import glob
import numpy as np
import openslide
from PIL import Image
import cv2
from shapely.geometry import shape
from tqdm import tqdm
import random
from pathlib import Path
import logging
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prevent PIL from complaining about large images
Image.MAX_IMAGE_PIXELS = None

class SlidePreprocessor:
    def __init__(
        self,
        slide_path: str,
        annotation_path: str,
        output_dir: str,
        patch_size: int = 512,  # Standard size for glomeruli segmentation
        level: int = 0,  # Use highest magnification
        min_tissue_fraction: float = 0.1,
        min_glomeruli_fraction: float = 0.05,  # Increased to ensure glomeruli presence
        max_glomeruli_fraction: float = 0.9,  # Allow more glomeruli content
        classes: dict = None,
        visualize: bool = False,
        overlap: float = 0.0
    ):
        """
        Initialize the slide preprocessor.
        
        Args:
            slide_path: Path to the .svs slide file
            annotation_path: Path to the QuPath GeoJSON annotation file
            output_dir: Directory to save processed patches and masks
            patch_size: Size of patches to extract (default: 512)
            level: Magnification level to process (default: 0 = highest resolution)
            min_tissue_fraction: Minimum fraction of tissue required in a patch
            min_glomeruli_fraction: Minimum fraction of glomeruli required in a patch
            max_glomeruli_fraction: Maximum fraction of glomeruli allowed in a patch
            classes: Dictionary mapping class names to label indices
            visualize: Whether to create visualizations of patches and masks
            overlap: Fraction of overlap between patches (0.0 to 1.0)
        """
        self.slide_path = slide_path
        self.annotation_path = annotation_path
        self.output_dir = Path(output_dir)
        self.patch_size = patch_size
        self.level = level
        self.min_tissue_fraction = min_tissue_fraction
        self.min_glomeruli_fraction = min_glomeruli_fraction
        self.max_glomeruli_fraction = max_glomeruli_fraction
        self.overlap = overlap
        self.visualize = visualize
        
        # Default class mapping if none provided
        self.classes = classes or {
            "Normal": 1,
            "Sclerotic": 2,
            "Partially_sclerotic": 3,
            "Uncertain": 4
        }
        
        # Color map for visualizations
        self.color_map = {
            0: (0, 0, 0),      # Background - Black
            1: (0, 255, 0),    # Normal - Green
            2: (255, 0, 0),    # Sclerotic - Red
            3: (255, 255, 0),  # Partially_sclerotic - Yellow
            4: (0, 0, 255)     # Uncertain - Blue
        }
        
        # Open slide
        self.slide = openslide.OpenSlide(slide_path)
        self.slide_dims = self.slide.dimensions
        
        # Load annotations
        with open(annotation_path) as f:
            self.annotations = json.load(f)
            
        # Create output directories
        self.patches_dir = self.output_dir / "patches"
        self.masks_dir = self.output_dir / "masks"
        self.patches_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        # Create visualization directory
        if visualize:
            self.vis_dir = self.output_dir / "vis"
            self.vis_dir.mkdir(parents=True, exist_ok=True)

    def _is_tissue(self, patch: np.ndarray) -> bool:
        """
        Check if a patch contains enough tissue (non-background).
        
        Args:
            patch: RGB image array
            
        Returns:
            bool: True if patch contains enough tissue
        """
        # Convert to grayscale
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        
        # Otsu's thresholding to separate tissue from background
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Calculate tissue fraction
        tissue_fraction = np.sum(binary > 0) / binary.size
        
        return tissue_fraction >= self.min_tissue_fraction

    def _create_mask(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """
        Create a binary mask for the given region using annotations.
        
        Args:
            x, y: Top-left coordinates at level 0
            width, height: Dimensions of the region
            
        Returns:
            np.ndarray: Binary mask with class labels
        """
        mask = np.zeros((height, width), dtype=np.uint8)
        scale = self.slide.level_downsamples[self.level]
        region_size = (width * scale, height * scale)
        
        # Debug info
        logger.info(f"Creating mask for region at ({x}, {y}) with scale {scale}")
        logger.info(f"Target dimensions: {width}x{height}")
        
        for feature in self.annotations['features']:
            try:
                # Get class label
                if 'properties' not in feature or 'classification' not in feature['properties']:
                    continue
                    
                class_name = feature['properties']['classification']['name']
                label = self.classes.get(class_name, 0)  # 0 for background
                if label == 0:
                    continue
                    
                logger.info(f"Processing annotation of class: {class_name} (label: {label})")
                
                # Convert geometry to shapely polygon
                geometry = shape(feature['geometry'])
                
                # Check if geometry intersects with our region
                geom_bounds = geometry.bounds
                region_bounds = (x, y, x + region_size[0], y + region_size[1])
                
                if not (geom_bounds[0] < region_bounds[2] and 
                       geom_bounds[2] > region_bounds[0] and 
                       geom_bounds[1] < region_bounds[3] and 
                       geom_bounds[3] > region_bounds[1]):
                    continue
                
                # Handle both Polygon and MultiPolygon
                polygons = geometry.geoms if hasattr(geometry, 'geoms') else [geometry]
                
                for polygon in polygons:
                    # Get exterior coordinates
                    coords = np.array(polygon.exterior.coords)
                    
                    # Transform coordinates to patch space
                    # First translate to region coordinates (at level 0)
                    coords = coords - np.array([x, y])
                    
                    # Then scale down to target level
                    if scale != 1:
                        coords = coords / scale
                    
                    coords = coords.astype(np.int32)
                    
                    # Ensure coordinates are within bounds
                    coords[:, 0] = np.clip(coords[:, 0], 0, width - 1)
                    coords[:, 1] = np.clip(coords[:, 1], 0, height - 1)
                    
                    # Draw polygon
                    cv2.fillPoly(mask, [coords], label)
                    
                    # Handle holes
                    for interior in polygon.interiors:
                        coords = np.array(interior.coords)
                        coords = coords - np.array([x, y])
                        if scale != 1:
                            coords = coords / scale
                        coords = coords.astype(np.int32)
                        coords[:, 0] = np.clip(coords[:, 0], 0, width - 1)
                        coords[:, 1] = np.clip(coords[:, 1], 0, height - 1)
                        cv2.fillPoly(mask, [coords], 0)
                        
            except Exception as e:
                logger.warning(f"Error processing annotation: {e}")
                continue
        
        # Debug final mask
        unique_labels = np.unique(mask)
        logger.info(f"Mask contains labels: {unique_labels}")
        if len(unique_labels) > 1:
            logger.info(f"Label counts: {[np.sum(mask == label) for label in unique_labels]}")
        
        return mask

    def _get_glomeruli_fraction(self, mask: np.ndarray) -> float:
        """
        Calculate the fraction of glomeruli in the mask.
        
        Args:
            mask: Binary mask array
            
        Returns:
            float: Fraction of pixels that are glomeruli
        """
        total_pixels = mask.size
        glomeruli_pixels = np.sum(mask > 0)  # Count non-background pixels
        return glomeruli_pixels / total_pixels

    def _visualize_patch_and_mask(self, patch: np.ndarray, mask: np.ndarray, patch_name: str):
        """
        Create a visualization of the patch and its corresponding mask.
        
        Args:
            patch: RGB image array
            mask: Binary mask array
            patch_name: Name of the patch file
        """
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Plot original patch
        ax1.imshow(patch)
        ax1.set_title('Original Patch')
        ax1.axis('off')
        
        # Plot mask with color coding
        mask_colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for label, color in self.color_map.items():
            mask_colored[mask == label] = color
        ax2.imshow(mask_colored)
        ax2.set_title('Mask')
        ax2.axis('off')
        
        # Save the visualization
        vis_path = self.vis_dir / f"{patch_name}_vis.png"
        plt.savefig(vis_path, bbox_inches='tight', dpi=150)
        plt.close()

    def process_slide(self):
        """
        Process the slide by extracting patches centered on glomeruli annotations.
        """
        processed_patches = []
        scale = self.slide.level_downsamples[self.level]
        patch_size_0 = int(self.patch_size * scale)  # patch size at level 0
        
        logger.info(f"Processing slide {Path(self.slide_path).name}")
        logger.info(f"Dimensions: {self.slide_dims}")
        
        # First, collect all glomeruli centroids
        glomeruli = []
        for feature in self.annotations['features']:
            try:
                if 'properties' not in feature or 'classification' not in feature['properties']:
                    continue
                    
                class_name = feature['properties']['classification']['name']
                label = self.classes.get(class_name, 0)
                
                if label > 0:  # Only process non-background annotations
                    geometry = shape(feature['geometry'])
                    centroid = geometry.centroid
                    glomeruli.append((centroid.x, centroid.y, label))
                    
            except Exception as e:
                logger.warning(f"Error processing annotation: {e}")
                continue
        
        # Sort glomeruli by x coordinate to process them in order
        glomeruli.sort(key=lambda x: x[0])
        
        # Process each glomerulus
        for i, (centroid_x, centroid_y, label) in enumerate(glomeruli):
            # Calculate patch coordinates
            x = int(centroid_x - patch_size_0 / 2)
            y = int(centroid_y - patch_size_0 / 2)
            
            # Ensure coordinates are within slide bounds
            x = max(0, min(x, self.slide_dims[0] - patch_size_0))
            y = max(0, min(y, self.slide_dims[1] - patch_size_0))
            
            # Extract patch
            patch = np.array(self.slide.read_region((x, y), self.level, (self.patch_size, self.patch_size)))
            patch = cv2.cvtColor(patch, cv2.COLOR_RGBA2RGB)
            
            # Create mask
            mask = self._create_mask(x, y, self.patch_size, self.patch_size)
            
            # Check if patch meets criteria
            if self._is_tissue(patch) and self._get_glomeruli_fraction(mask) >= self.min_glomeruli_fraction and self._get_glomeruli_fraction(mask) <= self.max_glomeruli_fraction:
                # Save patch and mask
                patch_name = f"patch_{i:04d}"
                patch_path = self.patches_dir / f"{patch_name}.png"
                mask_path = self.masks_dir / f"{patch_name}.png"
                
                cv2.imwrite(str(patch_path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(mask_path), mask)
                
                processed_patches.append(patch_name)
                
                if self.visualize:
                    self._visualize_patch_and_mask(patch, mask, patch_name)
        
        return processed_patches

    def close(self):
        """Close the slide."""
        self.slide.close()


def create_splits(processed_patches: list, output_dir: Path, train_ratio=0.7, val_ratio=0.15):
    """
    Create train/val/test splits and save the split information.
    
    Args:
        processed_patches: List of processed patch information
        output_dir: Directory to save split information
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
    """
    # Shuffle patches
    random.seed(42)
    random.shuffle(processed_patches)
    
    # Calculate split indices
    n_patches = len(processed_patches)
    n_train = int(n_patches * train_ratio)
    n_val = int(n_patches * val_ratio)
    
    # Split patches
    train_patches = processed_patches[:n_train]
    val_patches = processed_patches[n_train:n_train + n_val]
    test_patches = processed_patches[n_train + n_val:]
    
    # Create split dictionary
    splits = {
        'train': train_patches,
        'val': val_patches,
        'test': test_patches
    }
    
    # Save splits
    with open(output_dir / 'splits.json', 'w') as f:
        json.dump(splits, f, indent=2)
        
    logger.info(f"Created splits: train={len(train_patches)}, val={len(val_patches)}, test={len(test_patches)}")
    return splits


def clean_processed_data(output_dir: str):
    """
    Clean up existing processed data to start fresh.
    
    Args:
        output_dir: Directory containing processed data
    """
    output_path = Path(output_dir)
    if output_path.exists():
        logger.info(f"Cleaning up existing processed data in {output_dir}")
        import shutil
        shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info("Cleanup complete")


def main():
    """Main function to process all slides in a directory."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess slides for glomeruli segmentation')
    parser.add_argument('--slides_dir', type=str, required=True,
                      help='Directory containing .svs slides')
    parser.add_argument('--annotations_dir', type=str, required=True,
                      help='Directory containing QuPath annotations')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save processed data')
    parser.add_argument('--patch_size', type=int, default=512,
                      help='Size of patches to extract')
    parser.add_argument('--level', type=int, default=0,
                      help='Magnification level to process')
    parser.add_argument('--min_tissue_fraction', type=float, default=0.1,
                      help='Minimum fraction of tissue required in a patch')
    parser.add_argument('--min_glomeruli_fraction', type=float, default=0.05,
                      help='Minimum fraction of glomeruli required in a patch')
    parser.add_argument('--max_glomeruli_fraction', type=float, default=0.9,
                      help='Maximum fraction of glomeruli allowed in a patch')
    parser.add_argument('--visualize', action='store_true',
                      help='Create visualizations of patches and masks')
    
    args = parser.parse_args()
    
    # Clean up existing processed data
    clean_processed_data(args.output_dir)
    
    # Process all slides
    all_processed_patches = []
    slides = glob.glob(os.path.join(args.slides_dir, '*.svs'))
    
    if not slides:
        logger.error(f"No .svs files found in {args.slides_dir}")
        return
        
    logger.info(f"Found {len(slides)} slides to process")
    
    for slide_path in slides:
        # Find matching annotation file
        slide_name = Path(slide_path).stem
        annotation_path = os.path.join(args.annotations_dir, f"{slide_name}.geojson")
        
        if not os.path.exists(annotation_path):
            logger.warning(f"No annotation found for {slide_name}, skipping")
            continue
            
        logger.info(f"Processing slide: {slide_name}")
            
        # Create preprocessor
        preprocessor = SlidePreprocessor(
            slide_path=slide_path,
            annotation_path=annotation_path,
            output_dir=args.output_dir,
            patch_size=args.patch_size,
            level=args.level,
            min_tissue_fraction=args.min_tissue_fraction,
            min_glomeruli_fraction=args.min_glomeruli_fraction,
            max_glomeruli_fraction=args.max_glomeruli_fraction,
            visualize=args.visualize
        )
        
        # Process slide
        try:
            patches = preprocessor.process_slide()
            all_processed_patches.extend(patches)
            logger.info(f"Successfully processed {len(patches)} patches from {slide_name}")
        except Exception as e:
            logger.error(f"Error processing {slide_name}: {e}")
        finally:
            preprocessor.close()
    
    # Create splits
    if all_processed_patches:
        logger.info(f"Total patches processed: {len(all_processed_patches)}")
        create_splits(all_processed_patches, Path(args.output_dir))
    else:
        logger.error("No patches were processed successfully")


if __name__ == '__main__':
    main() 