import os
import argparse
import json
import shutil
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger
from src.utils.io import load_json, save_json
from src.utils.slide_io import SlideReader, TileExtractor


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare dataset for Cascade Mask R-CNN')
    
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Path to raw data directory (with slide files)')
    parser.add_argument('--annotations-dir', type=str, default=None,
                        help='Path to annotations directory (optional, for QuPath annotations)')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Path to output directory')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--test-split', type=float, default=0.1,
                        help='Test split ratio')
    parser.add_argument('--tile-size', type=int, default=1024,
                        help='Size of tiles for WSI processing')
    parser.add_argument('--overlap', type=int, default=256,
                        help='Overlap between adjacent tiles')
    parser.add_argument('--level', type=int, default=0,
                        help='Magnification level (0 is highest resolution)')
    parser.add_argument('--filter-background', action='store_true',
                        help='Filter out background tiles')
    parser.add_argument('--background-threshold', type=int, default=220,
                        help='Threshold for background detection')
    
    return parser.parse_args()


def find_and_load_annotations(annotations_dir, slide_name):
    """Find and load annotations for a slide."""
    if not annotations_dir:
        return None
    
    # Check for exact match
    annotation_path = os.path.join(annotations_dir, f"{slide_name}_annotations.json")
    if os.path.exists(annotation_path):
        return load_json(annotation_path)
    
    # Check for partial match
    for file_path in glob(os.path.join(annotations_dir, "*_annotations.json")):
        basename = os.path.basename(file_path)
        file_slide_name = basename.replace("_annotations.json", "")
        
        if file_slide_name in slide_name or slide_name in file_slide_name:
            return load_json(file_path)
    
    return None


def match_annotations_to_tile(tile_info, annotations, level=0):
    """Match annotations to a specific tile."""
    if not annotations:
        return []
    
    matched_annotations = []
    
    # Get tile coordinates and dimensions
    tile_x = tile_info['x']
    tile_y = tile_info['y']
    tile_width = tile_info['width']
    tile_height = tile_info['height']
    
    # Get level scale factor
    scale_factor = 2 ** level if level > 0 else 1
    
    # Create tile bounding box
    tile_bbox = (tile_x, tile_y, tile_x + tile_width, tile_y + tile_height)
    
    # Check each annotation
    for annotation in annotations:
        # Get annotation bounding box
        bbox = annotation.get('bbox', [0, 0, 0, 0])
        
        # Convert from [x, y, width, height] to [x1, y1, x2, y2]
        annotation_bbox = (
            bbox[0], 
            bbox[1], 
            bbox[0] + bbox[2], 
            bbox[1] + bbox[3]
        )
        
        # Scale annotation if needed
        if level > 0:
            annotation_bbox = (
                annotation_bbox[0] / scale_factor,
                annotation_bbox[1] / scale_factor,
                annotation_bbox[2] / scale_factor,
                annotation_bbox[3] / scale_factor
            )
        
        # Check if annotation overlaps with tile
        if (annotation_bbox[0] < tile_bbox[2] and annotation_bbox[2] > tile_bbox[0] and
            annotation_bbox[1] < tile_bbox[3] and annotation_bbox[3] > tile_bbox[1]):
            
            # If it overlaps, adjust coordinates to be relative to tile
            rel_bbox = [
                max(0, annotation_bbox[0] - tile_bbox[0]),
                max(0, annotation_bbox[1] - tile_bbox[1]),
                min(tile_width, annotation_bbox[2] - tile_bbox[0]),
                min(tile_height, annotation_bbox[3] - tile_bbox[1])
            ]
            
            # Check if annotation is too small after adjustment
            if rel_bbox[2] - rel_bbox[0] < 10 or rel_bbox[3] - rel_bbox[1] < 10:
                continue
            
            # Convert back to [x, y, width, height]
            rel_bbox = [
                rel_bbox[0],
                rel_bbox[1],
                rel_bbox[2] - rel_bbox[0],
                rel_bbox[3] - rel_bbox[1]
            ]
            
            # Adjust segmentation if present
            if 'segmentation' in annotation:
                segmentation = annotation['segmentation']
                adjusted_segmentation = []
                
                for points in segmentation:
                    adjusted_points = []
                    
                    # Adjust polygon points
                    for i in range(0, len(points), 2):
                        if i + 1 < len(points):
                            x = points[i]
                            y = points[i + 1]
                            
                            # Scale if needed
                            if level > 0:
                                x /= scale_factor
                                y /= scale_factor
                            
                            # Make relative to tile
                            rel_x = x - tile_bbox[0]
                            rel_y = y - tile_bbox[1]
                            
                            # Keep only if inside tile
                            if (0 <= rel_x <= tile_width and 0 <= rel_y <= tile_height):
                                adjusted_points.extend([rel_x, rel_y])
                    
                    # Add only if we have enough points for a polygon
                    if len(adjusted_points) >= 6:
                        adjusted_segmentation.append(adjusted_points)
                
                # Create matched annotation
                if adjusted_segmentation:
                    matched_annotation = {
                        'bbox': rel_bbox,
                        'category': annotation['category'],
                        'segmentation': adjusted_segmentation
                    }
                    
                    matched_annotations.append(matched_annotation)
            else:
                # If no segmentation, create a rectangular one from bbox
                matched_annotation = {
                    'bbox': rel_bbox,
                    'category': annotation['category'],
                    'segmentation': [[
                        rel_bbox[0], rel_bbox[1],
                        rel_bbox[0] + rel_bbox[2], rel_bbox[1],
                        rel_bbox[0] + rel_bbox[2], rel_bbox[1] + rel_bbox[3],
                        rel_bbox[0], rel_bbox[1] + rel_bbox[3]
                    ]]
                }
                
                matched_annotations.append(matched_annotation)
    
    return matched_annotations


def process_slide(slide_path, output_dir, annotations_dir=None, tile_size=1024, overlap=256, 
                 level=0, filter_background=True, background_threshold=220):
    """Process a slide into tiles with annotations.
    
    Args:
        slide_path: Path to the slide file
        output_dir: Output directory for tiles and annotations
        annotations_dir: Directory with annotation files
        tile_size: Size of the tiles
        overlap: Overlap between adjacent tiles
        level: Magnification level to process
        filter_background: Whether to filter out background tiles
        background_threshold: Threshold for background detection
        
    Returns:
        Dictionary mapping tile IDs to annotations
    """
    logger = get_logger(name="prepare_dataset")
    
    # Get slide name
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    
    # Create output directory for this slide
    slide_output_dir = os.path.join(output_dir, slide_name)
    os.makedirs(slide_output_dir, exist_ok=True)
    
    try:
        # Open slide
        slide_reader = SlideReader(slide_path)
        
        # Create tile extractor
        tile_extractor = TileExtractor(
            slide_reader,
            tile_size=tile_size,
            overlap=overlap,
            level=level
        )
        
        # Find and load annotations if available
        slide_annotations = None
        if annotations_dir:
            annotation_data = find_and_load_annotations(annotations_dir, slide_name)
            if annotation_data:
                # Get annotations for this slide
                slide_key = next(iter(annotation_data.keys()))
                slide_annotations = annotation_data[slide_key].get('annotations', [])
                logger.info(f"Loaded {len(slide_annotations)} annotations for {slide_name}")
        
        # Get tile coordinates
        tile_coordinates = tile_extractor.get_tile_coordinates()
        
        # Process each tile
        tile_annotations = {}
        
        for tile_info in tqdm(tile_coordinates, desc=f"Processing {slide_name} tiles"):
            tile_x, tile_y = tile_info['index']
            
            # Extract tile
            tile_image, _ = tile_extractor.extract_tile(tile_x, tile_y)
            
            # Skip background tiles if requested
            if filter_background:
                # Convert to numpy array
                tile_np = np.array(tile_image)
                
                # Calculate mean pixel value for background detection
                mean_value = np.mean(tile_np)
                
                if mean_value > background_threshold:
                    continue
            
            # Generate tile ID
            tile_id = f"{slide_name}_tile_{tile_x}_{tile_y}"
            
            # Match annotations to this tile
            matched_annotations = []
            if slide_annotations:
                matched_annotations = match_annotations_to_tile(
                    tile_info, slide_annotations, level=level
                )
            
            # Save tile image
            tile_path = os.path.join(slide_output_dir, f"{tile_id}.png")
            tile_image.save(tile_path)
            
            # Store annotations for this tile
            if matched_annotations:
                tile_annotations[tile_id] = {
                    'file_path': f"{tile_id}.png",
                    'annotations': matched_annotations
                }
        
        # Close slide
        slide_reader.close()
        
        return tile_annotations
    
    except Exception as e:
        logger.error(f"Error processing slide {slide_path}: {e}")
        return {}


def main():
    # Parse arguments
    args = parse_args()
    
    # Create logger
    logger = get_logger(
        name="prepare_dataset",
        log_file=os.path.join(args.output_dir, "prepare_dataset.log")
    )
    
    # Create output directories
    os.makedirs(os.path.join(args.output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'val'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'test'), exist_ok=True)
    
    # Find all slide files
    slide_extensions = ['.svs', '.ndpi', '.tif', '.tiff']
    slide_paths = []
    
    for ext in slide_extensions:
        slide_paths.extend(glob(os.path.join(args.input_dir, f'**/*{ext}'), recursive=True))
    
    # Also search for regular image files (for backward compatibility)
    for ext in ['.jpg', '.jpeg', '.png']:
        slide_paths.extend(glob(os.path.join(args.input_dir, f'**/*{ext}'), recursive=True))
    
    logger.info(f"Found {len(slide_paths)} slide/image files")
    
    # Split dataset
    train_paths, test_paths = train_test_split(
        slide_paths, test_size=args.test_split, random_state=42)
    
    train_paths, val_paths = train_test_split(
        train_paths, test_size=args.val_split / (1 - args.test_split), random_state=42)
    
    logger.info(f"Split dataset: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")
    
    # Process each split
    train_annotations = {}
    val_annotations = {}
    test_annotations = {}
    
    # Process training slides
    for slide_path in tqdm(train_paths, desc="Processing training slides"):
        slide_annotations = process_slide(
            slide_path=slide_path,
            output_dir=os.path.join(args.output_dir, 'train'),
            annotations_dir=args.annotations_dir,
            tile_size=args.tile_size,
            overlap=args.overlap,
            level=args.level,
            filter_background=args.filter_background,
            background_threshold=args.background_threshold
        )
        train_annotations.update(slide_annotations)
    
    # Process validation slides
    for slide_path in tqdm(val_paths, desc="Processing validation slides"):
        slide_annotations = process_slide(
            slide_path=slide_path,
            output_dir=os.path.join(args.output_dir, 'val'),
            annotations_dir=args.annotations_dir,
            tile_size=args.tile_size,
            overlap=args.overlap,
            level=args.level,
            filter_background=args.filter_background,
            background_threshold=args.background_threshold
        )
        val_annotations.update(slide_annotations)
    
    # Process test slides
    for slide_path in tqdm(test_paths, desc="Processing test slides"):
        slide_annotations = process_slide(
            slide_path=slide_path,
            output_dir=os.path.join(args.output_dir, 'test'),
            annotations_dir=args.annotations_dir,
            tile_size=args.tile_size,
            overlap=args.overlap,
            level=args.level,
            filter_background=args.filter_background,
            background_threshold=args.background_threshold
        )
        test_annotations.update(slide_annotations)
    
    # Save annotations
    save_json(train_annotations, os.path.join(args.output_dir, 'train', 'train_annotations.json'))
    save_json(val_annotations, os.path.join(args.output_dir, 'val', 'val_annotations.json'))
    save_json(test_annotations, os.path.join(args.output_dir, 'test', 'test_annotations.json'))
    
    logger.info(f"Processed {len(train_annotations)} training tiles, {len(val_annotations)} validation tiles, and {len(test_annotations)} test tiles")
    logger.info("Dataset preparation completed")


if __name__ == '__main__':
    main()