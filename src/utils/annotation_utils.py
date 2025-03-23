"""Utilities for handling annotations."""

import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

from src.utils.logger import get_logger


def load_annotations(annotation_path):
    """Load annotations from a JSON file.
    
    Args:
        annotation_path: Path to the annotation file
        
    Returns:
        List of annotations
    """
    logger = get_logger(name="load_annotations")
    
    try:
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        
        # Get the slide name from the file path
        slide_name = os.path.splitext(os.path.basename(annotation_path))[0].replace('_annotations', '')
        
        # Extract annotations from the nested structure
        if isinstance(data, dict) and slide_name in data and 'annotations' in data[slide_name]:
            annotations = data[slide_name]['annotations']
            logger.info(f"Loaded {len(annotations)} annotations from {annotation_path}")
            return annotations
        elif isinstance(data, list):
            logger.info(f"Loaded {len(data)} annotations from {annotation_path}")
            return data
        else:
            logger.warning(f"No annotations found in {annotation_path}")
            return []
            
    except Exception as e:
        logger.error(f"Error loading annotations from {annotation_path}: {e}")
        return []


def process_annotations(slide, annotations, output_dir, tile_size=512, overlap=64):
    """Process annotations and extract tiles from a slide.
    
    Args:
        slide: SlideReader object
        annotations: List of annotations
        output_dir: Directory to save tiles and annotations
        tile_size: Size of tiles to extract
        overlap: Overlap between tiles
        
    Returns:
        Dictionary of tile annotations
    """
    logger = get_logger(name="process_annotations")
    
    # Get slide dimensions
    width, height = slide.get_dimensions()
    logger.info(f"Processing slide with dimensions {width}x{height}")
    
    # Create slide-specific output directory
    slide_name = os.path.splitext(os.path.basename(slide.slide_path))[0]
    slide_output_dir = os.path.join(output_dir, slide_name)
    os.makedirs(slide_output_dir, exist_ok=True)
    logger.info(f"Created slide output directory: {slide_output_dir}")
    
    # Calculate tile coordinates
    tile_coordinates = []
    for y in range(0, height, tile_size - overlap):
        for x in range(0, width, tile_size - overlap):
            tile_coordinates.append((x, y))
    
    logger.info(f"Found {len(tile_coordinates)} tiles to process")
    
    # Process each tile
    tile_annotations = {}
    n_tiles_saved = 0
    
    for x, y in tqdm(tile_coordinates, desc="Processing tiles"):
        try:
            # Extract tile
            tile = slide.read_region((x, y), 0, (tile_size, tile_size))
            
            # Convert to RGB if necessary
            if tile.mode != 'RGB':
                tile = tile.convert('RGB')
            
            # Generate tile ID
            tile_id = f"tile_{x}_{y}"
            
            # Match annotations to this tile
            matched_annotations = match_annotations_to_tile(
                tile_id, (x, y), tile_size, annotations
            )
            
            # Save tile image
            tile_path = os.path.join(slide_output_dir, f"{tile_id}.png")
            logger.info(f"Saving tile to {tile_path}")
            
            try:
                tile.save(tile_path, format='PNG')
                n_tiles_saved += 1
                
                # Store annotations if any
                if matched_annotations:
                    tile_annotations[tile_id] = {
                        'file_path': f"{tile_id}.png",
                        'annotations': matched_annotations
                    }
                    logger.info(f"Saved tile {tile_id} with {len(matched_annotations)} annotations")
                else:
                    tile_annotations[tile_id] = {
                        'file_path': f"{tile_id}.png",
                        'annotations': []
                    }
                    logger.info(f"Saved tile {tile_id} with no annotations")
                
            except Exception as e:
                logger.error(f"Error saving tile {tile_id}: {e}")
            
            # Close tile
            tile.close()
            
        except Exception as e:
            logger.error(f"Error processing tile at ({x}, {y}): {e}")
            continue
    
    logger.info(f"Processed {len(tile_coordinates)} tiles, saved {n_tiles_saved} tiles")
    
    # Save annotations
    if tile_annotations:
        annotation_path = os.path.join(slide_output_dir, "annotations.json")
        try:
            with open(annotation_path, 'w') as f:
                json.dump(tile_annotations, f, indent=2)
            logger.info(f"Saved {len(tile_annotations)} tile annotations to {annotation_path}")
        except Exception as e:
            logger.error(f"Error saving annotations to {annotation_path}: {e}")
    else:
        logger.warning("No tiles were saved")
    
    return tile_annotations


def match_annotations_to_tile(tile_id, tile_pos, tile_size, annotations):
    """Match annotations to a specific tile.
    
    Args:
        tile_id: ID of the tile
        tile_pos: Position of the tile (x, y)
        tile_size: Size of the tile
        annotations: List of annotations
        
    Returns:
        List of matched annotations
    """
    logger = get_logger(name="match_annotations")
    
    matched_annotations = []
    tile_x, tile_y = tile_pos
    
    # Create tile bounding box
    tile_bbox = (tile_x, tile_y, tile_x + tile_size, tile_y + tile_size)
    logger.info(f"Checking annotations for tile {tile_id} at {tile_bbox}")
    
    # Check each annotation
    for annotation in annotations:
        try:
            # Get annotation bounding box
            bbox = annotation.get('bbox', None)
            if not bbox:
                logger.warning(f"No bbox found in annotation: {annotation}")
                continue
            
            # Convert from [x, y, width, height] to [x1, y1, x2, y2]
            annotation_bbox = (
                bbox[0], 
                bbox[1], 
                bbox[0] + bbox[2], 
                bbox[1] + bbox[3]
            )
            logger.info(f"Checking annotation with bbox {annotation_bbox}")
            
            # Check if annotation overlaps with tile
            if (annotation_bbox[0] < tile_bbox[2] and annotation_bbox[2] > tile_bbox[0] and
                annotation_bbox[1] < tile_bbox[3] and annotation_bbox[3] > tile_bbox[1]):
                
                # If it overlaps, adjust coordinates to be relative to tile
                rel_bbox = [
                    max(0, annotation_bbox[0] - tile_bbox[0]),
                    max(0, annotation_bbox[1] - tile_bbox[1]),
                    min(tile_size, annotation_bbox[2] - tile_bbox[0]),
                    min(tile_size, annotation_bbox[3] - tile_bbox[1])
                ]
                
                # Check if annotation is too small after adjustment
                if rel_bbox[2] - rel_bbox[0] < 10 or rel_bbox[3] - rel_bbox[1] < 10:
                    logger.info(f"Annotation too small after adjustment: {rel_bbox}")
                    continue
                
                # Convert back to [x, y, width, height]
                rel_bbox = [
                    rel_bbox[0],
                    rel_bbox[1],
                    rel_bbox[2] - rel_bbox[0],
                    rel_bbox[3] - rel_bbox[1]
                ]
                logger.info(f"Adjusted bbox: {rel_bbox}")
                
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
                                
                                # Make relative to tile
                                rel_x = x - tile_bbox[0]
                                rel_y = y - tile_bbox[1]
                                
                                # Keep only if inside tile
                                if (0 <= rel_x <= tile_size and 0 <= rel_y <= tile_size):
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
                        logger.info(f"Added annotation with {len(adjusted_segmentation)} segments")
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
                    logger.info(f"Added rectangular annotation from bbox")
            
        except Exception as e:
            logger.error(f"Error processing annotation: {e}")
            continue
    
    return matched_annotations 