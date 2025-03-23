"""Utilities for handling annotations."""

import os
import json
import gc
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from typing import Dict, List, Optional, Tuple

from src.utils.logger import get_logger
from src.utils.tissue_detection import process_tile, visualize_tissue_detection


def load_annotations(annotation_path: str) -> List[Dict]:
    """Load annotations from JSON file."""
    logger = get_logger(name="load_annotations")
    
    try:
        with open(annotation_path, 'r') as f:
            annotations = json.load(f)
        
        # Get the slide name from the file path
        slide_name = os.path.splitext(os.path.basename(annotation_path))[0].replace('_annotations', '')
        
        # Extract annotations from the nested structure
        if isinstance(annotations, dict) and slide_name in annotations and 'annotations' in annotations[slide_name]:
            annotations = annotations[slide_name]['annotations']
            logger.info(f"Loaded {len(annotations)} annotations from {annotation_path}")
            return annotations
        elif isinstance(annotations, list):
            logger.info(f"Loaded {len(annotations)} annotations from {annotation_path}")
            return annotations
        else:
            logger.warning(f"No annotations found in {annotation_path}")
            return []
            
    except Exception as e:
        logger.error(f"Error loading annotations from {annotation_path}: {e}")
        return []


def save_partial_annotations(annotations, output_dir, prefix="partial"):
    """Save annotations to a temporary file to avoid data loss in case of crash.
    
    Args:
        annotations: Dictionary of annotations
        output_dir: Directory to save the annotations
        prefix: Prefix for the output file name
    """
    logger = get_logger(name="save_partial_annotations")
    
    try:
        # Create temporary file path
        partial_path = os.path.join(output_dir, f"{prefix}_annotations.json")
        
        # Save annotations
        with open(partial_path, 'w') as f:
            json.dump(annotations, f, indent=2)
            
        logger.info(f"Saved {len(annotations)} partial annotations to {partial_path}")
    except Exception as e:
        logger.error(f"Error saving partial annotations: {e}")


def save_annotations(annotations, output_dir):
    """Save annotations to a file.
    
    Args:
        annotations: Dictionary of annotations
        output_dir: Directory to save the annotations
    """
    logger = get_logger(name="save_annotations")
    
    try:
        # Create file path
        annotation_path = os.path.join(output_dir, "annotations.json")
        
        # Save annotations
        with open(annotation_path, 'w') as f:
            json.dump(annotations, f, indent=2)
            
        logger.info(f"Saved {len(annotations)} annotations to {annotation_path}")
    except Exception as e:
        logger.error(f"Error saving annotations: {e}")


def process_annotations(
    slide,
    annotations: List[Dict],
    output_dir: str,
    tile_size: int = 512,
    overlap: int = 64,
    filter_background: bool = True,
    background_threshold: int = 240,
    min_tissue_percentage: float = 0.08
) -> Dict:
    """Process annotations and extract tiles.
    
    Args:
        slide: SlideReader object
        annotations: List of annotations
        output_dir: Directory to save tiles
        tile_size: Size of tiles to extract
        overlap: Overlap between tiles
        filter_background: Whether to filter out background tiles
        background_threshold: Threshold for background filtering
        min_tissue_percentage: Minimum percentage of tissue required
        
    Returns:
        Dictionary mapping tile paths to their annotations
    """
    logger = get_logger(name="process_annotations")
    
    tile_annotations = {}
    
    # Get slide dimensions
    width, height = slide.dimensions
    
    # Calculate tile positions
    tile_positions = []
    for x in range(0, width, tile_size - overlap):
        for y in range(0, height, tile_size - overlap):
            tile_positions.append((x, y))
    
    logger.info(f"Processing {len(tile_positions)} tiles")
    
    # Process each tile
    for tile_idx, (x, y) in enumerate(tile_positions):
        try:
            # Read tile
            tile = slide.read_region((x, y), 0, (tile_size, tile_size))
            tile_array = np.array(tile)
            
            # Check tissue content if filtering is enabled
            if filter_background:
                has_enough_tissue, tissue_percentage = process_tile(
                    tile_array,
                    threshold=background_threshold,
                    min_tissue_percentage=min_tissue_percentage
                )
                
                if not has_enough_tissue:
                    logger.debug(f"Skipping tile at ({x}, {y}) - insufficient tissue ({tissue_percentage:.2%})")
                    continue
            
            # Find annotations that overlap with this tile
            tile_anns = []
            for ann in annotations:
                # Get annotation bounding box
                bbox = ann.get('bbox', [])
                if not bbox:
                    continue
                
                # Convert bbox to tile coordinates
                ann_x, ann_y, ann_w, ann_h = bbox
                tile_bbox = [
                    ann_x - x,
                    ann_y - y,
                    ann_w,
                    ann_h
                ]
                
                # Check if annotation overlaps with tile
                if (tile_bbox[0] < tile_size and
                    tile_bbox[1] < tile_size and
                    tile_bbox[0] + tile_bbox[2] > 0 and
                    tile_bbox[1] + tile_bbox[3] > 0):
                    
                    # Clip bbox to tile boundaries
                    tile_bbox[0] = max(0, tile_bbox[0])
                    tile_bbox[1] = max(0, tile_bbox[1])
                    tile_bbox[2] = min(tile_bbox[2], tile_size - tile_bbox[0])
                    tile_bbox[3] = min(tile_bbox[3], tile_size - tile_bbox[1])
                    
                    # Create tile annotation
                    tile_ann = {
                        'bbox': tile_bbox,
                        'category': ann.get('category', ''),
                        'segmentation': ann.get('segmentation', [])
                    }
                    tile_anns.append(tile_ann)
            
            if tile_anns:
                # Save tile
                tile_path = os.path.join(output_dir, f"tile_{tile_idx:06d}.png")
                tile.save(tile_path)
                
                # Save visualization if debugging
                if logger.level <= 10:  # DEBUG level
                    vis_path = os.path.join(output_dir, f"tile_{tile_idx:06d}_vis.png")
                    visualize_tissue_detection(
                        tile_array,
                        threshold=background_threshold,
                        output_path=vis_path
                    )
                
                # Add to annotations
                tile_annotations[tile_path] = tile_anns
                
        except Exception as e:
            logger.error(f"Error processing tile at ({x}, {y}): {e}")
            continue
    
    logger.info(f"Processed {len(tile_annotations)} tiles with annotations")
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