# scripts/visualize_tile_annotations.py
import os
import argparse
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm
import random

from src.utils.logger import get_logger
from src.utils.io import load_json


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize tile annotations')
    
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to processed data directory (e.g., data/train)')
    parser.add_argument('--output-dir', type=str, default='tile_visualizations',
                        help='Path to output directory for visualizations')
    parser.add_argument('--draw-polygons', action='store_true',
                        help='Draw polygons instead of bounding boxes')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Alpha transparency for annotation overlays')
    parser.add_argument('--sample', type=int, default=0,
                        help='Number of random tiles to sample (0 = all)')
    
    return parser.parse_args()


def visualize_tile_annotations(tile_path, annotations, output_path, draw_polygons=False, alpha=0.3):
    """Visualize annotations on a tile.
    
    Args:
        tile_path: Path to the tile image file
        annotations: List of annotations for this tile
        output_path: Path to save the visualization
        draw_polygons: Whether to draw polygon segmentations instead of bounding boxes
        alpha: Alpha transparency for annotation overlays
    """
    # Load tile image
    tile_image = Image.open(tile_path).convert('RGBA')
    width, height = tile_image.size
    
    # Create overlay image (transparent)
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    
    # Define colors for each class
    class_colors = {
        'Normal': (0, 255, 0, int(255 * alpha)),       # Green
        'Sclerotic': (255, 0, 0, int(255 * alpha)),    # Red
        'Partially_sclerotic': (0, 0, 255, int(255 * alpha)),  # Blue
        'Uncertain': (255, 255, 0, int(255 * alpha)),  # Yellow
        'background': (200, 200, 200, int(128 * alpha))  # Gray for unknown
    }
    
    # Draw each annotation
    for annotation in annotations:
        category = annotation.get('category', 'background')
        color = class_colors.get(category, class_colors['background'])
        
        # Get bounding box
        bbox = annotation.get('bbox', [0, 0, 0, 0])
        x, y, width, height = bbox
        
        if draw_polygons and 'segmentation' in annotation:
            # Draw polygon segmentation
            for segment in annotation['segmentation']:
                # Group points into (x, y) pairs
                points = []
                for i in range(0, len(segment), 2):
                    if i + 1 < len(segment):
                        px = segment[i]
                        py = segment[i + 1]
                        points.append((px, py))
                
                # Draw polygon
                if len(points) >= 3:
                    draw.polygon(points, fill=color, outline=color[:3] + (255,))
        else:
            # Draw bounding box
            draw.rectangle(
                [(x, y), (x + width, y + height)],
                fill=color,
                outline=color[:3] + (255,)
            )
            
            # Add category label
            draw.text((x, y - 10), category, fill=color[:3] + (255,))
    
    # Composite the overlay onto the tile image
    tile_with_annotations = Image.alpha_composite(
        tile_image,
        overlay
    ).convert('RGB')
    
    # Save the visualization
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tile_with_annotations.save(output_path)
    
    return True


def main():
    # Parse arguments
    args = parse_args()
    
    # Create logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(
        name="visualize_tile_annotations",
        log_file=os.path.join(args.output_dir, "visualize_tile_annotations.log")
    )
    
    # Load annotations
    annotations_path = os.path.join(args.data_dir, f"{os.path.basename(args.data_dir)}_annotations.json")
    if not os.path.exists(annotations_path):
        logger.error(f"Annotations file not found: {annotations_path}")
        return
    
    try:
        annotations = load_json(annotations_path)
        logger.info(f"Loaded annotations for {len(annotations)} tiles")
    except Exception as e:
        logger.error(f"Error loading annotations: {e}")
        return
    
    # Get tiles to visualize
    tile_ids = list(annotations.keys())
    
    if args.sample > 0 and args.sample < len(tile_ids):
        # Sample random tiles
        sampled_tile_ids = random.sample(tile_ids, args.sample)
        logger.info(f"Sampled {len(sampled_tile_ids)} random tiles")
    else:
        sampled_tile_ids = tile_ids
        logger.info(f"Visualizing all {len(sampled_tile_ids)} tiles")
    
    # Visualize each tile
    success_count = 0
    for tile_id in tqdm(sampled_tile_ids, desc="Visualizing tiles"):
        # Get file path and annotations
        tile_info = annotations[tile_id]
        tile_path = os.path.join(args.data_dir, tile_info['file_path'])
        tile_annotations = tile_info.get('annotations', [])
        
        if not os.path.exists(tile_path):
            logger.warning(f"Tile image not found: {tile_path}")
            continue
        
        # Create output path
        output_path = os.path.join(args.output_dir, f"{tile_id}_annotated.png")
        
        # Visualize annotations
        success = visualize_tile_annotations(
            tile_path=tile_path,
            annotations=tile_annotations,
            output_path=output_path,
            draw_polygons=args.draw_polygons,
            alpha=args.alpha
        )
        
        if success:
            success_count += 1
        else:
            logger.warning(f"Failed to visualize annotations for {tile_id}")
    
    logger.info(f"Successfully visualized {success_count} tiles")
    
    # Generate summary visualization (grid of random annotated tiles)
    if success_count > 0:
        try:
            # Find visualized tiles
            visualized_tiles = glob.glob(os.path.join(args.output_dir, "*_annotated.png"))
            
            if visualized_tiles:
                # Sample tiles for grid
                grid_size = min(5, int(np.sqrt(len(visualized_tiles))))
                grid_tiles = random.sample(visualized_tiles, min(grid_size * grid_size, len(visualized_tiles)))
                
                # Create grid
                fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
                
                for i, tile_path in enumerate(grid_tiles):
                    row = i // grid_size
                    col = i % grid_size
                    
                    # Load tile
                    tile = Image.open(tile_path)
                    
                    # Display in grid
                    if grid_size > 1:
                        axes[row, col].imshow(np.array(tile))
                        axes[row, col].set_title(os.path.basename(tile_path).replace("_annotated.png", ""))
                        axes[row, col].axis('off')
                    else:
                        axes[i].imshow(np.array(tile))
                        axes[i].set_title(os.path.basename(tile_path).replace("_annotated.png", ""))
                        axes[i].axis('off')
                
                # Adjust layout
                plt.tight_layout()
                
                # Save grid
                grid_path = os.path.join(args.output_dir, "tile_grid.png")
                plt.savefig(grid_path, dpi=300)
                plt.close()
                
                logger.info(f"Saved tile grid to {grid_path}")
        
        except Exception as e:
            logger.error(f"Error creating tile grid: {e}")
    
    logger.info("Visualization completed")


if __name__ == '__main__':
    main()