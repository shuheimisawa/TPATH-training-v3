# scripts/visualize_qupath_annotations.py
import os
import argparse
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import cv2
from tqdm import tqdm

from src.utils.logger import get_logger
from src.utils.slide_io import SlideReader
from src.utils.io import load_json


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize QuPath annotations on slides')
    
    parser.add_argument('--slides-dir', type=str, required=True,
                        help='Path to directory with slide files')
    parser.add_argument('--annotations-dir', type=str, required=True,
                        help='Path to directory with converted annotation JSON files')
    parser.add_argument('--output-dir', type=str, default='annotation_visualizations',
                        help='Path to output directory for visualizations')
    parser.add_argument('--level', type=int, default=3,
                        help='Magnification level for visualization (higher = lower resolution)')
    parser.add_argument('--draw-polygons', action='store_true',
                        help='Draw polygons instead of bounding boxes')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Alpha transparency for annotation overlays')
    
    return parser.parse_args()


def find_slide_path(slides_dir, slide_name):
    """Find the path to a slide file based on its name."""
    # Try different extensions
    slide_extensions = ['.svs', '.ndpi', '.tif', '.tiff']
    
    for ext in slide_extensions:
        potential_path = os.path.join(slides_dir, f"{slide_name}{ext}")
        if os.path.exists(potential_path):
            return potential_path
    
    # If exact match not found, look for partial matches
    for ext in slide_extensions:
        for file_path in glob.glob(os.path.join(slides_dir, f"*{ext}")):
            basename = os.path.splitext(os.path.basename(file_path))[0]
            if slide_name in basename or basename in slide_name:
                return file_path
    
    return None


def visualize_annotations(slide_path, annotation_data, output_path, level=3, 
                         draw_polygons=False, alpha=0.3):
    """Visualize annotations on a slide.
    
    Args:
        slide_path: Path to the slide file
        annotation_data: Annotation data in project format
        output_path: Path to save the visualization
        level: Magnification level for visualization
        draw_polygons: Whether to draw polygon segmentations instead of bounding boxes
        alpha: Alpha transparency for annotation overlays
    """
    logger = get_logger(name="visualize_annotations")
    
    try:
        # Open slide
        slide_reader = SlideReader(slide_path)
        
        # Get level dimensions
        level = min(level, slide_reader.level_count - 1)
        level_width, level_height = slide_reader.level_dimensions[level]
        
        # Read region at specified level
        slide_image = slide_reader.read_region((0, 0), level, (level_width, level_height))
        
        # Convert to numpy array
        slide_np = np.array(slide_image)
        
        # Create overlay image (transparent)
        overlay = Image.new('RGBA', (level_width, level_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        
        # Calculate scale factor between level 0 and current level
        scale_factor = slide_reader.level_downsamples[level]
        
        # Define colors for each class
        # Define colors for each class
        class_colors = {
            'Normal': (0, 255, 0, int(255 * alpha)),       # Green
            'Sclerotic': (255, 0, 0, int(255 * alpha)),    # Red
            'Partially_sclerotic': (0, 0, 255, int(255 * alpha)),  # Blue
            'Uncertain': (255, 255, 0, int(255 * alpha)),  # Yellow
            'background': (200, 200, 200, int(128 * alpha))  # Gray for unknown
        }
        
        # Extract annotations for the slide
        slide_key = next(iter(annotation_data.keys()))
        annotations = annotation_data[slide_key].get('annotations', [])
        
        logger.info(f"Drawing {len(annotations)} annotations")
        
        # Draw each annotation
        for annotation in annotations:
            category = annotation.get('category', 'background')
            color = class_colors.get(category, class_colors['background'])
            
            # Get bounding box
            bbox = annotation.get('bbox', [0, 0, 0, 0])
            
            # Scale to current level
            scaled_bbox = [
                bbox[0] / scale_factor,
                bbox[1] / scale_factor,
                bbox[2] / scale_factor,
                bbox[3] / scale_factor
            ]
            
            x, y, width, height = scaled_bbox
            
            if draw_polygons and 'segmentation' in annotation:
                # Draw polygon segmentation
                for segment in annotation['segmentation']:
                    # Scale polygon points to current level
                    scaled_points = []
                    for i in range(0, len(segment), 2):
                        if i + 1 < len(segment):
                            px = segment[i] / scale_factor
                            py = segment[i + 1] / scale_factor
                            scaled_points.append((px, py))
                    
                    # Draw polygon
                    if len(scaled_points) >= 3:
                        draw.polygon(scaled_points, fill=color, outline=color[:3] + (255,))
            else:
                # Draw bounding box
                draw.rectangle(
                    [(x, y), (x + width, y + height)],
                    fill=color,
                    outline=color[:3] + (255,)
                )
                
                # Add category label
                draw.text((x, y - 10), category, fill=color[:3] + (255,))
        
        # Composite the overlay onto the slide image
        slide_with_annotations = Image.alpha_composite(
            slide_image.convert('RGBA'), 
            overlay
        ).convert('RGB')
        
        # Save the visualization
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        slide_with_annotations.save(output_path)
        
        # Close slide
        slide_reader.close()
        
        logger.info(f"Saved visualization to {output_path}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error visualizing annotations for {slide_path}: {e}")
        return False


def main():
    # Parse arguments
    args = parse_args()
    
    # Create logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(
        name="visualize_annotations",
        log_file=os.path.join(args.output_dir, "visualize_annotations.log")
    )
    
    # Find annotation files
    annotation_files = glob.glob(os.path.join(args.annotations_dir, "*_annotations.json"))
    logger.info(f"Found {len(annotation_files)} annotation files")
    
    # Process each annotation file
    for annotation_file in tqdm(annotation_files, desc="Visualizing annotations"):
        # Get slide name from annotation filename
        slide_name = os.path.basename(annotation_file).replace("_annotations.json", "")
        
        # Find corresponding slide file
        slide_path = find_slide_path(args.slides_dir, slide_name)
        
        if not slide_path:
            logger.warning(f"Could not find slide for {slide_name}, skipping visualization")
            continue
        
        # Load annotations
        try:
            annotation_data = load_json(annotation_file)
        except Exception as e:
            logger.error(f"Error loading annotations from {annotation_file}: {e}")
            continue
        
        # Create output path
        output_path = os.path.join(args.output_dir, f"{slide_name}_annotations.png")
        
        # Visualize annotations
        success = visualize_annotations(
            slide_path=slide_path,
            annotation_data=annotation_data,
            output_path=output_path,
            level=args.level,
            draw_polygons=args.draw_polygons,
            alpha=args.alpha
        )
        
        if not success:
            logger.warning(f"Failed to visualize annotations for {slide_name}")
    
    logger.info("Visualization completed")


if __name__ == '__main__':
    main()