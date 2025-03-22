# src/evaluation/slide_visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import cv2
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union, Any

from ..utils.slide_io import SlideReader
from ..utils.logger import get_logger


def create_heatmap(
    slide_path: str,
    results: Dict,
    output_path: Optional[str] = None,
    class_colors: Dict[int, Tuple[float, float, float]] = None,
    alpha: float = 0.5,
    resolution_level: int = 3
) -> np.ndarray:
    """Create a heatmap of glomeruli detections on a slide.
    
    Args:
        slide_path: Path to the SVS/WSI file
        results: Dictionary with detection results
        output_path: Path to save the heatmap
        class_colors: Dictionary mapping class indices to RGB colors (0-1 range)
        alpha: Opacity of the heatmap overlay
        resolution_level: Resolution level for the slide thumbnail
        
    Returns:
        Numpy array with the heatmap visualization
    """
    # Default class colors if not provided
    if class_colors is None:
        class_colors = {
            1: (1.0, 0.0, 0.0),   # GN - Red
            2: (0.0, 1.0, 0.0),   # GL - Green
            3: (0.0, 0.0, 1.0)    # GS - Blue
        }
    
    logger = get_logger(name="slide_visualization")
    
    # Open slide
    slide_reader = SlideReader(slide_path)
    
    # Get thumbnail at specified resolution level
    level = min(resolution_level, slide_reader.level_count - 1)
    thumb_width, thumb_height = slide_reader.level_dimensions[level]
    
    # Create figure with the correct aspect ratio
    dpi = 100
    figsize = (thumb_width / dpi, thumb_height / dpi)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Read slide thumbnail
    thumb = slide_reader.read_region((0, 0), level, (thumb_width, thumb_height))
    
    # Display slide thumbnail
    ax.imshow(thumb)
    
    # Calculate scale factor between level 0 and current level
    scale_factor = slide_reader.level_downsamples[level]
    
    # Initialize heatmap data for each class
    heatmap_data = np.zeros((thumb_height, thumb_width, 3))
    
    # Add detection density to heatmap
    for tile_result in results['results']:
        # Get predictions
        prediction = tile_result['prediction']
        
        if len(prediction['boxes']) > 0:
            # Get boxes and labels
            boxes = prediction['boxes']
            labels = prediction['labels']
            scores = prediction['scores']
            
            # Add each detection to heatmap
            for box, label, score in zip(boxes, labels, scores):
                # Get tile coordinates
                tile_x = tile_result['x']
                tile_y = tile_result['y']
                
                # Convert box to level 0 coordinates
                x1, y1, x2, y2 = box
                x1 = tile_x + x1
                y1 = tile_y + y1
                x2 = tile_x + x2
                y2 = tile_y + y2
                
                # Convert to current level coordinates
                x1 = int(x1 / scale_factor)
                y1 = int(y1 / scale_factor)
                x2 = int(x2 / scale_factor)
                y2 = int(y2 / scale_factor)
                
                # Ensure coordinates are within thumbnail bounds
                x1 = max(0, min(x1, thumb_width - 1))
                y1 = max(0, min(y1, thumb_height - 1))
                x2 = max(0, min(x2, thumb_width - 1))
                y2 = max(0, min(y2, thumb_height - 1))
                
                # Get color for this class
                color = class_colors.get(label.item(), (1.0, 1.0, 1.0))
                
                # Draw bounding box on heatmap
                heatmap_data[y1:y2, x1:x2, 0] += color[0] * score.item()
                heatmap_data[y1:y2, x1:x2, 1] += color[1] * score.item()
                heatmap_data[y1:y2, x1:x2, 2] += color[2] * score.item()
                
                # Draw rectangle on the plot
                rect = patches.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=1,
                    edgecolor=color,
                    facecolor='none'
                )
                ax.add_patch(rect)
    
    # Normalize heatmap
    if np.max(heatmap_data) > 0:
        heatmap_data = heatmap_data / np.max(heatmap_data)
    
    # Display heatmap overlay with alpha blending
    ax.imshow(heatmap_data, alpha=alpha)
    
    # Remove axis
    ax.axis('off')
    
    # Save if output path specified
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    
    # Close figure to free memory
    plt.close(fig)
    
    # Close slide reader
    slide_reader.close()
    
    # Convert matplotlib figure to numpy array
    heatmap_with_slide = np.array(thumb) * (1 - alpha) + heatmap_data * 255 * alpha
    heatmap_with_slide = heatmap_with_slide.astype(np.uint8)
    
    return heatmap_with_slide


def create_annotation_overlay(
    slide_path: str,
    results: Dict,
    output_path: Optional[str] = None,
    class_colors: Dict[int, Tuple[int, int, int]] = None,
    line_thickness: int = 2,
    resolution_level: int = 0
) -> np.ndarray:
    """Create an overlay of annotations on a slide.
    
    Args:
        slide_path: Path to the SVS/WSI file
        results: Dictionary with detection results
        output_path: Path to save the overlay
        class_colors: Dictionary mapping class indices to RGB colors
        line_thickness: Thickness of annotation lines
        resolution_level: Resolution level for the slide
        
    Returns:
        Numpy array with the annotation overlay
    """
    # Default class colors if not provided
    # Default class colors if not provided
    if class_colors is None:
        class_colors = {
            1: (0, 255, 0),   # Normal - Green
            2: (255, 0, 0),   # Sclerotic - Red
            3: (0, 0, 255),   # Partially_sclerotic - Blue
            4: (255, 255, 0)  # Uncertain - Yellow
        }
    
    logger = get_logger(name="slide_visualization")
    
    # Open slide
    slide_reader = SlideReader(slide_path)
    
    # Get dimensions at specified resolution level
    level = min(resolution_level, slide_reader.level_count - 1)
    level_width, level_height = slide_reader.level_dimensions[level]
    
    # Create empty overlay
    overlay = np.zeros((level_height, level_width, 4), dtype=np.uint8)
    
    # Calculate scale factor between level 0 and current level
    scale_factor = slide_reader.level_downsamples[level]
    
    # Add detection boxes to overlay
    for tile_result in results['results']:
        # Get predictions
        prediction = tile_result['prediction']
        
        if len(prediction['boxes']) > 0:
            # Get boxes and labels
            boxes = prediction['boxes']
            labels = prediction['labels']
            scores = prediction['scores']
            
            # Add each detection to overlay
            for box, label, score in zip(boxes, labels, scores):
                # Get tile coordinates
                tile_x = tile_result['x']
                tile_y = tile_result['y']
                
                # Convert box to level 0 coordinates
                x1, y1, x2, y2 = box
                x1 = tile_x + x1
                y1 = tile_y + y1
                x2 = tile_x + x2
                y2 = tile_y + y2
                
                # Convert to current level coordinates
                x1 = int(x1 / scale_factor)
                y1 = int(y1 / scale_factor)
                x2 = int(x2 / scale_factor)
                y2 = int(y2 / scale_factor)
                
                # Ensure coordinates are within overlay bounds
                x1 = max(0, min(x1, level_width - 1))
                y1 = max(0, min(y1, level_height - 1))
                x2 = max(0, min(x2, level_width - 1))
                y2 = max(0, min(y2, level_height - 1))
                
                # Get color for this class
                color = class_colors.get(label.item(), (255, 255, 255))
                
                # Draw bounding box on overlay
                cv2.rectangle(
                    overlay,
                    (x1, y1),
                    (x2, y2),
                    (*color, 255),  # RGBA with full opacity
                    line_thickness
                )
                
                # Draw label and score
                label_text = f"{label.item()}: {score.item():.2f}"
                text_size, _ = cv2.getTextSize(
                    label_text,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    1
                )
                
                cv2.rectangle(
                    overlay,
                    (x1, y1 - text_size[1] - 5),
                    (x1 + text_size[0], y1),
                    (*color, 191),  # RGBA with 75% opacity
                    -1  # Filled rectangle
                )
                
                cv2.putText(
                    overlay,
                    label_text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255, 255),  # White text
                    1
                )
    
    # Save if output path specified
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGBA2BGRA))
    
    # Close slide reader
    slide_reader.close()
    
    return overlay


def apply_overlay_to_slide(
    slide_path: str,
    overlay: np.ndarray,
    output_path: str,
    resolution_level: int = 0,
    alpha: float = 0.5
) -> None:
    """Apply an overlay to a slide and save the result.
    
    Args:
        slide_path: Path to the SVS/WSI file
        overlay: RGBA overlay image
        output_path: Path to save the result
        resolution_level: Resolution level for the slide
        alpha: Opacity of the overlay
    """
    logger = get_logger(name="slide_visualization")
    
    # Open slide
    slide_reader = SlideReader(slide_path)
    
    # Get slide dimensions at specified resolution level
    level = min(resolution_level, slide_reader.level_count - 1)
    level_width, level_height = slide_reader.level_dimensions[level]
    
    # Check overlay dimensions
    if overlay.shape[:2] != (level_height, level_width):
        logger.warning(f"Overlay dimensions {overlay.shape[:2]} don't match slide dimensions {(level_height, level_width)}")
        logger.warning("Resizing overlay to match slide dimensions")
        overlay = cv2.resize(overlay, (level_width, level_height))
    
    # Read slide region
    slide_image = slide_reader.read_region((0, 0), level, (level_width, level_height))
    slide_array = np.array(slide_image)
    
    # Convert slide to RGB if needed
    if slide_array.shape[2] == 4:  # RGBA
        slide_array = slide_array[:, :, :3]  # Remove alpha channel
    
    # Check alpha channel in overlay
    if overlay.shape[2] == 4:  # RGBA
        # Extract alpha channel
        alpha_channel = overlay[:, :, 3:4] / 255.0 * alpha
        
        # Convert to proper shape for broadcasting
        alpha_channel = np.repeat(alpha_channel, 3, axis=2)
        
        # Combine slide and overlay
        composite = slide_array * (1.0 - alpha_channel) + overlay[:, :, :3] * alpha_channel
        composite = np.clip(composite, 0, 255).astype(np.uint8)
    else:
        # No alpha channel, use global alpha
        composite = cv2.addWeighted(slide_array, 1.0 - alpha, overlay, alpha, 0)
    
    # Save result
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    Image.fromarray(composite).save(output_path)
    
    # Close slide reader
    slide_reader.close()