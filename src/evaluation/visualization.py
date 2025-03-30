import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
from typing import Dict, List, Tuple, Optional, Union, Any

def visualize_prediction(
    image: np.ndarray,
    prediction: Dict,
    class_names: List[str] = ['Normal', 'Sclerotic', 'Partially_sclerotic', 'Uncertain'],
    save_path: Optional[str] = None,
    show_masks: bool = True,
    show_boxes: bool = True,
    show_scores: bool = True,
    figure_size: Tuple[int, int] = (12, 12),
    title: Optional[str] = None
) -> Image.Image:
    """Visualize model prediction on an image."""
    # Use Agg backend explicitly
    import matplotlib
    matplotlib.use('Agg')  # Use the Agg backend
    
    # Make a copy of the image to avoid modifying the original
    img = image.copy()
    
    # Create figure and axis
    fig, ax = plt.subplots(1, figsize=figure_size)
    
    # Display image
    ax.imshow(img)
    
    # Hide axis
    ax.axis('off')
    
    # Set title if provided
    if title:
        ax.set_title(title, fontsize=14)
    
    # Define colors for each class
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Get prediction components
    boxes = prediction.get('boxes', None)
    labels = prediction.get('labels', None)
    scores = prediction.get('scores', None)
    masks = prediction.get('masks', None)
    
    # Check if we have valid predictions
    if boxes is None or len(boxes) == 0:
        # No predictions, just show the image
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=100)
        plt.close(fig)
        return Image.fromarray(img)
    
    # Convert tensors to numpy arrays if needed
    if isinstance(boxes, torch.Tensor):
        boxes = boxes.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    if isinstance(scores, torch.Tensor):
        scores = scores.cpu().numpy()
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    
    # Visualize each prediction
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        # Get class index (subtract 1 if classes start from 1)
        class_idx = int(label) - 1 if int(label) > 0 else 0
        
        # Get class name
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {label}"
        
        # Get color for this class
        color = colors[class_idx % 10]
        
        # Draw masks if available and requested
        if show_masks and masks is not None and i < len(masks):
            # Get mask for this instance
            mask = masks[i]
            
            # If mask has multiple channels, take the first one
            if len(mask.shape) > 2:
                mask = mask[0] if mask.shape[0] == 1 else mask
            
            # Apply mask as overlay
            colored_mask = np.zeros_like(img)
            mask_bool = mask > 0.5
            colored_mask[mask_bool] = color[:3] * 255
            
            # Blend mask with image
            alpha = 0.4
            mask_area = mask_bool.astype(np.uint8)
            img = cv2.addWeighted(img, 1, colored_mask.astype(np.uint8), alpha, 0)
        
        # Draw bounding box if requested
        if show_boxes:
            # Get box coordinates (ensure they are integers)
            x1, y1, x2, y2 = map(int, box)
            
            # Create rectangle patch
            rect = patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            
            # Add rectangle to plot
            ax.add_patch(rect)
        
        # Show label and score if requested
        if show_scores:
            score_text = f"{class_name}: {score:.2f}"
            ax.text(
                box[0], box[1] - 5,
                score_text,
                color='white', fontsize=10, weight='bold',
                bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=2)
            )
    
    # Save figure if path provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1, dpi=100)
    
    # Convert figure to PIL Image
    plt.tight_layout()
    
    # Save to a BytesIO object and then to PIL Image
    from io import BytesIO
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    
    # Close figure to free memory
    plt.close(fig)
    
    # Return as PIL Image
    return Image.open(buf)

def visualize_batch_predictions(
    images: List[np.ndarray],
    predictions: List[Dict],
    class_names: List[str] = ['Normal', 'Sclerotic', 'Partially_sclerotic', 'Uncertain'],
    save_path: Optional[str] = None,
    max_images: int = 16,
    grid_size: Optional[Tuple[int, int]] = None,
    show_masks: bool = True,
    show_boxes: bool = True,
    show_scores: bool = True
) -> Image.Image:
    """Visualize predictions for a batch of images.
    
    Args:
        images: List of input images as numpy arrays
        predictions: List of prediction dictionaries
        class_names: List of class names
        save_path: Path to save the visualization
        max_images: Maximum number of images to visualize
        grid_size: Optional size of the grid (rows, cols)
        show_masks: Whether to show masks
        show_boxes: Whether to show bounding boxes
        show_scores: Whether to show confidence scores
        
    Returns:
        Visualization as PIL Image
    """
    # Limit number of images
    n_images = min(len(images), max_images)
    
    # Determine grid size if not provided
    if grid_size is None:
        grid_cols = min(4, n_images)
        grid_rows = (n_images + grid_cols - 1) // grid_cols
        grid_size = (grid_rows, grid_cols)
    else:
        grid_rows, grid_cols = grid_size
    
    # Create figure and axes
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(4 * grid_cols, 4 * grid_rows))
    
    # Flatten axes for easier indexing if we have a grid
    if grid_rows == 1 and grid_cols == 1:
        axes = np.array([axes])
    elif grid_rows == 1 or grid_cols == 1:
        axes = axes.ravel()
    
    # Define colors for each class
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    # Visualize each image
    for i in range(grid_rows * grid_cols):
        if i < n_images:
            # Get image and prediction
            img = images[i]
            pred = predictions[i]
            
            # Get current axis
            ax = axes.flat[i]
            
            # Display image
            ax.imshow(img)
            
            # Hide axis
            ax.axis('off')
            
            # Get prediction components
            boxes = pred.get('boxes', None)
            labels = pred.get('labels', None)
            scores = pred.get('scores', None)
            masks = pred.get('masks', None)
            
            # Check if we have valid predictions
            if boxes is not None and len(boxes) > 0:
                # Convert tensors to numpy arrays if needed
                if isinstance(boxes, torch.Tensor):
                    boxes = boxes.cpu().numpy()
                if isinstance(labels, torch.Tensor):
                    labels = labels.cpu().numpy()
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                if isinstance(masks, torch.Tensor):
                    masks = masks.cpu().numpy()
                
                # Visualize each prediction
                for j, (box, label, score) in enumerate(zip(boxes, labels, scores)):
                    # Get class index (subtract 1 if classes start from 1)
                    class_idx = int(label) - 1 if int(label) > 0 else 0
                    
                    # Get class name
                    class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {label}"
                    
                    # Get color for this class
                    color = colors[class_idx % 10]
                    
                    # Draw masks if available and requested
                    if show_masks and masks is not None and j < len(masks):
                        # Get mask for this instance
                        mask = masks[j]
                        
                        # If mask has multiple channels, take the first one
                        if len(mask.shape) > 2:
                            mask = mask[0] if mask.shape[0] == 1 else mask
                        
                        # Apply mask as overlay with alpha blending
                        mask_bool = mask > 0.5
                        
                        # Show mask outline
                        ax.contour(
                            mask_bool, 
                            levels=[0.5], 
                            colors=[color], 
                            alpha=0.7, 
                            linewidths=2
                        )
                    
                    # Draw bounding box if requested
                    if show_boxes:
                        # Get box coordinates (ensure they are integers)
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Create rectangle patch
                        rect = patches.Rectangle(
                            (x1, y1), x2 - x1, y2 - y1,
                            linewidth=2, edgecolor=color, facecolor='none'
                        )
                        
                        # Add rectangle to plot
                        ax.add_patch(rect)
                    
                    # Show label and score if requested
                    if show_scores:
                        score_text = f"{class_name}: {score:.2f}"
                        ax.text(
                            box[0], box[1] - 5,
                            score_text,
                            color='white', fontsize=8, weight='bold',
                            bbox=dict(facecolor=color, alpha=0.8, edgecolor='none', pad=1)
                        )
        else:
            # Hide unused subplots
            axes.flat[i].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=100)
    
    # Convert figure to PIL Image
    fig.canvas.draw()
    
    # Convert canvas to numpy array
    img_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_np = img_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    # Close figure to free memory
    plt.close(fig)
    
    # Return as PIL Image
    return Image.fromarray(img_np)