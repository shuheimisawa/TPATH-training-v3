import numpy as np
import cv2
from typing import Tuple, Optional
from PIL import Image

def detect_tissue(
    image: np.ndarray,
    threshold: int = 240,
    min_tissue_percentage: float = 0.08
) -> Tuple[bool, float]:
    """
    Detect if an image contains enough tissue.
    
    Args:
        image: RGB image as numpy array
        threshold: Intensity threshold for tissue detection (0-255)
        min_tissue_percentage: Minimum percentage of tissue required (0-1)
        
    Returns:
        Tuple of (has_enough_tissue, tissue_percentage)
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
        
    # Create tissue mask (non-white pixels)
    tissue_mask = gray < threshold
    
    # Calculate tissue percentage
    tissue_percentage = np.mean(tissue_mask)
    
    # Check if enough tissue is present
    has_enough_tissue = tissue_percentage >= min_tissue_percentage
    
    return has_enough_tissue, tissue_percentage

def process_tile(
    tile: np.ndarray,
    threshold: int = 240,
    min_tissue_percentage: float = 0.08
) -> Tuple[bool, float]:
    """
    Process a tile to determine if it contains enough tissue.
    
    Args:
        tile: RGB tile as numpy array
        threshold: Intensity threshold for tissue detection (0-255)
        min_tissue_percentage: Minimum percentage of tissue required (0-1)
        
    Returns:
        Tuple of (should_keep_tile, tissue_percentage)
    """
    has_enough_tissue, tissue_percentage = detect_tissue(
        tile,
        threshold=threshold,
        min_tissue_percentage=min_tissue_percentage
    )
    
    return has_enough_tissue, tissue_percentage

def visualize_tissue_detection(
    image: np.ndarray,
    threshold: int = 240,
    output_path: Optional[str] = None
) -> np.ndarray:
    """
    Create a visualization of tissue detection.
    
    Args:
        image: RGB image as numpy array
        threshold: Intensity threshold for tissue detection (0-255)
        output_path: Optional path to save visualization
        
    Returns:
        Visualization image as numpy array
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Create tissue mask
    tissue_mask = gray < threshold
    
    # Create visualization
    vis = image.copy()
    vis[tissue_mask] = [0, 255, 0]  # Mark tissue in green
    
    # Add text with tissue percentage
    tissue_percentage = np.mean(tissue_mask) * 100
    cv2.putText(
        vis,
        f"Tissue: {tissue_percentage:.1f}%",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    
    # Save if path provided
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    
    return vis 