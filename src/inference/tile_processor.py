# src/inference/tile_processor.py
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any

from ..utils.logger import get_logger
from ..utils.slide_io import SlideReader, TileExtractor, TileStitcher
from ..evaluation.visualization import visualize_prediction


class TileProcessor:
    """Process tiles from a whole slide image through the model."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        tile_size: int = 1024,
        overlap: int = 0,
        batch_size: int = 1,
        confidence_threshold: float = 0.5
    ):
        """Initialize the tile processor.
        
        Args:
            model: Model to use for inference
            device: Device to use for inference
            tile_size: Size of the tiles (width and height)
            overlap: Overlap between adjacent tiles in pixels
            batch_size: Batch size for inference
            confidence_threshold: Confidence threshold for detections
        """
        self.model = model
        self.device = device
        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        
        # Set model to evaluation mode
        self.model.eval()
        
        self.logger = get_logger(name="tile_processor")
        self.logger.info(f"Initialized tile processor")
        self.logger.info(f"Tile size: {tile_size}x{tile_size}")
        self.logger.info(f"Overlap: {overlap}")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"Confidence threshold: {confidence_threshold}")
    
    def process_tile(self, tile_tensor: torch.Tensor) -> Dict:
        """Process a single tile.
        
        Args:
            tile_tensor: Tensor of the tile image
            
        Returns:
            Dictionary with detection results
        """
        with torch.no_grad():
            # Add batch dimension if needed
            if len(tile_tensor.shape) == 3:
                tile_tensor = tile_tensor.unsqueeze(0)
                
            # Move to device
            tile_tensor = tile_tensor.to(self.device)
            
            # Run model
            predictions = self.model([tile_tensor])
            
            # Get first prediction (single tile)
            prediction = predictions[0]
            
            # Filter by confidence
            keep = prediction['scores'] > self.confidence_threshold
            
            filtered_prediction = {
                'boxes': prediction['boxes'][keep],
                'labels': prediction['labels'][keep],
                'scores': prediction['scores'][keep],
                'masks': prediction['masks'][keep] if 'masks' in prediction else None
            }
            
            # Move back to CPU
            for k, v in filtered_prediction.items():
                if isinstance(v, torch.Tensor):
                    filtered_prediction[k] = v.cpu()
            
            return filtered_prediction
    
    def process_tiles_batch(self, tile_tensors: List[torch.Tensor]) -> List[Dict]:
        """Process a batch of tiles.
        
        Args:
            tile_tensors: List of tile tensors
            
        Returns:
            List of dictionaries with detection results
        """
        if not tile_tensors:
            return []
            
        with torch.no_grad():
            # Move to device
            tile_tensors = [t.to(self.device) for t in tile_tensors]
            
            # Run model
            predictions = self.model(tile_tensors)
            
            # Filter by confidence
            filtered_predictions = []
            for prediction in predictions:
                keep = prediction['scores'] > self.confidence_threshold
                
                filtered_prediction = {
                    'boxes': prediction['boxes'][keep],
                    'labels': prediction['labels'][keep],
                    'scores': prediction['scores'][keep],
                    'masks': prediction['masks'][keep] if 'masks' in prediction else None
                }
                
                # Move back to CPU
                for k, v in filtered_prediction.items():
                    if isinstance(v, torch.Tensor):
                        filtered_prediction[k] = v.cpu()
                
                filtered_predictions.append(filtered_prediction)
            
            return filtered_predictions
    
    def visualize_tile_prediction(
        self,
        tile_image: np.ndarray,
        prediction: Dict,
        class_names: List[str] = ['Normal', 'Sclerotic', 'Partially_sclerotic', 'Uncertain'],
        output_path: Optional[str] = None
    ) -> Image.Image:
        """Visualize prediction on a tile.
        
        Args:
            tile_image: Tile image as numpy array
            prediction: Dictionary with detection results
            class_names: List of class names
            output_path: Path to save the visualization
            
        Returns:
            PIL Image with visualization
        """
        return visualize_prediction(
            image=tile_image,
            prediction=prediction,
            class_names=class_names,
            save_path=output_path
        )