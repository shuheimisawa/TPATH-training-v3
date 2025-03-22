# src/inference/slide_inference.py
import os
import torch
import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Union, Any

from ..utils.logger import get_logger
from ..utils.slide_io import SlideReader, TileExtractor, TileStitcher
from ..data.slide_dataset import SlideTileDataset
from .tile_processor import TileProcessor
from ..evaluation.visualization import visualize_prediction


class SlideInference:
    """Run inference on a whole slide image."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        tile_size: int = 1024,
        overlap: int = 256,
        batch_size: int = 4,
        level: int = 0,
        confidence_threshold: float = 0.5,
        filter_background: bool = True,
        background_threshold: int = 220
    ):
        """Initialize the slide inference.
        
        Args:
            model: Model to use for inference
            device: Device to use for inference
            tile_size: Size of tiles (width and height)
            overlap: Overlap between adjacent tiles in pixels
            batch_size: Batch size for inference
            level: Magnification level to process (0 is highest)
            confidence_threshold: Confidence threshold for detections
            filter_background: Whether to filter out background tiles
            background_threshold: Threshold for background detection
        """
        # Create tile processor first
        self.tile_processor = TileProcessor(
            model=model,
            device=device,
            tile_size=tile_size,
            overlap=overlap,
            batch_size=batch_size,
            confidence_threshold=confidence_threshold
        )
        
        # Store parameters
        self.model = model
        self.device = device
        self.tile_size = tile_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.level = level
        self.confidence_threshold = confidence_threshold
        self.filter_background = filter_background
        self.background_threshold = background_threshold
        
        # Set model to evaluation mode
        self.model.eval()
        
        self.logger = get_logger(name="slide_inference")
        self.logger.info(f"Initialized slide inference")
        self.logger.info(f"Tile size: {tile_size}x{tile_size}")
        self.logger.info(f"Overlap: {overlap}")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"Level: {level}")
        self.logger.info(f"Confidence threshold: {confidence_threshold}")
    
    def run_inference(self, slide_path: str, output_dir: Optional[str] = None) -> Dict:
        """Run inference on a whole slide image.
        
        Args:
            slide_path: Path to the SVS/WSI file
            output_dir: Directory to save visualization results
            
        Returns:
            Dictionary with detection results for all tiles
        """
        self.logger.info(f"Running inference on {slide_path}")
        
        try:
            # Create slide dataset
            dataset = SlideTileDataset(
                slide_path=slide_path,
                tile_size=self.tile_size,
                overlap=self.overlap,
                level=self.level,
                transform=self._get_transform(),
                return_coordinates=True,
                filter_background=self.filter_background,
                background_threshold=self.background_threshold
            )
            
            # Create data loader
            from torch.utils.data import DataLoader
            
            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4
            )
            
            # Run inference
            all_results = []
            
            self.logger.info(f"Processing {len(dataset)} tiles")
            
            for batch in tqdm(loader, desc="Processing tiles"):
                # Get batch data
                batch_images = batch['image']
                batch_info = {k: batch[k] for k in batch.keys() if k != 'image'}
                
                # Process batch
                batch_predictions = self.tile_processor.process_tiles_batch(batch_images)
                
                # Adjust coordinates for predictions
                for i, prediction in enumerate(batch_predictions):
                    # Create tile result with coordinates
                    tile_result = {
                        'prediction': prediction,
                        'x': batch_info['x'][i].item(),
                        'y': batch_info['y'][i].item(),
                        'width': batch_info['width'][i].item(),
                        'height': batch_info['height'][i].item(),
                        'level': batch_info['level'][i].item(),
                        'index': (batch_info['index'][0][i].item(), batch_info['index'][1][i].item())
                    }
                    
                    # Add to results
                    all_results.append(tile_result)
                    
                    # Visualize if output directory specified
                    if output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                        
                        # Convert tensor to numpy for visualization
                        tile_np = batch_images[i].permute(1, 2, 0).cpu().numpy()
                        
                        # Denormalize if needed
                        tile_np = tile_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                        tile_np = (tile_np * 255).astype(np.uint8)
                        
                        # Create output path
                        tile_x, tile_y = tile_result['index']
                        output_path = os.path.join(output_dir, f"tile_{tile_x}_{tile_y}.png")
                        
                        # Visualize
                        self.tile_processor.visualize_tile_prediction(
                            tile_image=tile_np,
                            prediction=prediction,
                            output_path=output_path
                        )
            
            self.logger.info(f"Inference completed")
            
            # Close slide reader
            dataset.slide_reader.close()
            
            return {
                'slide_path': slide_path,
                'results': all_results
            }
        except Exception as e:
            self.logger.error(f"Error running inference on {slide_path}: {e}")
            # Return empty results
            return {
                'slide_path': slide_path,
                'results': [],
                'error': str(e)
            }
    
    def visualize_slide_results(
        self,
        slide_path: str,
        results: Dict,
        output_path: Optional[str] = None,
        draw_detections: bool = True,
        class_colors: Dict[int, Tuple[int, int, int]] = None,
        class_names: List[str] = ['Normal', 'Sclerotic', 'Partially_sclerotic', 'Uncertain']
    ) -> Image.Image:
        """Visualize results on the whole slide.
        
        Args:
            slide_path: Path to the SVS/WSI file
            results: Dictionary with detection results
            output_path: Path to save the visualization
            draw_detections: Whether to draw detection boxes/masks
            class_colors: Dictionary mapping class indices to RGB colors
            class_names: List of class names
            
        Returns:
            PIL Image with visualization
        """
        # Default class colors if not provided
        if class_colors is None:
            class_colors = {
                1: (0, 255, 0),   # Normal - Green
                2: (255, 0, 0),   # Sclerotic - Red
                3: (0, 0, 255),   # Partially_sclerotic - Blue
                4: (255, 255, 0)  # Uncertain - Yellow
            }
        
        try:
            # Open slide
            slide_reader = SlideReader(slide_path)
            
            # Create thumbnail
            thumb_size = (1024, 1024)
            thumbnail = slide_reader.get_slide_thumbnail(thumb_size)
            
            # Calculate scale factor between level 0 and thumbnail
            scale_x = thumbnail.width / slide_reader.width
            scale_y = thumbnail.height / slide_reader.height
            
            # Convert to OpenCV format for drawing
            thumb_np = np.array(thumbnail)
            thumb_np = cv2.cvtColor(thumb_np, cv2.COLOR_RGB2BGR)
            
            # Draw tiles and detections
            if draw_detections and 'results' in results:
                for tile_result in results['results']:
                    # Get tile coordinates
                    tile_x = tile_result['x']
                    tile_y = tile_result['y']
                    tile_width = tile_result['width']
                    tile_height = tile_result['height']
                    
                    # Scale to thumbnail
                    thumb_x = int(tile_x * scale_x)
                    thumb_y = int(tile_y * scale_y)
                    thumb_width = int(tile_width * scale_x)
                    thumb_height = int(tile_height * scale_y)
                    
                    # Draw tile boundary
                    cv2.rectangle(
                        thumb_np,
                        (thumb_x, thumb_y),
                        (thumb_x + thumb_width, thumb_y + thumb_height),
                        (255, 255, 255),
                        1
                    )
                    
                    # Get predictions
                    prediction = tile_result['prediction']
                    
                    # Draw detection boxes
                    if len(prediction['boxes']) > 0:
                        boxes = prediction['boxes']
                        labels = prediction['labels']
                        scores = prediction['scores']
                        
                        for box, label, score in zip(boxes, labels, scores):
                            # Scale box to thumbnail
                            x1, y1, x2, y2 = box
                            x1 = int((tile_x + x1) * scale_x)
                            y1 = int((tile_y + y1) * scale_y)
                            x2 = int((tile_x + x2) * scale_x)
                            y2 = int((tile_y + y2) * scale_y)
                            
                            # Get color for this class
                            color = class_colors.get(label.item(), (255, 255, 255))
                            color = (color[2], color[1], color[0])  # Convert RGB to BGR
                            
                            # Draw box
                            cv2.rectangle(
                                thumb_np,
                                (x1, y1),
                                (x2, y2),
                                color,
                                2
                            )
                            
                            # Get label text
                            label_idx = label.item()
                            label_text = class_names[label_idx - 1] if 0 <= label_idx - 1 < len(class_names) else f"Class {label_idx}"
                            
                            # Draw label and score
                            label_score_text = f"{label_text}: {score.item():.2f}"
                            cv2.putText(
                                thumb_np,
                                label_score_text,
                                (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                color,
                                1
                            )
            
            # Convert back to PIL Image
            thumb_np = cv2.cvtColor(thumb_np, cv2.COLOR_BGR2RGB)
            visualization = Image.fromarray(thumb_np)
            
            # Save if output path specified
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                visualization.save(output_path)
            
            # Close slide reader
            slide_reader.close()
            
            return visualization
        except Exception as e:
            self.logger.error(f"Error visualizing slide results: {e}")
            # Return a blank image as fallback
            blank = Image.new('RGB', (1024, 1024), color=(255, 255, 255))
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                blank.save(output_path)
            return blank
    
    def stitch_tile_predictions(
        self,
        slide_path: str,
        results: Dict,
        output_path: Optional[str] = None,
        draw_detections: bool = True,
        class_colors: Dict[int, Tuple[int, int, int]] = None
    ) -> Image.Image:
        """Stitch tile predictions into a whole slide image.
        
        Args:
            slide_path: Path to the SVS/WSI file
            results: Dictionary with detection results
            output_path: Path to save the stitched image
            draw_detections: Whether to draw detection boxes/masks
            class_colors: Dictionary mapping class indices to RGB colors
            
        Returns:
            PIL Image with stitched predictions
        """
        # Default class colors if not provided
        
        if class_colors is None:
            class_colors = {
                1: (0, 255, 0),     # Normal - Green
                2: (255, 0, 0),     # Sclerotic - Red
                3: (0, 0, 255),     # Partially_sclerotic - Blue
                4: (255, 255, 0)    # Uncertain - Yellow
            }
        
        try:
            # Open slide
            slide_reader = SlideReader(slide_path)
            
            # Get dimensions for current level
            level_width, level_height = slide_reader.level_dimensions[self.level]
            
            # Create tile stitcher
            stitcher = TileStitcher(
                output_width=level_width,
                output_height=level_height
            )
            
            # Process each tile
            if 'results' in results:
                for tile_result in tqdm(results['results'], desc="Stitching tiles"):
                    # Get tile coordinates
                    tile_x = tile_result['x']
                    tile_y = tile_result['y']
                    tile_width = tile_result['width']
                    tile_height = tile_result['height']
                    
                    # Get original tile image
                    tile_image = slide_reader.get_tile(tile_x, tile_y, tile_width, tile_height, self.level)
                    
                    # Draw detections if requested
                    if draw_detections:
                        # Get predictions
                        prediction = tile_result['prediction']
                        
                        # Convert to numpy for drawing
                        tile_np = np.array(tile_image)
                        
                        # Draw detection boxes
                        if len(prediction['boxes']) > 0:
                            boxes = prediction['boxes']
                            labels = prediction['labels']
                            scores = prediction['scores']
                            masks = prediction.get('masks', None)
                            
                            # Draw masks if available
                            if masks is not None and len(masks) > 0:
                                for mask, label, score in zip(masks, labels, scores):
                                    # Get color for this class
                                    color = class_colors.get(label.item(), (255, 255, 255))
                                    
                                    # Create mask overlay
                                    mask_np = mask.squeeze().numpy()
                                    mask_overlay = np.zeros((*tile_np.shape[:2], 3), dtype=np.uint8)
                                    mask_overlay[mask_np > 0.5] = color
                                    
                                    # Blend with original image
                                    alpha = 0.5
                                    tile_np = cv2.addWeighted(
                                        tile_np,
                                        1 - alpha,
                                        mask_overlay,
                                        alpha,
                                        0
                                    )
                            
                            # Draw boxes
                            for box, label, score in zip(boxes, labels, scores):
                                # Get coordinates
                                x1, y1, x2, y2 = box.int().tolist()
                                
                                # Get color for this class
                                color = class_colors.get(label.item(), (255, 255, 255))
                                color_bgr = (color[2], color[1], color[0])  # RGB to BGR
                                
                                # Draw box
                                cv2.rectangle(
                                    tile_np,
                                    (x1, y1),
                                    (x2, y2),
                                    color_bgr,
                                    2
                                )
                                
                                # Draw label and score
                                label_text = f"{label.item()}: {score.item():.2f}"
                                cv2.putText(
                                    tile_np,
                                    label_text,
                                    (x1, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5,
                                    color_bgr,
                                    1
                                )
                        
                        # Convert back to PIL Image
                        tile_image = Image.fromarray(tile_np)
                    
                    # Add to stitcher
                    stitcher.add_tile(tile_image, tile_x, tile_y)
            
            # Get stitched image
            stitched_image = stitcher.get_output_image()
            
            # Save if output path specified
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                stitched_image.save(output_path)
            
            # Close slide reader
            slide_reader.close()
            
            return stitched_image
        except Exception as e:
            self.logger.error(f"Error stitching slide predictions: {e}")
            # Return a blank image as fallback
            blank = Image.new('RGB', (1024, 1024), color=(255, 255, 255))
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                blank.save(output_path)
            return blank
    
    def _get_transform(self):
        """Get the transform for slide tiles."""
        try:
            import albumentations as A
            from albumentations.pytorch import ToTensorV2
            
            return A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        except ImportError:
            self.logger.error("Albumentations library not found. Using default transforms.")
            # Fallback to basic transforms using PyTorch
            from torchvision import transforms
            
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])