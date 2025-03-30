# New file: src/inference/two_stage_pipeline.py

import os
import torch
import numpy as np
from PIL import Image
import cv2
from typing import Dict, List, Optional, Union, Tuple, Any
import warnings

from ..models.cascade_mask_rcnn import CascadeMaskRCNN
from ..models.glomeruli_classifier import GlomeruliClassifier
from ..utils.stain_normalization import VahadaneNormalizer, StainNormalizationTransform
from ..utils.feature_extraction import TextureFeatureExtractor, MorphologicalFeatureExtractor, ColorFeatureExtractor
from ..utils.slide_io import SlideReader, TileExtractor
from ..utils.logger import get_logger


class TwoStagePipeline:
    """
    Two-stage pipeline for glomeruli detection and classification.
    
    Stage 1: Detect all glomeruli in a WSI using Cascade Mask R-CNN
    Stage 2: Classify each detected glomerulus using a dedicated classifier
    """
    
    def __init__(self, 
                 detection_model: CascadeMaskRCNN,
                 classification_model: GlomeruliClassifier,
                 device: torch.device,
                 config: Dict,
                 class_names: List[str] = ['Normal', 'Sclerotic', 'Partially_sclerotic', 'Uncertain']):
        """
        Initialize two-stage pipeline.
        
        Args:
            detection_model: Stage 1 detection model
            classification_model: Stage 2 classification model
            device: Device to run inference on
            config: Configuration dictionary
            class_names: Names of glomeruli classes
        """
        self.logger = get_logger(name="two_stage_pipeline")
        
        # Set models
        self.detection_model = detection_model
        self.classification_model = classification_model
        
        # Set device
        self.device = device
        self.detection_model.to(device)
        self.classification_model.to(device)
        
        # Set evaluation mode
        self.detection_model.eval()
        self.classification_model.eval()
        
        # Set class names
        self.class_names = class_names
        
        # Set configuration
        self.config = config
        
        # Initialize stain normalizer
        self.normalizer = StainNormalizationTransform(
            method=config['normalization']['method'],
            target_image_path=config['normalization']['reference_image_path'],
            params_path=config['normalization']['params_path']
        )
        
        # Initialize feature extractors
        if config['feature_extraction']['use_texture_features']:
            self.texture_extractor = TextureFeatureExtractor(
                gabor_frequencies=config['feature_extraction']['gabor_frequencies'],
                gabor_orientations=[float(o * np.pi) for o in config['feature_extraction']['gabor_orientations']],
                lbp_radius=config['feature_extraction']['lbp_radius'],
                lbp_points=config['feature_extraction']['lbp_points']
            )
        else:
            self.texture_extractor = None
        
        if config['feature_extraction']['use_color_features']:
            self.color_extractor = ColorFeatureExtractor(
                bins=config['feature_extraction']['color_bins']
            )
        else:
            self.color_extractor = None
        
        if config['feature_extraction']['use_morphological_features']:
            self.morphological_extractor = MorphologicalFeatureExtractor()
        else:
            self.morphological_extractor = None
        
        self.logger.info("Initialized two-stage pipeline")
    
    def _preprocess_image(self, image: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess image for inference.
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Tuple of (normalized tensor, normalized numpy array)
        """
        # Apply stain normalization
        normalized_image = self.normalizer(image)
        
        # Convert to tensor and normalize for model
        tensor = torch.from_numpy(normalized_image.transpose((2, 0, 1))).float() / 255.0
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor, normalized_image
    
    def _extract_manual_features(self, image: np.ndarray, mask: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Extract manual features from an image.
        
        Args:
            image: RGB image as numpy array
            mask: Optional binary mask of the region
            
        Returns:
            Dictionary of features
        """
        features = {}
        
        # Extract texture features
        if self.texture_extractor is not None:
            texture_features = self.texture_extractor.extract_features(image)
            features.update(texture_features)
        
        # Extract color features
        if self.color_extractor is not None:
            color_features = self.color_extractor.extract_features(image)
            features.update(color_features)
        
        # Extract morphological features if mask is provided
        if self.morphological_extractor is not None and mask is not None:
            morphological_features = self.morphological_extractor.extract_features(mask)
            features.update(morphological_features)
        
        return features
    
    def _convert_manual_features_to_tensor(self, features: Dict[str, float]) -> torch.Tensor:
        """
        Convert manual features dictionary to tensor.
        
        Args:
            features: Dictionary of features
            
        Returns:
            Tensor of features
        """
        # Sort keys for consistent ordering
        sorted_keys = sorted(features.keys())
        
        # Create tensor from values
        tensor = torch.tensor([features[k] for k in sorted_keys], dtype=torch.float32)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def detect_glomeruli(self, image: np.ndarray) -> Dict:
        """
        Detect glomeruli in an image (Stage 1).
        
        Args:
            image: RGB image as numpy array
            
        Returns:
            Detection results
        """
        # Preprocess image
        tensor, normalized_image = self._preprocess_image(image)
        
        # Move to device
        tensor = tensor.to(self.device)
        
        # Run detection
        with torch.no_grad():
            detections = self.detection_model(tensor)
        
        # Filter detections
        score_threshold = self.config['detection']['score_threshold']
        
        # Process detections
        boxes = []
        masks = []
        labels = []
        scores = []
        
        if len(detections) > 0:
            detection = detections[0]  # Get first batch item
            
            # Filter by score
            keep = detection['scores'] > score_threshold
            
            if keep.sum() > 0:
                # Get filtered detections
                filtered_boxes = detection['boxes'][keep].cpu().numpy()
                filtered_masks = detection['masks'][keep].cpu().numpy()
                filtered_labels = detection['labels'][keep].cpu().numpy()
                filtered_scores = detection['scores'][keep].cpu().numpy()
                
                # Convert masks to binary
                binary_masks = (filtered_masks > 0.5).astype(np.uint8)
                
                # Add to lists
                boxes = filtered_boxes
                masks = binary_masks
                labels = filtered_labels
                scores = filtered_scores
        
        # Return results
        return {
            'boxes': boxes,
            'masks': masks,
            'labels': labels,
            'scores': scores,
            'normalized_image': normalized_image
        }
    
    def classify_glomeruli(self, image: np.ndarray, boxes: np.ndarray, 
                          masks: np.ndarray) -> Dict:
        """
        Classify detected glomeruli (Stage 2).
        
        Args:
            image: RGB image as numpy array
            boxes: Bounding boxes from detection stage
            masks: Masks from detection stage
            
        Returns:
            Classification results
        """
        # Get patch size
        patch_size = self.config['classification']['patch_size']
        
        # Initialize result lists
        class_labels = []
        class_scores = []
        
        # Process each detection
        for i in range(len(boxes)):
            try:
                # Get bounding box
                box = boxes[i].astype(int)
                x1, y1, x2, y2 = box
                
                # Ensure box is within image bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(image.shape[1], x2)
                y2 = min(image.shape[0], y2)
                
                # Check if box is valid
                if x2 <= x1 or y2 <= y1:
                    class_labels.append(0)  # Background
                    class_scores.append(0.0)
                    continue
                
                # Extract patch
                patch = image[y1:y2, x1:x2]
                
                # Extract mask
                mask = masks[i]
                patch_mask = mask[y1:y2, x1:x2]
                
                # Resize patch to model input size
                patch_resized = cv2.resize(patch, (patch_size, patch_size), 
                                           interpolation=cv2.INTER_LINEAR)
                mask_resized = cv2.resize(patch_mask, (patch_size, patch_size), 
                                          interpolation=cv2.INTER_NEAREST)
                
                # Extract manual features
                manual_features = self._extract_manual_features(patch_resized, mask_resized)
                manual_features_tensor = self._convert_manual_features_to_tensor(manual_features)
                
                # Preprocess for model
                patch_tensor = torch.from_numpy(patch_resized.transpose((2, 0, 1))).float() / 255.0
                patch_tensor = patch_tensor.unsqueeze(0)
                
                # Move to device
                patch_tensor = patch_tensor.to(self.device)
                manual_features_tensor = manual_features_tensor.to(self.device)
                
                # Run classification
                with torch.no_grad():
                    output = self.classification_model(patch_tensor, manual_features_tensor)
                
                # Get predictions
                logits = output['logits']
                probabilities = torch.softmax(logits, dim=1)
                
                # Get class with highest probability
                class_idx = torch.argmax(probabilities, dim=1).item()
                class_score = probabilities[0, class_idx].item()
                
                # Apply confidence threshold
                confidence_threshold = self.config['classification']['confidence_threshold']
                
                if class_score >= confidence_threshold:
                    class_labels.append(class_idx)
                    class_scores.append(class_score)
                else:
                    class_labels.append(0)  # Background if low confidence
                    class_scores.append(class_score)
                
            except Exception as e:
                self.logger.error(f"Error classifying glomerulus: {e}")
                # Add default values on error
                class_labels.append(0)
                class_scores.append(0.0)
        
        # Return results
        return {
            'class_labels': np.array(class_labels),
            'class_scores': np.array(class_scores)
        }
    
    def process_tile(self, tile: np.ndarray) -> Dict:
        """
        Process a single tile through the entire pipeline.
        
        Args:
            tile: RGB image as numpy array
            
        Returns:
            Combined detection and classification results
        """
        # Stage 1: Detect glomeruli
        detection_results = self.detect_glomeruli(tile)
        
        # Skip classification if no detections
        if len(detection_results['boxes']) == 0:
            return {
                'detection': detection_results,
                'classification': {
                    'class_labels': np.array([]),
                    'class_scores': np.array([])
                },
                'combined': {
                    'boxes': np.array([]),
                    'masks': np.array([]),
                    'labels': np.array([]),
                    'scores': np.array([]),
                    'class_names': []
                }
            }
        
        # Stage 2: Classify detected glomeruli
        classification_results = self.classify_glomeruli(
            detection_results['normalized_image'],
            detection_results['boxes'],
            detection_results['masks']
        )
        
        # Combine results
        combined = {
            'boxes': detection_results['boxes'],
            'masks': detection_results['masks'],
            'labels': classification_results['class_labels'],
            'scores': classification_results['class_scores'],
            'class_names': [self.class_names[l] for l in classification_results['class_labels']]
        }
        
        return {
            'detection': detection_results,
            'classification': classification_results,
            'combined': combined
        }
    
    def process_slide(self, slide_path: str, tile_size: int = 1024, 
                     overlap: int = 256, level: int = 0, 
                     filter_background: bool = True) -> Dict:
        """
        Process a whole slide through the pipeline.
        
        Args:
            slide_path: Path to slide file
            tile_size: Size of tiles to extract
            overlap: Overlap between adjacent tiles
            level: Magnification level to process
            filter_background: Whether to filter out background tiles
            
        Returns:
            Processed slide results
        """
        self.logger.info(f"Processing slide {slide_path}")
        
        # Open slide
        slide_reader = SlideReader(slide_path)
        
        # Extract tiles
        tile_extractor = TileExtractor(
            tile_size=tile_size,
            overlap=overlap,
            level=level
        )
        
        tiles = tile_extractor.extract_tiles(
            slide_reader=slide_reader,
            filter_background=filter_background
        )
        
        self.logger.info(f"Extracted {len(tiles)} tiles")
        
        # Process each tile
        results = []
        
        for tile_idx, tile_info in enumerate(tiles):
            try:
                # Read tile
                tile = slide_reader.read_region(
                    location=(tile_info['x'], tile_info['y']),
                    level=tile_info['level'],
                    size=(tile_info['width'], tile_info['height'])
                )
                
                # Convert to numpy array
                tile_array = np.array(tile)
                
                # Process tile
                tile_result = self.process_tile(tile_array)
                
                # Add tile coordinates and index
                tile_result['x'] = tile_info['x']
                tile_result['y'] = tile_info['y']
                tile_result['width'] = tile_info['width']
                tile_result['height'] = tile_info['height']
                tile_result['level'] = tile_info['level']
                tile_result['index'] = tile_idx
                
                # Add to results
                results.append(tile_result)
                
                self.logger.info(f"Processed tile {tile_idx+1}/{len(tiles)}")
                
            except Exception as e:
                self.logger.error(f"Error processing tile {tile_idx}: {e}")
        
        # Close slide
        slide_reader.close()
        
        return {
            'slide_path': slide_path,
            'results': results
        }
    
    def visualize_results(self, image: np.ndarray, results: Dict, 
                         output_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize detection and classification results.
        
        Args:
            image: RGB image as numpy array
            results: Results from process_tile method
            output_path: Optional path to save visualization
            
        Returns:
            Visualization image
        """
        # Make a copy of the image
        vis = image.copy()
        
        # Draw detections
        for i in range(len(results['combined']['boxes'])):
            # Get detection info
            box = results['combined']['boxes'][i].astype(int)
            mask = results['combined']['masks'][i]
            label = results['combined']['labels'][i]
            score = results['combined']['scores'][i]
            class_name = results['combined']['class_names'][i]
            
            # Generate color based on class
            color_map = {
                'Normal': (0, 255, 0),  # Green
                'Sclerotic': (255, 0, 0),  # Red
                'Partially_sclerotic': (255, 255, 0),  # Yellow
                'Uncertain': (0, 0, 255)  # Blue
            }
            
            color = color_map.get(class_name, (255, 255, 255))
            
            # Draw bounding box
            x1, y1, x2, y2 = box
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw class name and score
            text = f"{class_name}: {score:.2f}"
            cv2.putText(vis, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw mask overlay
            mask_colored = np.zeros_like(vis)
            mask_colored[mask > 0] = (*color, 128)  # Semi-transparent color
            cv2.addWeighted(mask_colored, 0.5, vis, 1, 0, vis)
        
        # Save if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        return vis