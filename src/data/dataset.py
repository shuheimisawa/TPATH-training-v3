# src/data/dataset.py
import os
import torch
import numpy as np
from PIL import Image
import cv2
import json
import glob
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import traceback
import warnings

from ..utils.io import load_image, load_json


class GlomeruliDataset(Dataset):
    """Dataset for glomeruli segmentation."""
    
    CLASSES = ['background', 'Normal', 'Partially_sclerotic', 'Sclerotic', 'Uncertain']
    
    def __init__(self, 
                 data_dir: str, 
                 mode: str = 'train', 
                 transform: Optional[Callable] = None,
                 image_extension: str = '.png',
                 max_retries: int = 3):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory with images and annotations
            mode: Dataset mode ('train', 'val', or 'test')
            transform: Optional transform to apply to images
            image_extension: Extension of image files
            max_retries: Maximum number of retries for loading images/annotations
        """
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.image_extension = image_extension
        self.max_retries = max_retries
        
        # Create category to id mapping
        self.category_to_id = {cat: idx for idx, cat in enumerate(self.CLASSES)}
        
        # Load annotations - try both fixed and original annotation files
        annotation_files = [
            os.path.join(data_dir, f"{mode}_annotations_fixed.json"),
            os.path.join(data_dir, f"{mode}_annotations.json")
        ]
        
        self.annotations = {}
        for annotation_file in annotation_files:
            if os.path.exists(annotation_file):
                print(f"Loading annotations from {annotation_file}")
                with open(annotation_file, 'r') as f:
                    self.annotations = json.load(f)
                break
        else:
            print(f"Warning: No annotation file found in {data_dir}")
            print(f"Tried: {annotation_files}")
        
        # Get list of image files
        self.image_files = []
        self.image_ids = []
        
        # First, try loading with full path keys
        for filename in os.listdir(data_dir):
            if filename.endswith('.png') and not filename.endswith('_vis.png'):
                # Create multiple potential keys to check
                image_path = os.path.join(data_dir, filename)
                basename = os.path.basename(image_path)
                rel_path = os.path.relpath(image_path)
                alt_path = rel_path.replace('\\', '/')
                
                # Check if any key format exists in annotations
                if image_path in self.annotations:
                    self.image_files.append(image_path)
                    self.image_ids.append(image_path)
                elif rel_path in self.annotations:
                    self.image_files.append(image_path)
                    self.image_ids.append(rel_path)
                elif alt_path in self.annotations:
                    self.image_files.append(image_path)
                    self.image_ids.append(alt_path)
                elif basename in self.annotations:
                    self.image_files.append(image_path)
                    self.image_ids.append(basename)
                else:
                    # Try fallback: check if the annotation key ends with this filename
                    found = False
                    for anno_key in self.annotations.keys():
                        if anno_key.endswith(basename):
                            self.image_files.append(image_path)
                            self.image_ids.append(anno_key)
                            found = True
                            break
                    
                    if not found:
                        print(f"Warning: No annotation found for {filename}")
        
        print(f"Loaded {len(self.image_files)} images with annotations")
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.image_files)
    
    def __getitem__(self, index: int) -> Optional[Dict]:
        """
        Get dataset item.
        
        Args:
            index: Index of the item
            
        Returns:
            Dictionary with image and target
        """
        try:
            # Load image
            image_path = self.image_files[index]
            image_id = self.image_ids[index]
            
            # Open image with PIL
            image = Image.open(image_path).convert('RGB')
            
            # Convert to numpy array
            image_np = np.array(image)
            
            # Get annotations for this image
            annotations = self.annotations[image_id]
            
            # Prepare target
            boxes = []
            labels = []
            masks = []
            
            # Process each annotation
            if 'annotations' in annotations:
                for anno in annotations['annotations']:
                    bbox = anno.get('bbox', [0, 0, 1, 1])
                    category = anno.get('category', 'background')
                    segmentation = anno.get('segmentation', [])
                    
                    # Convert category to label index
                    label = self.category_to_id.get(category, 0)  # Default to background
                    
                    # Add to lists
                    boxes.append(bbox)
                    labels.append(label)
                    
                    # Create mask from segmentation
                    mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
                    
                    # Fill mask using segmentation polygons
                    if isinstance(segmentation, list):
                        for polygon in segmentation:
                            if len(polygon) >= 6:  # At least 3 points (x,y pairs)
                                # Convert flat list to list of points
                                points = []
                                for i in range(0, len(polygon), 2):
                                    if i + 1 < len(polygon):
                                        points.append([polygon[i], polygon[i + 1]])
                                
                                # Convert to numpy array
                                points = np.array(points, dtype=np.int32)
                                
                                # Fill polygon
                                mask = cv2.fillPoly(mask, [points], 1)
                    elif isinstance(segmentation, dict) and 'counts' in segmentation and 'size' in segmentation:
                        # RLE format (from COCO)
                        try:
                            from pycocotools import mask as mask_util
                            mask = mask_util.decode(segmentation)
                        except:
                            # Fallback: create mask from bounding box
                            x1, y1, w, h = bbox
                            mask[int(y1):int(y1+h), int(x1):int(x1+w)] = 1
                    
                    masks.append(mask)
            
            # Convert to torch tensors
            if boxes:
                boxes = torch.tensor(boxes, dtype=torch.float32)
                labels = torch.tensor(labels, dtype=torch.int64)
                masks = torch.tensor(masks, dtype=torch.uint8)
            else:
                # Empty annotations
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros(0, dtype=torch.int64)
                masks = torch.zeros((0, image_np.shape[0], image_np.shape[1]), dtype=torch.uint8)
            
            # Create target dictionary
            target = {
                'boxes': boxes,
                'labels': labels,
                'masks': masks,
                'image_id': torch.tensor([index]),
                'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) > 0 else torch.zeros(0),
                'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
            }
            
            # Apply transforms
            if self.transform is not None:
                transformed = self.transform(image=image_np, masks=target['masks'])
                image_np = transformed['image']
                target['masks'] = transformed['masks']
            
            return {
                'image': image_np,
                'target': target,
                'image_id': image_id,
                'image_path': image_path
            }
            
        except Exception as e:
            warnings.warn(f"Error loading item at index {index}: {e}")
            # Return empty sample as fallback
            return {
                'image': np.zeros((3, 32, 32), dtype=np.uint8),
                'target': {
                    'boxes': torch.zeros((0, 4), dtype=torch.float32),
                    'labels': torch.zeros(0, dtype=torch.int64),
                    'masks': torch.zeros((0, 32, 32), dtype=torch.uint8),
                    'image_id': torch.tensor([index]),
                    'area': torch.zeros(0),
                    'iscrowd': torch.zeros(0, dtype=torch.int64)
                }
            }