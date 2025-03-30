import numpy as np
import cv2
import random
from typing import Dict, List, Tuple, Optional, Union, Any


class RandomCrop:
    """Randomly crop the image and adjust bounding boxes and masks."""
    
    def __init__(self, height: int, width: int, p: float = 0.5):
        self.height = height
        self.width = width
        self.p = p
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.p:
            return sample
        
        image = sample['image']
        boxes = sample['boxes']
        masks = sample['masks']
        
        h, w = image.shape[:2]
        
        if h <= self.height or w <= self.width:
            return sample
        
        # Generate random crop coordinates
        top = random.randint(0, h - self.height)
        left = random.randint(0, w - self.width)
        
        # Crop image
        image = image[top:top+self.height, left:left+self.width]
        
        # Adjust masks
        new_masks = []
        for mask in masks:
            new_mask = mask[top:top+self.height, left:left+self.width]
            new_masks.append(new_mask)
        
        # Adjust boxes
        new_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # Adjust coordinates
            x1 = max(0, x1 - left)
            y1 = max(0, y1 - top)
            x2 = min(self.width, x2 - left)
            y2 = min(self.height, y2 - top)
            
            # Skip boxes that are outside crop or too small
            if x2 <= x1 or y2 <= y1:
                continue
                
            new_boxes.append([x1, y1, x2, y2])
        
        # Update sample
        sample['image'] = image
        sample['boxes'] = np.array(new_boxes)
        sample['masks'] = np.array(new_masks)
        
        return sample


class RandomRotate90:
    """Randomly rotate the image and adjust bounding boxes and masks by 90 degrees."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.p:
            return sample
        
        image = sample['image']
        boxes = sample['boxes']
        masks = sample['masks']
        
        # Get image dimensions
        h, w = image.shape[:2]
        
        # Choose random rotation (0, 90, 180, 270 degrees)
        k = random.randint(0, 3)
        
        if k == 0:  # No rotation
            return sample
        
        # Rotate image
        image = np.rot90(image, k=k)
        
        # Rotate masks
        new_masks = [np.rot90(mask, k=k) for mask in masks]
        
        # Rotate boxes
        new_boxes = []
        for box in boxes:
            x1, y1, x2, y2 = box
            
            if k == 1:  # 90 degrees
                new_x1 = h - y2
                new_y1 = x1
                new_x2 = h - y1
                new_y2 = x2
            elif k == 2:  # 180 degrees
                new_x1 = w - x2
                new_y1 = h - y2
                new_x2 = w - x1
                new_y2 = h - y1
            else:  # 270 degrees
                new_x1 = y1
                new_y1 = w - x2
                new_x2 = y2
                new_y2 = w - x1
            
            new_boxes.append([new_x1, new_y1, new_x2, new_y2])
        
        # Update sample
        sample['image'] = image
        sample['boxes'] = np.array(new_boxes)
        sample['masks'] = np.array(new_masks)
        
        return sample