# src/data/transforms.py
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from typing import Dict, List, Union, Optional, Any, Callable


def get_train_transforms():
    """Get training transforms."""
    return A.Compose([
        A.RandomResizedCrop(
            height=512,
            width=512,
            scale=(0.8, 1.0),
            ratio=(0.75, 1.33),
            p=0.5
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1),
            A.GridDistortion(p=1),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=1),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(p=1),
            A.RandomGamma(p=1),
            A.HueSaturationValue(p=1),
        ], p=0.3),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def get_val_transforms():
    """Get validation transforms."""
    return A.Compose([
        A.Resize(height=512, width=512),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def apply_transforms(image: np.ndarray, transforms: A.Compose) -> Dict:
    """Apply transformations to an image.
    
    Args:
        image: Input image as numpy array
        transforms: Transformation pipeline
        
    Returns:
        Transformed image
    """
    return transforms(image=image)


def denormalize_image(image: torch.Tensor, mean: List[float] = [0.485, 0.456, 0.406], 
                     std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
    """Denormalize a tensor image to numpy array.
    
    Args:
        image: Normalized image tensor [C, H, W]
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Denormalized image as numpy array [H, W, C]
    """
    # Convert to numpy and transpose
    if isinstance(image, torch.Tensor):
        # Move to CPU if on GPU
        image = image.cpu().detach().numpy()
        
        # Transpose from [C, H, W] to [H, W, C]
        image = image.transpose(1, 2, 0)
    
    # Denormalize
    image = image * np.array(std) + np.array(mean)
    
    # Scale to 0-255 and convert to uint8
    image = (image * 255).astype(np.uint8)
    
    return image