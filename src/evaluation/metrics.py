import torch
import numpy as np
from typing import Tuple, Union

def calculate_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """Calculate IoU between two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    # Get coordinates of intersection
    x1 = torch.max(box1[0], box2[0])
    y1 = torch.max(box1[1], box2[1])
    x2 = torch.min(box1[2], box2[2])
    y2 = torch.min(box1[3], box2[3])
    
    # Calculate area of intersection
    width = torch.clamp(x2 - x1, min=0)
    height = torch.clamp(y2 - y1, min=0)
    intersection = width * height
    
    # Calculate areas of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate union area
    union = box1_area + box2_area - intersection
    
    # Calculate IoU
    iou = intersection / union
    
    return iou

def calculate_mask_iou(mask1: Union[torch.Tensor, np.ndarray], 
                      mask2: Union[torch.Tensor, np.ndarray]) -> float:
    """Calculate IoU between two binary masks.
    
    Args:
        mask1: First binary mask (1 for object, 0 for background)
        mask2: Second binary mask (1 for object, 0 for background)
        
    Returns:
        IoU value
    """
    # Convert to numpy if tensors
    if isinstance(mask1, torch.Tensor):
        mask1 = mask1.cpu().numpy()
    if isinstance(mask2, torch.Tensor):
        mask2 = mask2.cpu().numpy()
    
    # Ensure binary masks
    mask1 = mask1 > 0.5
    mask2 = mask2 > 0.5
    
    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    
    # Calculate IoU
    if union == 0:
        return 0.0
    
    iou = intersection / union
    return float(iou)

def calculate_precision_recall(true_positives: int, 
                              false_positives: int, 
                              false_negatives: int) -> Tuple[float, float]:
    """Calculate precision and recall.
    
    Args:
        true_positives: Number of true positives
        false_positives: Number of false positives
        false_negatives: Number of false negatives
        
    Returns:
        Tuple of (precision, recall)
    """
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    return precision, recall

def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall.
    
    Args:
        precision: Precision value
        recall: Recall value
        
    Returns:
        F1 score
    """
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

def calculate_map(average_precisions: list) -> float:
    """Calculate mean Average Precision (mAP).
    
    Args:
        average_precisions: List of average precision values for each class
        
    Returns:
        mAP value
    """
    # Filter out None or NaN values
    valid_aps = [ap for ap in average_precisions if ap is not None and not np.isnan(ap)]
    
    # Calculate mean AP
    if not valid_aps:
        return 0.0
    
    map_value = sum(valid_aps) / len(valid_aps)
    return map_value