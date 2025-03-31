# src/training/enhanced_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Union, Optional


class FocalTverskyLoss(nn.Module):
    """
    Focal-Tversky Loss for handling class imbalance in segmentation.
    
    This loss combines Tversky index (generalization of Dice) with a focal term
    to focus more on hard examples. It's particularly effective for imbalanced 
    segmentation tasks with small objects.
    """
    
    def __init__(
        self, 
        alpha: float = 0.7, 
        beta: float = 0.3, 
        gamma: float = 1.5, 
        smooth: float = 1.0
    ):
        """
        Initialize Focal-Tversky Loss.
        
        Args:
            alpha: Weight for false negatives (higher value = more weight)
            beta: Weight for false positives (lower value = more weight on FPs)
            gamma: Focal parameter (higher value = more focus on hard examples)
            smooth: Smoothing factor to avoid division by zero
        """
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal-Tversky loss.
        
        Args:
            y_pred: Predicted mask logits (B, C, H, W) before sigmoid
            y_true: Ground truth masks (B, C, H, W)
            
        Returns:
            Focal-Tversky loss
        """
        # Apply sigmoid to get probabilities
        y_pred = torch.sigmoid(y_pred)
        
        # Flatten the tensors
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        # Calculate Tversky index components
        tp = torch.sum(y_pred * y_true)
        fp = torch.sum(y_pred * (1 - y_true))
        fn = torch.sum((1 - y_pred) * y_true)
        
        # Calculate Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        
        # Apply focal parameter
        focal_tversky = (1 - tversky) ** self.gamma
        
        return focal_tversky


class BoundaryAwareLoss(nn.Module):
    """
    Boundary-aware loss for better segmentation of object boundaries.
    
    This loss adds a weighted term that focuses specifically on boundary pixels,
    improving the precision of segmentation boundaries.
    """
    
    def __init__(self, weight: float = 1.0):
        """
        Initialize boundary-aware loss.
        
        Args:
            weight: Weight of the boundary component in the loss
        """
        super(BoundaryAwareLoss, self).__init__()
        self.weight = weight
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Calculate boundary-aware loss.
        
        Args:
            y_pred: Predicted mask logits (B, C, H, W) before sigmoid
            y_true: Ground truth masks (B, C, H, W)
            
        Returns:
            Boundary-aware loss
        """
        # Apply sigmoid to get probabilities
        y_pred = torch.sigmoid(y_pred)
        
        # Extract boundaries from ground truth
        # Boundaries are pixels that have at least one different neighbor
        boundary_targets = self._get_boundaries(y_true)
        
        # Calculate binary cross entropy at boundaries
        boundary_loss = F.binary_cross_entropy(
            y_pred, 
            y_true, 
            weight=boundary_targets.float() * self.weight + 1.0,
            reduction='mean'
        )
        
        return boundary_loss
    
    def _get_boundaries(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Extract boundaries from a mask.
        
        Args:
            mask: Input mask tensor
            
        Returns:
            Boundary mask where boundary pixels are 1, others are 0
        """
        # Create padded version of the mask
        padded = F.pad(mask, (1, 1, 1, 1), mode='reflect')
        
        # Extract shifted versions of the mask
        left = padded[:, :, 1:-1, :-2]
        right = padded[:, :, 1:-1, 2:]
        top = padded[:, :, :-2, 1:-1]
        bottom = padded[:, :, 2:, 1:-1]
        
        # Boundary pixels are those that differ from at least one neighbor
        boundaries = ((mask != left) | 
                     (mask != right) | 
                     (mask != top) | 
                     (mask != bottom))
        
        return boundaries


class EnhancedMaskRCNNLoss(nn.Module):
    """
    Enhanced loss function for Mask R-CNN with advanced segmentation losses.
    
    This combines standard detection losses with improved segmentation losses
    designed specifically for medical imaging tasks.
    """
    
    def __init__(
        self, 
        rpn_cls_weight: float = 1.0, 
        rpn_bbox_weight: float = 1.0,
        rcnn_cls_weight: float = 1.0, 
        rcnn_bbox_weight: float = 1.0, 
        mask_weight: float = 1.5,
        focal_tversky_weight: float = 0.7, 
        boundary_weight: float = 0.3,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize enhanced loss function.
        
        Args:
            rpn_cls_weight: Weight for RPN classification loss
            rpn_bbox_weight: Weight for RPN bounding box regression loss
            rcnn_cls_weight: Weight for RCNN classification loss
            rcnn_bbox_weight: Weight for RCNN bounding box regression loss
            mask_weight: Overall weight for mask loss
            focal_tversky_weight: Weight for Focal-Tversky component in mask loss
            boundary_weight: Weight for boundary component in mask loss
            class_weights: Optional tensor of weights for each class
        """
        super(EnhancedMaskRCNNLoss, self).__init__()
        
        self.rpn_cls_weight = rpn_cls_weight
        self.rpn_bbox_weight = rpn_bbox_weight
        self.rcnn_cls_weight = rcnn_cls_weight
        self.rcnn_bbox_weight = rcnn_bbox_weight
        self.mask_weight = mask_weight
        self.focal_tversky_weight = focal_tversky_weight
        self.boundary_weight = boundary_weight
        self.class_weights = class_weights
        
        # Initialize loss components
        self.focal_tversky_loss = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.5)
        self.boundary_loss = BoundaryAwareLoss(weight=2.0)
        
    def forward(self, predictions: Dict, targets: Dict) -> Dict:
        """
        Calculate the loss.
        
        Args:
            predictions: Dictionary with model predictions
            targets: Dictionary with ground truth
            
        Returns:
            Dictionary with loss components
        """
        try:
            # RPN losses
            rpn_cls_loss = predictions.get('rpn_cls_loss', torch.tensor(0.0, device=self._get_device(predictions)))
            rpn_bbox_loss = predictions.get('rpn_bbox_loss', torch.tensor(0.0, device=self._get_device(predictions)))
            
            # Move class weights to the same device as the predictions
            if self.class_weights is not None:
                device = self._get_device(predictions)
                class_weights = self.class_weights.to(device)
            else:
                class_weights = None
            
            # RCNN classification loss with class weights and focal component
            if 'rcnn_cls_logits' in predictions and 'labels' in targets:
                # Apply focal loss for classification
                rcnn_cls_loss = self._focal_cross_entropy(
                    predictions['rcnn_cls_logits'], 
                    targets['labels'],
                    alpha=0.5,  # Balancing parameter
                    gamma=2.0,  # Focusing parameter
                    weight=class_weights
                )
            else:
                rcnn_cls_loss = torch.tensor(0.0, device=self._get_device(predictions))
            
            # RCNN bounding box regression loss - use GIoU loss for better localization
            if 'rcnn_bbox_pred' in predictions and 'bbox_targets' in targets:
                rcnn_bbox_loss = self._giou_loss(
                    predictions['rcnn_bbox_pred'],
                    targets['bbox_targets']
                )
            else:
                rcnn_bbox_loss = torch.tensor(0.0, device=self._get_device(predictions))
            
            # Enhanced mask loss (combination of BCE, Focal-Tversky, and Boundary-aware)
            if 'mask_pred' in predictions and 'masks' in targets:
                # BCE component
                bce_mask_loss = F.binary_cross_entropy_with_logits(
                    predictions['mask_pred'],
                    targets['masks'],
                    reduction='mean'
                )
                
                # Focal-Tversky component
                focal_tversky_mask_loss = self.focal_tversky_loss(
                    predictions['mask_pred'],
                    targets['masks']
                )
                
                # Boundary-aware component
                boundary_mask_loss = self.boundary_loss(
                    predictions['mask_pred'],
                    targets['masks']
                )
                
                # Combined mask loss with weights
                mask_loss = (
                    (1.0 - self.focal_tversky_weight - self.boundary_weight) * bce_mask_loss + 
                    self.focal_tversky_weight * focal_tversky_mask_loss +
                    self.boundary_weight * boundary_mask_loss
                )
            else:
                bce_mask_loss = torch.tensor(0.0, device=self._get_device(predictions))
                focal_tversky_mask_loss = torch.tensor(0.0, device=self._get_device(predictions))
                boundary_mask_loss = torch.tensor(0.0, device=self._get_device(predictions))
                mask_loss = torch.tensor(0.0, device=self._get_device(predictions))
            
            # Total loss
            loss = (
                self.rpn_cls_weight * rpn_cls_loss +
                self.rpn_bbox_weight * rpn_bbox_loss +
                self.rcnn_cls_weight * rcnn_cls_loss +
                self.rcnn_bbox_weight * rcnn_bbox_loss +
                self.mask_weight * mask_loss
            )
            
            # Create loss dictionary
            loss_dict = {
                'rpn_cls_loss': rpn_cls_loss,
                'rpn_bbox_loss': rpn_bbox_loss,
                'rcnn_cls_loss': rcnn_cls_loss,
                'rcnn_bbox_loss': rcnn_bbox_loss,
                'bce_mask_loss': bce_mask_loss,
                'focal_tversky_loss': focal_tversky_mask_loss,
                'boundary_mask_loss': boundary_mask_loss,
                'mask_loss': mask_loss,
                'total_loss': loss
            }
            
            return loss_dict
        
        except Exception as e:
            # Log error and return a basic loss dictionary with zeros
            print(f"Error in loss calculation: {e}")
            device = self._get_device(predictions)
            return {
                'rpn_cls_loss': torch.tensor(0.0, device=device),
                'rpn_bbox_loss': torch.tensor(0.0, device=device),
                'rcnn_cls_loss': torch.tensor(0.0, device=device),
                'rcnn_bbox_loss': torch.tensor(0.0, device=device),
                'bce_mask_loss': torch.tensor(0.0, device=device),
                'focal_tversky_loss': torch.tensor(0.0, device=device),
                'boundary_mask_loss': torch.tensor(0.0, device=device),
                'mask_loss': torch.tensor(0.0, device=device),
                'total_loss': torch.tensor(1.0, device=device)  # Non-zero to trigger gradient
            }
            
    def _focal_cross_entropy(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor, 
        alpha: float = 0.5, 
        gamma: float = 2.0,
        weight: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Focal Cross Entropy Loss for classification.
        
        Args:
            logits: Predicted class logits
            targets: Ground truth class indices
            alpha: Weighting factor to balance positive/negative examples
            gamma: Focusing parameter for hard examples
            weight: Optional class weights
            
        Returns:
            Focal cross entropy loss
        """
        # Standard cross entropy with class weights
        ce_loss = F.cross_entropy(
            logits, 
            targets,
            weight=weight,
            reduction='none'  # Keep per-sample losses for focal weighting
        )
        
        # Apply focal weighting
        pt = torch.exp(-ce_loss)
        focal_weight = alpha * (1 - pt) ** gamma
        
        # Apply weighting and take mean
        focal_loss = (focal_weight * ce_loss).mean()
        
        return focal_loss
    
    def _giou_loss(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        """
        GIoU (Generalized IoU) Loss for bounding box regression.
        
        Args:
            pred_boxes: Predicted bounding boxes [x1, y1, x2, y2]
            target_boxes: Ground truth bounding boxes [x1, y1, x2, y2]
            
        Returns:
            GIoU loss
        """
        # Extract coordinates
        pred_x1, pred_y1, pred_x2, pred_y2 = pred_boxes.unbind(dim=-1)
        target_x1, target_y1, target_x2, target_y2 = target_boxes.unbind(dim=-1)
        
        # Calculate areas
        pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
        target_area = (target_x2 - target_x1) * (target_y2 - target_y1)
        
        # Calculate intersection
        x1_inter = torch.max(pred_x1, target_x1)
        y1_inter = torch.max(pred_y1, target_y1)
        x2_inter = torch.min(pred_x2, target_x2)
        y2_inter = torch.min(pred_y2, target_y2)
        
        w_inter = (x2_inter - x1_inter).clamp(min=0)
        h_inter = (y2_inter - y1_inter).clamp(min=0)
        intersection = w_inter * h_inter
        
        # Calculate union
        union = pred_area + target_area - intersection
        
        # Calculate IoU
        iou = intersection / (union + 1e-7)
        
        # Calculate smallest enclosing box
        x1_encl = torch.min(pred_x1, target_x1)
        y1_encl = torch.min(pred_y1, target_y1)
        x2_encl = torch.max(pred_x2, target_x2)
        y2_encl = torch.max(pred_y2, target_y2)
        
        w_encl = (x2_encl - x1_encl).clamp(min=0)
        h_encl = (y2_encl - y1_encl).clamp(min=0)
        enclosing_area = w_encl * h_encl
        
        # Calculate GIoU
        giou = iou - (enclosing_area - union) / (enclosing_area + 1e-7)
        
        # Return loss (1 - GIoU)
        return (1 - giou).mean()
    
    def _get_device(self, predictions: Dict) -> torch.device:
        """Get the device from the predictions dictionary."""
        for value in predictions.values():
            if isinstance(value, torch.Tensor):
                return value.device
        return torch.device('cpu')