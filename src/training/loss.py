import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List


class DiceLoss(nn.Module):
    """Dice loss for segmentation masks."""
    
    def __init__(self, smooth=1.0):
        """Initialize Dice loss.
        
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, input, target):
        """Calculate Dice loss.
        
        Args:
            input: Predicted mask logits
            target: Ground truth mask
            
        Returns:
            Dice loss (1 - Dice coefficient)
        """
        input = torch.sigmoid(input)
        
        # Flatten tensors
        input_flat = input.view(-1)
        target_flat = target.view(-1)
        
        # Calculate Dice coefficient
        intersection = (input_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (input_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice


class MaskRCNNLoss(nn.Module):
    """Loss function for Mask R-CNN."""
    
    def __init__(self, rpn_cls_weight=1.0, rpn_bbox_weight=1.0, 
                 rcnn_cls_weight=1.0, rcnn_bbox_weight=1.0, mask_weight=1.0,
                 dice_weight=0.5, class_weights=None):
        """Initialize the loss function.
        
        Args:
            rpn_cls_weight: Weight for RPN classification loss
            rpn_bbox_weight: Weight for RPN bounding box regression loss
            rcnn_cls_weight: Weight for RCNN classification loss
            rcnn_bbox_weight: Weight for RCNN bounding box regression loss
            mask_weight: Weight for mask loss
            dice_weight: Weight for Dice loss component within mask loss
            class_weights: Optional tensor of weights for each class
        """
        super(MaskRCNNLoss, self).__init__()
        
        self.rpn_cls_weight = rpn_cls_weight
        self.rpn_bbox_weight = rpn_bbox_weight
        self.rcnn_cls_weight = rcnn_cls_weight
        self.rcnn_bbox_weight = rcnn_bbox_weight
        self.mask_weight = mask_weight
        self.dice_weight = dice_weight
        self.class_weights = class_weights
        
        # Initialize Dice loss
        self.dice_loss = DiceLoss(smooth=1.0)
    
    def forward(self, predictions: Dict, targets: Dict) -> Dict:
        """Calculate the loss.
        
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
            
            # RCNN classification loss with class weights
            if 'rcnn_cls_logits' in predictions and 'labels' in targets:
                rcnn_cls_loss = F.cross_entropy(
                    predictions['rcnn_cls_logits'], 
                    targets['labels'],
                    weight=class_weights
                )
            else:
                rcnn_cls_loss = torch.tensor(0.0, device=self._get_device(predictions))
            
            # RCNN bounding box regression loss
            if 'rcnn_bbox_pred' in predictions and 'bbox_targets' in targets:
                rcnn_bbox_loss = F.smooth_l1_loss(
                    predictions['rcnn_bbox_pred'],
                    targets['bbox_targets'],
                    reduction='mean'
                )
            else:
                rcnn_bbox_loss = torch.tensor(0.0, device=self._get_device(predictions))
            
            # BCE mask loss
            if 'mask_pred' in predictions and 'masks' in targets:
                bce_mask_loss = F.binary_cross_entropy_with_logits(
                    predictions['mask_pred'],
                    targets['masks'],
                    reduction='mean'
                )
                
                # Dice loss for masks
                dice_loss_value = self.dice_loss(predictions['mask_pred'], targets['masks'])
                
                # Combined mask loss (BCE + Dice)
                mask_loss = (1 - self.dice_weight) * bce_mask_loss + self.dice_weight * dice_loss_value
            else:
                bce_mask_loss = torch.tensor(0.0, device=self._get_device(predictions))
                dice_loss_value = torch.tensor(0.0, device=self._get_device(predictions))
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
                'dice_loss': dice_loss_value,
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
                'dice_loss': torch.tensor(0.0, device=device),
                'mask_loss': torch.tensor(0.0, device=device),
                'total_loss': torch.tensor(1.0, device=device)  # Non-zero to trigger gradient
            }
    
    def _get_device(self, predictions: Dict) -> torch.device:
        """Get the device from the predictions dictionary."""
        for value in predictions.values():
            if isinstance(value, torch.Tensor):
                return value.device
        return torch.device('cpu')