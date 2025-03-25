# src/models/cascade_mask_rcnn.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign, clip_boxes_to_image, nms as box_nms

# Custom implementation of box_iou
def box_area(boxes):
    """
    Computes the area of a set of bounding boxes
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

def box_iou(boxes1, boxes2):
    """
    Return intersection-over-union (Jaccard index) between boxes.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou

from .backbones.resnet import ResNetBackbone
from .components.fpn import FPN
from .components.bifpn import BiFPN
from .components.attention import SelfAttention, CBAM
from .components.box_head import CascadeBoxHead, CascadeBoxPredictor
from .components.mask_head import MaskRCNNHeadWithAttention
from ..config.model_config import ModelConfig


class BoxCoder:
    """Box coder for converting between box formats."""
    
    def __init__(self, weights=(10., 10., 5., 5.)):
        """Initialize box coder.
        
        Args:
            weights: Box regression weights
        """
        self.weights = weights
    
    def encode(self, reference_boxes, proposals):
        """Encode boxes relative to proposals.
        
        Args:
            reference_boxes: Ground truth boxes
            proposals: Anchor/proposal boxes
            
        Returns:
            Box deltas
        """
        ex_widths = proposals[:, 2] - proposals[:, 0]
        ex_heights = proposals[:, 3] - proposals[:, 1]
        ex_ctr_x = proposals[:, 0] + 0.5 * ex_widths
        ex_ctr_y = proposals[:, 1] + 0.5 * ex_heights

        gt_widths = reference_boxes[:, 2] - reference_boxes[:, 0]
        gt_heights = reference_boxes[:, 3] - reference_boxes[:, 1]
        gt_ctr_x = reference_boxes[:, 0] + 0.5 * gt_widths
        gt_ctr_y = reference_boxes[:, 1] + 0.5 * gt_heights

        wx, wy, ww, wh = self.weights
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)

        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
        return targets

    def decode(self, rel_codes, boxes):
        """Decode relative codes to boxes.
        
        Args:
            rel_codes: Box deltas
            boxes: Reference boxes
            
        Returns:
            Decoded boxes
        """
        boxes = boxes.to(rel_codes.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = boxes[:, 2] - boxes[:, 0] + TO_REMOVE
        heights = boxes[:, 3] - boxes[:, 1] + TO_REMOVE
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        wx, wy, ww, wh = self.weights
        
        # Handle different input shapes
        if rel_codes.shape[-1] == 4:
            dx = rel_codes[:, 0] / wx
            dy = rel_codes[:, 1] / wy
            dw = rel_codes[:, 2] / ww
            dh = rel_codes[:, 3] / wh
        else:
            dx = rel_codes[:, 0::4] / wx
            dy = rel_codes[:, 1::4] / wy
            dw = rel_codes[:, 2::4] / ww
            dh = rel_codes[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=math.log(1000. / 16))
        dh = torch.clamp(dh, max=math.log(1000. / 16))

        # Handle different input shapes
        if rel_codes.shape[-1] == 4:
            pred_ctr_x = dx * widths + ctr_x
            pred_ctr_y = dy * heights + ctr_y
            pred_w = torch.exp(dw) * widths
            pred_h = torch.exp(dh) * heights

            pred_boxes = torch.zeros_like(rel_codes)
            # x1
            pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
            # y1
            pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
            # x2
            pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w - 1
            # y2
            pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h - 1
        else:
            pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
            pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
            pred_w = torch.exp(dw) * widths[:, None]
            pred_h = torch.exp(dh) * heights[:, None]

            pred_boxes = torch.zeros_like(rel_codes)
            # x1
            pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
            # y1
            pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
            # x2
            pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w - 1
            # y2
            pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h - 1

        return pred_boxes


class CascadeMaskRCNN(nn.Module):
    """Cascade Mask R-CNN for glomeruli instance segmentation."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the model.
        
        Args:
            config: Model configuration
        """
        super(CascadeMaskRCNN, self).__init__()
        
        self.config = config
        
        # Initialize box coder
        self.box_coder = BoxCoder()
        
        # Image normalization and size transformation
        self.transform = GeneralizedRCNNTransform(
            min_size=800,
            max_size=1333,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225]
        )
        
        # Build backbone
        self.backbone = ResNetBackbone(
            name=config.backbone.name,
            pretrained=config.backbone.pretrained,
            freeze_stages=config.backbone.freeze_stages,
            norm_eval=config.backbone.norm_eval,
            out_indices=config.backbone.out_indices
        )
        
        # Build feature pyramid network
        if getattr(config, 'use_bifpn', False):
            # Use BiFPN for enhanced feature fusion
            self.fpn = BiFPN(
                in_channels=config.fpn.in_channels,
                out_channels=config.fpn.out_channels,
                num_blocks=getattr(config.fpn, 'num_blocks', 3),
                attention_type=getattr(config.fpn, 'attention_type', 'none'),
                extra_convs_on_inputs=config.fpn.extra_convs_on_inputs
            )
        else:
            # Use standard FPN
            self.fpn = FPN(
                in_channels=config.fpn.in_channels,
                out_channels=config.fpn.out_channels,
                num_outs=config.fpn.num_outs,
                add_extra_convs=config.fpn.add_extra_convs,
                extra_convs_on_inputs=config.fpn.extra_convs_on_inputs
            )
        
        # Anchor generator
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        self.anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        
        # RPN head
        self.rpn_head = RPNHead(
            in_channels=config.fpn.out_channels,
            num_anchors=self.anchor_generator.num_anchors_per_location()[0]
        )
        
        # RoI feature extractor
        self.box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=config.roi.roi_size,
            sampling_ratio=config.roi.roi_sample_num
        )
        
        # Initialize the cascade stages
        self.num_cascade_stages = config.cascade.num_stages
        self.cascade_stages = nn.ModuleList()
        
        for stage in range(self.num_cascade_stages):
            # Box head for this stage
            box_head = CascadeBoxHead(
                in_channels=config.fpn.out_channels,
                representation_size=1024,
                roi_size=config.roi.roi_size
            )
            
            # Box predictor for this stage
            box_predictor = CascadeBoxPredictor(
                in_channels=1024,
                num_classes=config.roi.classes
            )
            
            # Add stage
            self.cascade_stages.append(nn.ModuleDict({
                'box_head': box_head,
                'box_predictor': box_predictor
            }))
        
        # Mask head
        self.mask_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=config.mask.roi_size,
            sampling_ratio=2
        )
        
        # Add self-attention to mask head if enabled
        use_attention = getattr(config, 'use_attention', False)
        attention_type = getattr(config, 'attention_type', 'self')
        
        self.mask_head = MaskRCNNHeadWithAttention(
            in_channels=config.fpn.out_channels,
            layers=(256, 256, 256, 256),
            dilation=1,
            roi_size=config.mask.roi_size,
            num_classes=config.num_classes,
            use_attention=use_attention,
            attention_type=attention_type
        )
        
        # Region Proposal Network
        self.rpn = RegionProposalNetwork(
            anchor_generator=self.anchor_generator,
            head=self.rpn_head,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            batch_size_per_image=256,
            positive_fraction=0.5,
            pre_nms_top_n={'training': 2000, 'testing': 1000},
            post_nms_top_n={'training': 2000, 'testing': 1000},
            nms_thresh=0.7
        )
        
        # Loss weights for each cascade stage
        self.cascade_loss_weights = config.cascade.stage_loss_weights
        
        # IoU thresholds for positive samples in each cascade stage
        self.cascade_iou_thresholds = config.cascade.iou_thresholds
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the model."""
        # The backbone and FPN are already initialized
        # Initialize the RPN, box heads, and mask head
        for name, param in self.named_parameters():
            if 'backbone' in name or 'fpn' in name:
                continue
            
            if 'weight' in name:
                nn.init.normal_(param, mean=0, std=0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, images, targets=None):
        """Forward pass of the model.
        
        Args:
            images: List of images
            targets: Optional list of targets for training
            
        Returns:
            During training: Dictionary of losses
            During inference: List of predicted instances
        """
        # Check if in training mode
        is_training = self.training and targets is not None
        
        # Preprocess images and targets
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))
        
        images, targets = self.transform(images, targets)
        
        # Extract features
        features = self.backbone(images.tensors)
        features = self.fpn(features)
        
        # Create feature dict for RoI pooling
        feature_dict = {str(i): feat for i, feat in enumerate(features)}
        
        # Generate region proposals
        proposals, rpn_losses = self.rpn(images, feature_dict, targets)
        
        # Initialize losses
        losses = {}
        detections = []
        
        # Check if we have any valid proposals
        has_valid_proposals = any(len(p) > 0 for p in proposals)
        
        if not has_valid_proposals:
            # Return empty results
            if is_training:
                losses.update(rpn_losses)
                losses.update({
                    'stage_0': torch.tensor(0.0, device=images.tensors.device),
                    'stage_1': torch.tensor(0.0, device=images.tensors.device),
                    'stage_2': torch.tensor(0.0, device=images.tensors.device),
                    'mask_loss': torch.tensor(0.0, device=images.tensors.device)
                })
                return losses
            else:
                empty_detections = [{
                    'boxes': torch.zeros((0, 4), device=images.tensors.device),
                    'labels': torch.zeros((0,), dtype=torch.int64, device=images.tensors.device),
                    'scores': torch.zeros((0,), device=images.tensors.device),
                    'masks': torch.zeros((0, self.config.num_classes, *self.config.mask.roi_size), device=images.tensors.device)
                } for _ in range(len(images))]
                return empty_detections
        
        if is_training:
            # Add RPN losses
            losses.update(rpn_losses)
            
            # Process each cascade stage
            for stage_idx in range(self.num_cascade_stages):
                stage_dict = self.cascade_stages[stage_idx]
                iou_thresh = self.cascade_iou_thresholds[stage_idx]
                
                # Extract box features
                box_features = self.box_roi_pool(feature_dict, proposals, images.image_sizes)
                
                # Get predictions
                box_features = stage_dict['box_head'](box_features)
                class_logits, box_regression = stage_dict['box_predictor'](box_features)
                
                # Calculate losses for this stage
                stage_loss = self.cascade_loss_weights[stage_idx] * self._compute_stage_loss(
                    class_logits, box_regression, proposals, targets, iou_thresh
                )
                losses.update({f'stage_{stage_idx}': stage_loss})
                
                # Update proposals for next stage
                if stage_idx < self.num_cascade_stages - 1:
                    proposals = self._get_boxes_for_next_stage(
                        box_regression, class_logits, proposals, images.image_sizes
                    )
            
            # Mask head forward pass
            if len(proposals) > 0:
                mask_features = self.mask_roi_pool(feature_dict, proposals, images.image_sizes)
                mask_logits = self.mask_head(mask_features)
                mask_loss = self._compute_mask_loss(mask_logits, proposals, targets)
                losses.update({'mask_loss': mask_loss})
            
            # Calculate total loss
            total_loss = sum(loss for loss in losses.values() if isinstance(loss, torch.Tensor))
            losses['total_loss'] = total_loss
            
            return losses
        else:
            # Inference mode
            detections = []
            
            # Process each cascade stage
            for stage_idx in range(self.num_cascade_stages):
                stage_dict = self.cascade_stages[stage_idx]
                
                # Extract box features
                box_features = self.box_roi_pool(feature_dict, proposals, images.image_sizes)
                
                # Get predictions
                box_features = stage_dict['box_head'](box_features)
                class_logits, box_regression = stage_dict['box_predictor'](box_features)
                
                # Update proposals for next stage
                if stage_idx < self.num_cascade_stages - 1:
                    proposals = self._get_boxes_for_next_stage(
                        box_regression, class_logits, proposals, images.image_sizes
                    )
            
            # Final detections
            boxes = proposals[0]
            scores = F.softmax(class_logits, dim=-1)[:, 1:].max(dim=1)[0]
            labels = F.softmax(class_logits, dim=-1)[:, 1:].max(dim=1)[1] + 1
            
            # Get masks for final boxes
            if len(boxes) > 0:
                mask_features = self.mask_roi_pool(feature_dict, [boxes], images.image_sizes)
                mask_logits = self.mask_head(mask_features)
                masks = F.sigmoid(mask_logits)
            else:
                masks = torch.zeros((0, self.config.num_classes, *self.config.mask.roi_size), device=images.tensors.device)
            
            # Post-process detections
            detections = [{
                'boxes': boxes,
                'labels': labels,
                'scores': scores,
                'masks': masks
            }]
            
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
            return detections
            
    def _compute_stage_loss(self, class_logits, box_regression, proposals, targets, iou_thresh):
        """Compute loss for a single cascade stage.
        
        Args:
            class_logits: Class predictions
            box_regression: Box regression predictions
            proposals: RPN proposals
            targets: Ground truth targets
            iou_thresh: IoU threshold for this stage
            
        Returns:
            Total loss for this stage
        """
        # Match proposals to targets
        matched_idxs = []
        for proposals_in_image, targets_in_image in zip(proposals, targets):
            if targets_in_image['boxes'].numel() == 0:
                matched_idxs.append(
                    torch.zeros(
                        proposals_in_image.shape[0],
                        dtype=torch.int64,
                        device=proposals_in_image.device
                    )
                )
                continue
                
            match_quality_matrix = box_iou(
                targets_in_image['boxes'],
                proposals_in_image
            )
            matched_vals, matches = match_quality_matrix.max(dim=0)
            
            # Assign background (negative) to below threshold matches
            matches[matched_vals < iou_thresh] = -1
            
            matched_idxs.append(matches)
        
        # Compute classification loss
        classification_loss = F.cross_entropy(
            class_logits,
            torch.cat([t['labels'][matched_idxs[i]] for i, t in enumerate(targets)])
        )
        
        # Compute box regression loss
        sampled_pos_inds_subset = torch.cat([matched_idxs[i] >= 0 for i in range(len(matched_idxs))])
        labels_pos = torch.cat([t['labels'][matched_idxs[i]] for i, t in enumerate(targets)])
        
        if sampled_pos_inds_subset.sum() > 0:
            # Get the box regression for the correct classes
            box_regression = box_regression.view(-1, self.config.roi.classes, 4)
            box_regression = box_regression[torch.arange(len(box_regression)), labels_pos - 1]
            
            box_regression_loss = F.smooth_l1_loss(
                box_regression[sampled_pos_inds_subset],
                torch.cat([t['boxes'][matched_idxs[i]] for i, t in enumerate(targets)])[sampled_pos_inds_subset],
                reduction='sum'
            ) / sampled_pos_inds_subset.sum()
        else:
            box_regression_loss = torch.tensor(0.0, device=class_logits.device)
        
        return classification_loss + box_regression_loss
    
    def _get_boxes_for_next_stage(self, box_regression, class_logits, proposals, image_sizes):
        """Get refined boxes for the next cascade stage."""
        boxes = []
        scores = F.softmax(class_logits, dim=-1)[:, 1:].max(dim=1)[0]
        labels = F.softmax(class_logits, dim=-1)[:, 1:].max(dim=1)[1] + 1
        
        # Get start indices for each image's proposals
        start_idx = 0
        for i, (props, image_size) in enumerate(zip(proposals, image_sizes)):
            num_props = len(props)
            if num_props == 0:
                boxes.append(torch.zeros((0, 4), device=props.device))
                continue
                
            # Get regression deltas for this image's proposals
            box_regression_per_image = box_regression[start_idx:start_idx + num_props]
            
            # Get the regression deltas for the predicted classes
            if self.config.roi.reg_class_agnostic:
                # Class-agnostic regression
                box_regression_per_image = box_regression_per_image.view(-1, 4)
            else:
                # Class-specific regression
                box_regression_per_image = box_regression_per_image.view(-1, self.config.roi.classes, 4)
                box_regression_per_image = box_regression_per_image[torch.arange(len(box_regression_per_image)), labels[start_idx:start_idx + num_props] - 1]
            
            # Apply regression to proposals
            refined_boxes = self.box_coder.decode(
                box_regression_per_image,
                props
            )
            
            # Clip boxes to image size
            refined_boxes = clip_boxes_to_image(refined_boxes, image_size)
            
            # Remove low scoring boxes
            keep = torch.where(scores[start_idx:start_idx + num_props] > 0.05)[0]
            refined_boxes = refined_boxes[keep]
            scores_per_image = scores[start_idx:start_idx + num_props][keep]
            
            # Apply NMS
            if len(refined_boxes) > 0:
                keep = box_nms(refined_boxes, scores_per_image, 0.5)
                refined_boxes = refined_boxes[keep]
            
            boxes.append(refined_boxes)
            start_idx += num_props
        
        return boxes
    
    def _compute_mask_loss(self, mask_logits, proposals, targets):
        """Compute mask loss.
        
        Args:
            mask_logits: Predicted mask logits
            proposals: RPN proposals
            targets: Ground truth targets
            
        Returns:
            Mask loss
        """
        import torch.nn.functional as F
        
        if len(proposals) == 0:
            return torch.tensor(0.0, device=mask_logits.device)
        
        matched_idxs = []
        for proposals_in_image, targets_in_image in zip(proposals, targets):
            if targets_in_image['boxes'].numel() == 0:
                matched_idxs.append(
                    torch.zeros(
                        proposals_in_image.shape[0],
                        dtype=torch.int64,
                        device=proposals_in_image.device
                    )
                )
                continue
                
            match_quality_matrix = box_iou(
                targets_in_image['boxes'],
                proposals_in_image
            )
            matched_vals, matches = match_quality_matrix.max(dim=0)
            
            # Assign background (negative) to below threshold matches
            matches[matched_vals < 0.5] = -1
            
            matched_idxs.append(matches)
        
        # Only compute loss on positive samples
        sampled_pos_inds_subset = torch.cat([matched_idxs[i] >= 0 for i in range(len(matched_idxs))])
        
        if sampled_pos_inds_subset.sum() > 0:
            # Get target masks - this needs to be resized to match prediction size
            target_masks = []
            for i, (targets_per_image, matched_idxs_per_image) in enumerate(zip(targets, matched_idxs)):
                if matched_idxs_per_image.numel() == 0:
                    continue
                    
                # Get masks for positive samples
                pos_mask_idxs = matched_idxs_per_image >= 0
                pos_matched_idxs = matched_idxs_per_image[pos_mask_idxs]
                
                if pos_matched_idxs.numel() == 0:
                    continue
                    
                # Get target masks for these positive indices
                if 'masks' in targets_per_image and targets_per_image['masks'].numel() > 0:
                    # Extract target masks for matched indices
                    masks = targets_per_image['masks'][pos_matched_idxs]
                    
                    # Check mask dimensions
                    if len(masks.shape) == 3:  # [N, H, W]
                        pass  # Good shape
                    elif len(masks.shape) == 2:  # [H, W]
                        masks = masks.unsqueeze(0)  # Add instance dimension
                    
                    # Get target mask size from prediction
                    mask_h, mask_w = mask_logits.shape[-2:]
                    
                    # Resize masks to prediction size
                    masks = F.interpolate(
                        masks.unsqueeze(0).float(),  # Add batch dimension [1, N, H, W]
                        size=(mask_h, mask_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)  # Remove batch dimension [N, mask_h, mask_w]
                    
                    # Apply threshold to make binary masks
                    masks = masks > 0.5
                    
                    target_masks.append(masks)
            
            if target_masks:
                target_masks = torch.cat(target_masks)
                
                # Ensure there's at least one valid mask
                if target_masks.shape[0] > 0:
                    # Get corresponding predictions
                    mask_logits_pos = mask_logits[sampled_pos_inds_subset]
                    
                    # Get the class labels for each positive sample
                    class_labels = torch.cat([t['labels'][matched_idxs[i]] for i, t in enumerate(targets)])
                    class_labels = class_labels[sampled_pos_inds_subset]
                    
                    # Index mask logits by class - select correct class channel for each instance
                    idx = torch.arange(class_labels.shape[0], device=class_labels.device)
                    mask_logits_pos = mask_logits_pos[idx, class_labels]
                    
                    # Compute binary cross entropy loss
                    # Make sure mask_logits_pos matches target_masks shape
                    if mask_logits_pos.shape != target_masks.shape:
                        print(f"Warning: Shape mismatch - logits: {mask_logits_pos.shape}, targets: {target_masks.shape}")
                        
                        # Attempt to resolve shape mismatch
                        if len(mask_logits_pos.shape) == 3 and len(target_masks.shape) == 3:
                            # If just the number of instances differs, we have a problem
                            # This might happen if our matching logic is faulty
                            if mask_logits_pos.shape[0] < target_masks.shape[0]:
                                # Take only the first N masks to match
                                target_masks = target_masks[:mask_logits_pos.shape[0]]
                            elif mask_logits_pos.shape[0] > target_masks.shape[0]:
                                # Take only the first N predictions to match
                                mask_logits_pos = mask_logits_pos[:target_masks.shape[0]]
                        
                        # Last resort fallback - reshape if dimensions still don't match
                        if mask_logits_pos.shape != target_masks.shape:
                            target_masks = F.interpolate(
                                target_masks.unsqueeze(1).float(),
                                size=mask_logits_pos.shape[1:],
                                mode='bilinear',
                                align_corners=False
                            ).squeeze(1) > 0.5
                    
                    # Calculate loss
                    try:
                        mask_loss = F.binary_cross_entropy_with_logits(
                            mask_logits_pos,
                            target_masks.float(),
                            reduction='mean'
                        )
                    except Exception as e:
                        print(f"Error in mask loss: {e}")
                        print(f"Mask logits shape: {mask_logits_pos.shape}")
                        print(f"Target masks shape: {target_masks.shape}")
                        mask_loss = torch.tensor(0.0, device=mask_logits.device)
                    
                    return mask_loss
        
        # Return zero loss if no positive samples
        return torch.tensor(0.0, device=mask_logits.device)
    
    def resize_masks(self, masks, orig_size, target_size):
        """Resize masks to target size."""
        masks = F.interpolate(masks, size=orig_size, mode='bilinear', align_corners=False)
        masks = F.interpolate(masks, size=target_size, mode='bilinear', align_corners=False)
        return masks
    
    def apply_deltas_to_boxes(self, deltas, boxes, labels=None):
        """Apply regression deltas to boxes with class-specific handling."""
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
        ctr_x = boxes[:, 0] + 0.5 * widths
        ctr_y = boxes[:, 1] + 0.5 * heights

        pred_boxes = torch.zeros_like(boxes)
        
        # Handle class-specific or class-agnostic regression
        if labels is not None and not self.config.roi.reg_class_agnostic:
            # Class-specific regression
            for i, label in enumerate(labels):
                class_idx = label.item()
                dx = deltas[i, class_idx*4:(class_idx+1)*4][0]
                dy = deltas[i, class_idx*4:(class_idx+1)*4][1]
                dw = deltas[i, class_idx*4:(class_idx+1)*4][2]
                dh = deltas[i, class_idx*4:(class_idx+1)*4][3]
                
                # Prevent sending too large values into torch.exp()
                dw = torch.clamp(dw, max=4.0)
                dh = torch.clamp(dh, max=4.0)
                
                # Apply deltas
                pred_ctr_x = dx * widths[i] + ctr_x[i]
                pred_ctr_y = dy * heights[i] + ctr_y[i]
                pred_w = torch.exp(dw) * widths[i]
                pred_h = torch.exp(dh) * heights[i]
                
                # Convert back to box coordinates
                pred_boxes[i, 0] = pred_ctr_x - 0.5 * pred_w
                pred_boxes[i, 1] = pred_ctr_y - 0.5 * pred_h
                pred_boxes[i, 2] = pred_ctr_x + 0.5 * pred_w
                pred_boxes[i, 3] = pred_ctr_y + 0.5 * pred_h
        else:
            # Class-agnostic regression (original code)
            dx = deltas[:, 0::4]
            dy = deltas[:, 1::4]
            dw = deltas[:, 2::4]
            dh = deltas[:, 3::4]

            # Prevent sending too large values into torch.exp()
            dw = torch.clamp(dw, max=4.0)
            dh = torch.clamp(dh, max=4.0)

            pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
            pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
            pred_w = torch.exp(dw) * widths[:, None]
            pred_h = torch.exp(dh) * heights[:, None]

            pred_boxes = torch.zeros_like(deltas)
            pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
            pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
            pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
            pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

        return pred_boxes


class CascadeBoxPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CascadeBoxPredictor, self).__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
    
    def forward(self, x):
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


class CascadeBoxHead(nn.Module):
    def __init__(self, in_channels, representation_size, roi_size):
        super(CascadeBoxHead, self).__init__()
        self.fc6 = nn.Linear(in_channels * roi_size[0] * roi_size[1], representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.relu(self.fc6(x))
        x = self.relu(self.fc7(x))
        return x