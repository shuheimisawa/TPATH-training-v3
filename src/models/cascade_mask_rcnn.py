# src/models/cascade_mask_rcnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
import torchvision
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign

from .backbones.resnet import ResNetBackbone
from .components.fpn import FPN
from .components.bifpn import BiFPN
from .components.attention import SelfAttention, CBAM
from ..config.model_config import ModelConfig


class CascadeMaskRCNN(nn.Module):
    """Cascade Mask R-CNN for glomeruli instance segmentation."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the model.
        
        Args:
            config: Model configuration
        """
        super(CascadeMaskRCNN, self).__init__()
        
        self.config = config
        
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
        
        # Image normalization and size transformation
        self.transform = GeneralizedRCNNTransform(
            min_size=800,
            max_size=1333,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225]
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
                num_classes=config.num_classes
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
        
        if is_training:
            # Add RPN losses
            losses.update(rpn_losses)
            
            # Cascade stages
            all_stage_box_features = []
            all_stage_predictions = []
            all_stage_box_regression = []
            all_stage_class_logits = []
            
            for stage_idx in range(self.num_cascade_stages):
                # Get IoU threshold for this stage
                iou_thresh = self.cascade_iou_thresholds[stage_idx]
                
                # Use boxes from previous stage for stages > 0
                if stage_idx > 0:
                    # Update proposals with refined boxes from previous stage
                    with torch.no_grad():
                        # Create per-image proposals
                        new_proposals = []
                        for i, image_size in enumerate(images.image_sizes):
                            # Get predictions for this image
                            mask = all_stage_predictions[-1]['image_ids'] == i
                            img_boxes = all_stage_predictions[-1]['boxes'][mask]
                            new_proposals.append(img_boxes)
                        proposals = new_proposals
                
                # Extract box features
                box_features = self.box_roi_pool(feature_dict, proposals, images.image_sizes)
                
                # Process box features through box head
                stage_dict = self.cascade_stages[stage_idx]
                box_head = stage_dict['box_head']
                box_predictor = stage_dict['box_predictor']
                
                box_features = box_head(box_features)
                class_logits, box_regression = box_predictor(box_features)
                
                # Store features and predictions
                all_stage_box_features.append(box_features)
                all_stage_class_logits.append(class_logits)
                all_stage_box_regression.append(box_regression)
                
                # Get predictions
                predictions = {
                    'boxes': proposals[0],
                    'scores': F.softmax(class_logits, dim=-1)[:, 1:].max(dim=1)[0],
                    'labels': F.softmax(class_logits, dim=-1)[:, 1:].max(dim=1)[1] + 1
                }
                all_stage_predictions.append(predictions)
                
                # Calculate and add box loss for this stage
                stage_weight = self.cascade_loss_weights[stage_idx]
                box_loss = F.smooth_l1_loss(box_regression, targets[0]['boxes'], reduction='sum')
                cls_loss = F.cross_entropy(class_logits, targets[0]['labels'])
                
                losses[f'stage{stage_idx}_box_loss'] = box_loss * stage_weight
                losses[f'stage{stage_idx}_cls_loss'] = cls_loss * stage_weight
            
            # Use boxes from the last stage
            boxes = all_stage_predictions[-1]['boxes']
            labels = all_stage_predictions[-1]['labels']
            
            # Get mask features
            mask_features = self.mask_roi_pool(feature_dict, [boxes], images.image_sizes)
            
            # Process mask features
            mask_logits = self.mask_head(mask_features, labels)
            
            # Calculate mask loss
            mask_loss = F.binary_cross_entropy_with_logits(mask_logits, targets[0]['masks'])
            losses['mask_loss'] = mask_loss
            
            return losses
        else:
            # Inference mode
            # Similar to training but without loss calculation
            # Use boxes from the last stage for mask prediction
            predictions = []
            
            # Process each image independently (batch size is typically 1 during inference)
            for idx, (image_size, image_proposals) in enumerate(zip(images.image_sizes, proposals)):
                # Initialize boxes, scores, and labels
                boxes = image_proposals
                scores = torch.ones(boxes.shape[0], device=boxes.device)
                labels = torch.ones(boxes.shape[0], dtype=torch.int64, device=boxes.device)
                
                # Cascade stages
                for stage_idx in range(self.num_cascade_stages):
                    # Extract box features
                    box_features = self.box_roi_pool(feature_dict, [boxes], [image_size])
                    
                    # Process box features through box head
                    stage_dict = self.cascade_stages[stage_idx]
                    box_head = stage_dict['box_head']
                    box_predictor = stage_dict['box_predictor']
                    
                    box_features = box_head(box_features)
                    class_logits, box_regression = box_predictor(box_features)
                    
                    # Update boxes, scores, and labels
                    pred_boxes = self.apply_deltas_to_boxes(box_regression, boxes)
                    pred_scores = F.softmax(class_logits, dim=-1)
                    
                    # Keep only predictions with score > threshold
                    keep_idxs = pred_scores[:, 1:].max(dim=1)[0] > 0.05
                    boxes = pred_boxes[keep_idxs]
                    scores = pred_scores[keep_idxs, 1:].max(dim=1)[0]
                    labels = pred_scores[keep_idxs, 1:].max(dim=1)[1] + 1
                    
                    if boxes.shape[0] == 0:
                        # No detections
                        result = {
                            'boxes': torch.zeros((0, 4), device=boxes.device),
                            'labels': torch.zeros(0, dtype=torch.int64, device=boxes.device),
                            'scores': torch.zeros(0, device=boxes.device),
                            'masks': torch.zeros((0, 1, *image_size), device=boxes.device)
                        }
                        predictions.append(result)
                        continue
                
                # Get mask features for final boxes
                mask_features = self.mask_roi_pool(feature_dict, [boxes], [image_size])
                
                # Process mask features
                mask_logits = self.mask_head(mask_features, labels)
                
                # Convert mask logits to binary masks
                masks = (mask_logits > 0).float()
                
                # Resize masks to original image size
                masks = self.resize_masks(masks, image_size, original_image_sizes[idx])
                
                # Apply NMS
                keep = torchvision.ops.nms(boxes, scores, 0.5)
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
                masks = masks[keep]
                
                # Create result dictionary
                result = {
                    'boxes': boxes,
                    'labels': labels,
                    'scores': scores,
                    'masks': masks
                }
                
                predictions.append(result)
            
            return predictions
    
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


class CascadeBoxHead(nn.Module):
    """Box head for Cascade R-CNN."""
    
    def __init__(self, in_channels, representation_size, roi_size):
        """Initialize the box head.
        
        Args:
            in_channels: Number of input channels
            representation_size: Size of the intermediate representation
            roi_size: Size of the RoI features
        """
        super(CascadeBoxHead, self).__init__()
        
        roi_height, roi_width = roi_size
        
        self.fc6 = nn.Linear(in_channels * roi_height * roi_width, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
    
    def forward(self, x):
        """Forward pass of the box head.
        
        Args:
            x: RoI features
            
        Returns:
            Box features
        """
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x


class CascadeBoxPredictor(nn.Module):
    """Box predictor for Cascade R-CNN."""
    
    def __init__(self, in_channels, num_classes):
        """Initialize the box predictor.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of classes including background
        """
        super(CascadeBoxPredictor, self).__init__()
        
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
    
    def forward(self, x):
        """Forward pass of the box predictor.
        
        Args:
            x: Box features
            
        Returns:
            Class logits and box regression deltas
        """
        cls_scores = self.cls_score(x)
        bbox_preds = self.bbox_pred(x)
        
        return cls_scores, bbox_preds


class MaskRCNNHeadWithAttention(nn.Module):
    """Enhanced Mask head for Mask R-CNN with attention."""
    
    def __init__(self, in_channels, layers, dilation, roi_size, num_classes, 
                 use_attention=False, attention_type='self'):
        """Initialize the mask head.
        
        Args:
            in_channels: Number of input channels
            layers: Number of channels in each convolutional layer
            dilation: Dilation rate of the convolutional layers
            roi_size: Size of the RoI features
            num_classes: Number of classes including background
            use_attention: Whether to use attention mechanism
            attention_type: Type of attention to use ('self', 'cbam')
        """
        super(MaskRCNNHeadWithAttention, self).__init__()
        
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.attention_type = attention_type
        
        # Build convolutional layers
        self.conv_layers = nn.ModuleList()
        next_channels = in_channels
        
        for layer_channels in layers:
            self.conv_layers.append(
                nn.Conv2d(
                    next_channels,
                    layer_channels,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation
                )
            )
            next_channels = layer_channels
        
        # Add attention module if requested
        if use_attention:
            if attention_type == 'self':
                self.attention = SelfAttention(next_channels)
            elif attention_type == 'cbam':
                self.attention = CBAM(next_channels)
            else:
                self.attention = None
        
        # Final layer
        self.mask_predictor = nn.Conv2d(
            next_channels,
            num_classes,
            kernel_size=1,
            stride=1
        )
    
    def forward(self, x, labels):
        """Forward pass of the mask head.
        
        Args:
            x: RoI features
            labels: Class labels
            
        Returns:
            Mask logits
        """
        for conv in self.conv_layers:
            x = F.relu(conv(x))
        
        # Apply attention if enabled
        if self.use_attention and hasattr(self, 'attention'):
            x = self.attention(x)
        
        mask_logits = self.mask_predictor(x)
        
        if labels is not None:
            # During training or inference with given labels
            # Select the mask corresponding to the predicted class
            mask_logits = mask_logits[torch.arange(mask_logits.shape[0], device=mask_logits.device), labels]
            mask_logits = mask_logits.unsqueeze(1)
        
        return mask_logits