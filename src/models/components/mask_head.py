# src/models/components/mask_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import SelfAttention, CBAM


class MaskRCNNHeadWithAttention(nn.Module):
    """Mask R-CNN head with optional attention."""
    
    def __init__(self, in_channels, layers=(256, 256, 256, 256), dilation=1, roi_size=(14, 14),
                 num_classes=5, use_attention=False, attention_type='self'):
        """Initialize mask head.
        
        Args:
            in_channels: Number of input channels
            layers: Number of channels in each conv layer
            dilation: Dilation rate for convolutions
            roi_size: Size of RoI features
            num_classes: Number of classes including background
            use_attention: Whether to use attention
            attention_type: Type of attention ('self' or 'cbam')
        """
        super(MaskRCNNHeadWithAttention, self).__init__()
        
        # Build convolutional layers
        next_feature = in_channels
        self.conv_layers = []
        
        for layer_features in layers:
            self.conv_layers.append(
                nn.Conv2d(
                    next_feature,
                    layer_features,
                    kernel_size=3,
                    stride=1,
                    padding=dilation,
                    dilation=dilation
                )
            )
            self.conv_layers.append(nn.ReLU(inplace=True))
            next_feature = layer_features
        
        # Convert list to ModuleList
        self.conv_layers = nn.ModuleList(self.conv_layers)
        
        # Add attention if requested
        self.use_attention = use_attention
        if use_attention:
            if attention_type == 'self':
                self.attention = SelfAttention(next_feature)
            elif attention_type == 'cbam':
                self.attention = CBAM(next_feature)
            else:
                raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Mask predictor
        self.mask_predictor = nn.ConvTranspose2d(
            next_feature,
            next_feature,
            kernel_size=2,
            stride=2,
            padding=0
        )
        self.mask_logits = nn.Conv2d(next_feature, num_classes, kernel_size=1, stride=1)
        
        # Initialize weights
        for layer in self.conv_layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(layer.bias, 0)
        
        nn.init.kaiming_normal_(self.mask_predictor.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.mask_predictor.bias, 0)
        
        nn.init.normal_(self.mask_logits.weight, std=0.001)
        nn.init.constant_(self.mask_logits.bias, 0)
    
    def forward(self, x):
        """Forward pass of mask head.
        
        Args:
            x: RoI features
            
        Returns:
            Mask logits
        """
        # Apply conv layers
        for layer in self.conv_layers:
            x = layer(x)
        
        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x)
        
        # Generate mask predictions
        x = F.relu(self.mask_predictor(x))
        x = self.mask_logits(x)
        
        return x 