# src/models/components/enhanced_mask_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union

from ..components.attention import SelfAttention, CBAM


class EnhancedMaskHead(nn.Module):
    """
    Enhanced Mask R-CNN head with higher resolution and better feature processing.
    
    Improvements:
    1. Higher resolution ROI features (28x28 instead of 14x14)
    2. Feature pyramid upsampling path with skip connections
    3. Integration of attention mechanisms
    4. Boundary awareness through edge detection branch
    """
    
    def __init__(
        self, 
        in_channels: int = 256,
        conv_dims: List[int] = [256, 256, 256, 256],
        deconv_dims: List[int] = [256, 256, 128, 64],
        roi_size: Tuple[int, int] = (28, 28),  # Increased from 14x14
        dilation: int = 1,
        num_classes: int = 5,
        use_attention: bool = True,
        attention_type: str = 'cbam',
        boundary_aware: bool = True
    ):
        """
        Initialize enhanced mask head.
        
        Args:
            in_channels: Number of input channels
            conv_dims: Dimensions of convolutional layers
            deconv_dims: Dimensions of deconvolution layers in upsampling path
            roi_size: Size of RoI features
            dilation: Dilation rate for convolutions
            num_classes: Number of classes including background
            use_attention: Whether to use attention mechanisms
            attention_type: Type of attention ('self' or 'cbam')
            boundary_aware: Whether to add boundary detection branch
        """
        super(EnhancedMaskHead, self).__init__()
        
        self.roi_size = roi_size
        self.use_attention = use_attention
        self.attention_type = attention_type
        self.boundary_aware = boundary_aware
        
        # Build convolutional encoder layers
        self.conv_layers = nn.ModuleList()
        next_feature = in_channels
        
        for i, dim in enumerate(conv_dims):
            conv = nn.Conv2d(
                next_feature,
                dim,
                kernel_size=3,
                stride=1,
                padding=dilation,
                dilation=dilation
            )
            self.conv_layers.append(conv)
            self.conv_layers.append(nn.BatchNorm2d(dim))
            self.conv_layers.append(nn.ReLU(inplace=True))
            next_feature = dim
        
        # Build deconvolution decoder layers for upsampling path
        self.deconv_layers = nn.ModuleList()
        for i, dim in enumerate(deconv_dims):
            if i == 0:
                # First layer takes output of convolution layers
                deconv = nn.ConvTranspose2d(
                    conv_dims[-1],
                    dim,
                    kernel_size=2,
                    stride=2,
                    padding=0
                )
            else:
                deconv = nn.ConvTranspose2d(
                    deconv_dims[i-1],
                    dim,
                    kernel_size=2,
                    stride=2,
                    padding=0
                )
            
            self.deconv_layers.append(deconv)
            self.deconv_layers.append(nn.BatchNorm2d(dim))
            self.deconv_layers.append(nn.ReLU(inplace=True))
        
        # Add attention modules if requested
        if use_attention:
            self.attention_modules = nn.ModuleList()
            
            # Add attention after each conv block
            for dim in conv_dims:
                if attention_type == 'self':
                    self.attention_modules.append(SelfAttention(dim))
                elif attention_type == 'cbam':
                    self.attention_modules.append(CBAM(dim))
                else:
                    raise ValueError(f"Unknown attention type: {attention_type}")
        
        # Final mask prediction layers
        final_channels = deconv_dims[-1] if deconv_dims else conv_dims[-1]
        
        # Main mask logits prediction
        self.mask_logits = nn.Conv2d(
            final_channels,
            num_classes,
            kernel_size=1,
            stride=1,
            padding=0
        )
        
        # Boundary detection branch (optional)
        if boundary_aware:
            self.boundary_conv = nn.Sequential(
                nn.Conv2d(final_channels, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes, kernel_size=1)
            )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the mask head using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of enhanced mask head.
        
        Args:
            x: RoI features (B, C, H, W)
            
        Returns:
            If boundary_aware is True: Tuple of (mask_logits, boundary_logits)
            Otherwise: mask_logits only
        """
        # Store encoder feature maps for skip connections
        encoder_features = []
        
        # Encoder path (convolutional layers)
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            
            # Store feature maps after each ReLU layer
            if isinstance(layer, nn.ReLU) and i > 0:
                encoder_features.append(x)
                
                # Apply attention if enabled
                if self.use_attention and len(encoder_features) <= len(self.attention_modules):
                    att_idx = len(encoder_features) - 1
                    x = self.attention_modules[att_idx](x)
        
        # Decoder path with skip connections
        if hasattr(self, 'deconv_layers') and self.deconv_layers:
            # Get feature maps in reverse order for skip connections
            reversed_features = list(reversed(encoder_features))
            
            skip_idx = 0
            for i, layer in enumerate(self.deconv_layers):
                x = layer(x)
                
                # Add skip connection after each deconv block
                if isinstance(layer, nn.ReLU) and skip_idx < len(reversed_features):
                    # Resize skip connection features if needed
                    skip_feature = reversed_features[skip_idx]
                    if skip_feature.shape[2:] != x.shape[2:]:
                        skip_feature = F.interpolate(
                            skip_feature, 
                            size=x.shape[2:],
                            mode='bilinear',
                            align_corners=False
                        )
                    
                    # Add skip connection
                    x = x + skip_feature
                    skip_idx += 1
        
        # Generate mask predictions
        mask_logits = self.mask_logits(x)
        
        # Generate boundary predictions if boundary-aware
        if self.boundary_aware:
            boundary_logits = self.boundary_conv(x)
            return mask_logits, boundary_logits
        
        return mask_logits


class MaskRCNNHeadWithBoundary(nn.Module):
    """
    Mask R-CNN head with boundary detection for improved edge accuracy.
    
    This combines the enhanced mask head with a fusion mechanism to integrate
    boundary information into the final mask predictions.
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        conv_dims: List[int] = [256, 256, 256, 256],
        deconv_dims: List[int] = [256, 256, 128, 64],
        roi_size: Tuple[int, int] = (28, 28),
        num_classes: int = 5,
        use_attention: bool = True,
        attention_type: str = 'cbam'
    ):
        """Initialize mask head with boundary detection."""
        super(MaskRCNNHeadWithBoundary, self).__init__()
        
        # Create enhanced mask head
        self.mask_head = EnhancedMaskHead(
            in_channels=in_channels,
            conv_dims=conv_dims,
            deconv_dims=deconv_dims,
            roi_size=roi_size,
            num_classes=num_classes,
            use_attention=use_attention,
            attention_type=attention_type,
            boundary_aware=True
        )
        
        # Fusion layer to combine mask and boundary predictions
        self.fusion_conv = nn.Conv2d(num_classes * 2, num_classes, kernel_size=1)
        
        # Initialize weights
        nn.init.normal_(self.fusion_conv.weight, std=0.001)
        nn.init.constant_(self.fusion_conv.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with boundary information fusion.
        
        Args:
            x: RoI features
            
        Returns:
            Final mask logits
        """
        # Get mask and boundary predictions
        mask_logits, boundary_logits = self.mask_head(x)
        
        # Apply sigmoid to boundary predictions for attention mechanism
        boundary_attention = torch.sigmoid(boundary_logits)
        
        # Apply boundary-based attention to mask predictions
        attended_mask = mask_logits * (1 + boundary_attention)
        
        # Concatenate original mask logits with boundary-attended ones
        fused_features = torch.cat([mask_logits, attended_mask], dim=1)
        
        # Generate final predictions
        final_mask_logits = self.fusion_conv(fused_features)
        
        return final_mask_logits