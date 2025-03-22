import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Union, Optional

from .attention import SelfAttention, CBAM


class BiFPNBlock(nn.Module):
    """Bidirectional Feature Pyramid Network (BiFPN) block.
    
    BiFPN is an enhanced version of Feature Pyramid Network (FPN) that uses
    bidirectional cross-scale connections and weighted feature fusion to
    improve feature fusion.
    
    Reference: EfficientDet: Scalable and Efficient Object Detection (CVPR 2020)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        attention_type: str = 'none',
        epsilon: float = 1e-4
    ):
        """Initialize the BiFPN block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            attention_type: Type of attention to use ('none', 'self', 'cbam')
            epsilon: Small constant to avoid division by zero
        """
        super(BiFPNBlock, self).__init__()
        
        self.epsilon = epsilon
        
        # Convolutions for each lateral and top-down path
        # For input feature maps P3, P4, P5, P6, P7
        self.conv_p3_td = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_p4_td = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_p5_td = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv_p6_td = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        
        # Top-down path
        self.conv_p4_out_td = self._build_conv_block(out_channels)
        self.conv_p5_out_td = self._build_conv_block(out_channels)
        self.conv_p6_out_td = self._build_conv_block(out_channels)
        self.conv_p7_out_td = self._build_conv_block(out_channels)
        
        # Bottom-up path
        self.conv_p3_out = self._build_conv_block(out_channels)
        self.conv_p4_out = self._build_conv_block(out_channels)
        self.conv_p5_out = self._build_conv_block(out_channels)
        self.conv_p6_out = self._build_conv_block(out_channels)
        self.conv_p7_out = self._build_conv_block(out_channels)
        
        # Fusion weights (learnable)
        self.p4_td_w = nn.Parameter(torch.ones(2))
        self.p5_td_w = nn.Parameter(torch.ones(2))
        self.p6_td_w = nn.Parameter(torch.ones(2))
        self.p7_td_w = nn.Parameter(torch.ones(1))
        
        self.p3_w = nn.Parameter(torch.ones(1))
        self.p4_w = nn.Parameter(torch.ones(2))
        self.p5_w = nn.Parameter(torch.ones(2))
        self.p6_w = nn.Parameter(torch.ones(2))
        self.p7_w = nn.Parameter(torch.ones(2))
        
        # Attention modules
        self.attention_type = attention_type
        if attention_type == 'self':
            self.attention_modules = nn.ModuleList([
                SelfAttention(out_channels) for _ in range(5)
            ])
        elif attention_type == 'cbam':
            self.attention_modules = nn.ModuleList([
                CBAM(out_channels) for _ in range(5)
            ])
        
        # Initialize weights
        self._init_weights()
    
    def _build_conv_block(self, channels: int) -> nn.Sequential:
        """Build a convolutional block with batch normalization and activation.
        
        Args:
            channels: Number of channels
            
        Returns:
            Sequential block of Conv2d + BatchNorm2d + ReLU
        """
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
    
    def _init_weights(self):
        """Initialize module weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _weighted_fusion(self, inputs: List[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        """Perform weighted fusion of feature maps.
        
        Args:
            inputs: List of input feature maps
            weights: Fusion weights
            
        Returns:
            Weighted sum of input feature maps
        """
        # Apply softmax to weights
        weights = F.softmax(weights, dim=0)
        
        # Weighted fusion
        output = 0
        for i, inp in enumerate(inputs):
            output = output + weights[i] * inp
            
        return output
    
    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass of the BiFPN block.
        
        Args:
            inputs: List of input feature maps [P3, P4, P5, P6, P7]
                where P3 has the highest resolution
                
        Returns:
            List of output feature maps [P3_out, P4_out, P5_out, P6_out, P7_out]
        """
        # Input feature maps
        p3, p4, p5, p6, p7 = inputs
        
        # Initial 1x1 convolutions
        p3_in = self.conv_p3_td(p3)
        p4_in = self.conv_p4_td(p4)
        p5_in = self.conv_p5_td(p5)
        p6_in = self.conv_p6_td(p6)
        p7_in = p7  # No need for initial conv for P7
        
        # Top-down path (from lower resolution to higher resolution)
        # P7_td = P7
        p7_td = p7_in
        p7_td = self.conv_p7_out_td(p7_td)
        
        # P6_td = Conv(Upsample(P7_td) + P6)
        p6_td_inputs = [p6_in]
        p6_td_inputs.append(F.interpolate(p7_td, size=p6_in.shape[2:], mode='nearest'))
        p6_td = self._weighted_fusion(p6_td_inputs, self.p6_td_w)
        p6_td = self.conv_p6_out_td(p6_td)
        
        # P5_td = Conv(Upsample(P6_td) + P5)
        p5_td_inputs = [p5_in]
        p5_td_inputs.append(F.interpolate(p6_td, size=p5_in.shape[2:], mode='nearest'))
        p5_td = self._weighted_fusion(p5_td_inputs, self.p5_td_w)
        p5_td = self.conv_p5_out_td(p5_td)
        
        # P4_td = Conv(Upsample(P5_td) + P4)
        p4_td_inputs = [p4_in]
        p4_td_inputs.append(F.interpolate(p5_td, size=p4_in.shape[2:], mode='nearest'))
        p4_td = self._weighted_fusion(p4_td_inputs, self.p4_td_w)
        p4_td = self.conv_p4_out_td(p4_td)
        
        # Bottom-up path (from higher resolution to lower resolution)
        # P3_out = Conv(P3 + P4_td)
        p3_out_inputs = [p3_in, F.interpolate(p4_td, size=p3_in.shape[2:], mode='nearest')]
        p3_out = self._weighted_fusion(p3_out_inputs, self.p3_w)
        p3_out = self.conv_p3_out(p3_out)
        
        # P4_out = Conv(P4 + P4_td + Downsample(P3_out))
        p4_out_inputs = [p4_in, p4_td]
        p4_out_inputs.append(F.max_pool2d(p3_out, kernel_size=2))
        p4_out = self._weighted_fusion(p4_out_inputs, self.p4_w)
        p4_out = self.conv_p4_out(p4_out)
        
        # P5_out = Conv(P5 + P5_td + Downsample(P4_out))
        p5_out_inputs = [p5_in, p5_td]
        p5_out_inputs.append(F.max_pool2d(p4_out, kernel_size=2))
        p5_out = self._weighted_fusion(p5_out_inputs, self.p5_w)
        p5_out = self.conv_p5_out(p5_out)
        
        # P6_out = Conv(P6 + P6_td + Downsample(P5_out))
        p6_out_inputs = [p6_in, p6_td]
        p6_out_inputs.append(F.max_pool2d(p5_out, kernel_size=2))
        p6_out = self._weighted_fusion(p6_out_inputs, self.p6_w)
        p6_out = self.conv_p6_out(p6_out)
        
        # P7_out = Conv(P7 + P7_td + Downsample(P6_out))
        p7_out_inputs = [p7_in, p7_td]
        p7_out_inputs.append(F.max_pool2d(p6_out, kernel_size=2))
        p7_out = self._weighted_fusion(p7_out_inputs, self.p7_w)
        p7_out = self.conv_p7_out(p7_out)
        
        # Apply attention if specified
        outputs = [p3_out, p4_out, p5_out, p6_out, p7_out]
        
        if self.attention_type in ['self', 'cbam']:
            for i in range(len(outputs)):
                outputs[i] = self.attention_modules[i](outputs[i])
        
        return outputs


class BiFPN(nn.Module):
    """Bidirectional Feature Pyramid Network.
    
    BiFPN enhances feature fusion through bidirectional cross-scale connections
    and weighted feature fusion.
    """
    
    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        num_blocks: int = 3,
        attention_type: str = 'none',
        extra_convs_on_inputs: bool = True
    ):
        """Initialize the BiFPN.
        
        Args:
            in_channels: List of input channel sizes for each level
            out_channels: Number of channels for all BiFPN levels
            num_blocks: Number of BiFPN blocks
            attention_type: Type of attention to use ('none', 'self', 'cbam')
            extra_convs_on_inputs: Whether to add extra downsampling to inputs
        """
        super(BiFPN, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.extra_convs_on_inputs = extra_convs_on_inputs
        
        # Input projection layers
        self.lateral_convs = nn.ModuleList()
        for c in self.in_channels:
            self.lateral_convs.append(
                nn.Conv2d(c, out_channels, kernel_size=1)
            )
        
        # Extra input convolutions for downsampling
        if self.extra_convs_on_inputs:
            self.downsample_convs = nn.ModuleList()
            for i in range(2):  # Create two extra levels (P6, P7)
                if i == 0:
                    # P6 from P5
                    self.downsample_convs.append(
                        nn.Conv2d(in_channels[-1], out_channels, kernel_size=3, stride=2, padding=1)
                    )
                else:
                    # P7 from P6 (using ReLU before)
                    self.downsample_convs.append(nn.Sequential(
                        nn.ReLU(inplace=True),
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
                    ))
        
        # BiFPN blocks
        self.bifpn_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.bifpn_blocks.append(
                BiFPNBlock(out_channels, out_channels, attention_type)
            )
    
    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass of the BiFPN.
        
        Args:
            inputs: List of feature maps from the backbone
            
        Returns:
            List of enhanced feature maps
        """
        # Ensure correct number of inputs
        assert len(inputs) == len(self.in_channels)
        
        # Apply lateral convs to each input
        features = [lat_conv(inputs[i]) for i, lat_conv in enumerate(self.lateral_convs)]
        
        # Add extra levels by downsampling
        if self.extra_convs_on_inputs:
            # P6 from P5
            p6 = self.downsample_convs[0](inputs[-1])
            features.append(p6)
            
            # P7 from P6
            p7 = self.downsample_convs[1](p6)
            features.append(p7)
        
        # Apply BiFPN blocks
        for bifpn_block in self.bifpn_blocks:
            features = bifpn_block(features)
        
        return tuple(features)