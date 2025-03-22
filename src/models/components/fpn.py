import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple


class FPN(nn.Module):
    """Feature Pyramid Network for feature extraction."""
    
    def __init__(self,
                 in_channels: List[int] = (256, 512, 1024, 2048),
                 out_channels: int = 256,
                 num_outs: int = 5,
                 add_extra_convs: bool = True,
                 extra_convs_on_inputs: bool = False):
        """Initialize FPN.
        
        Args:
            in_channels: Number of channels for each input feature map
            out_channels: Number of channels for output feature maps
            num_outs: Number of output feature maps
            add_extra_convs: Whether to add extra convolutions for higher-level features
            extra_convs_on_inputs: Whether to apply extra convs on input features
        """
        super(FPN, self).__init__()
        
        assert len(in_channels) > 1, "FPN requires at least 2 input feature maps"
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs
        
        # Build lateral connections
        self.lateral_convs = nn.ModuleList()
        # Build top-down connections
        self.fpn_convs = nn.ModuleList()
        
        for i in range(self.num_ins):
            lateral_conv = nn.Conv2d(
                in_channels[i], out_channels, kernel_size=1)
            fpn_conv = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1)
            
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
        
        # Extra convolutions for higher-level features
        if self.add_extra_convs:
            self.extra_convs = nn.ModuleList()
            for i in range(self.num_outs - self.num_ins):
                if i == 0 and self.extra_convs_on_inputs:
                    in_channels = self.in_channels[-1]
                else:
                    in_channels = out_channels
                
                extra_conv = nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=2, padding=1)
                self.extra_convs.append(extra_conv)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights of the FPN."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass of the FPN.
        
        Args:
            inputs: List of feature maps from the backbone
            
        Returns:
            List of FPN feature maps
        """
        assert len(inputs) == len(self.in_channels)
        
        # Build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode='nearest'
            )
        
        # Build outputs
        outs = [fpn_conv(laterals[i]) for i, fpn_conv in enumerate(self.fpn_convs)]
        
        # Add extra levels
        if self.num_outs > len(outs):
            if self.add_extra_convs:
                if self.extra_convs_on_inputs:
                    source = inputs[-1]
                else:
                    source = outs[-1]
                
                for i, extra_conv in enumerate(self.extra_convs):
                    outs.append(extra_conv(source if i == 0 else outs[-1]))
            else:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        
        return tuple(outs)