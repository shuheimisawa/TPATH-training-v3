import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SelfAttention(nn.Module):
    """Self-attention module for convolutional feature maps.
    
    This implementation uses the non-local self-attention mechanism to capture
    long-range dependencies in the feature maps. The attention maps are computed
    by dot product similarity between query and key features, and the output is
    a weighted sum of value features based on the attention maps.
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        """Initialize the self-attention module.
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Reduction ratio for query and key projections
                             to reduce computational cost
        """
        super(SelfAttention, self).__init__()
        
        # Calculate reduced channel dimension
        self.reduced_channels = max(in_channels // reduction_ratio, 1)
        
        # Define projections
        self.query = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, self.reduced_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # Learnable scaling parameter, initialized to zero
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the self-attention module.
        
        Args:
            x: Input feature map of shape (batch_size, channels, height, width)
            
        Returns:
            Attended feature map of the same shape as input
        """
        batch_size, channels, height, width = x.size()
        
        # Project input to query, key, and value
        proj_query = self.query(x)  # (B, C', H, W)
        proj_key = self.key(x)     # (B, C', H, W)
        proj_value = self.value(x)  # (B, C, H, W)
        
        # Reshape for matrix multiplication
        # Query: (B, C', H*W) -> (B, H*W, C')
        proj_query = proj_query.view(batch_size, self.reduced_channels, -1).permute(0, 2, 1)
        
        # Key: (B, C', H*W)
        proj_key = proj_key.view(batch_size, self.reduced_channels, -1)
        
        # Value: (B, C, H*W)
        proj_value = proj_value.view(batch_size, channels, -1)
        
        # Compute attention map via matrix multiplication
        # Energy: (B, H*W, H*W)
        energy = torch.bmm(proj_query, proj_key)
        
        # Normalize attention weights
        attention = F.softmax(energy, dim=2)
        
        # Apply attention to value
        # Out: (B, C, H*W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        
        # Reshape back to feature map format
        out = out.view(batch_size, channels, height, width)
        
        # Apply scaling and residual connection
        out = self.gamma * out + x
        
        return out


class ChannelAttention(nn.Module):
    """Channel attention module for feature maps.
    
    This module applies attention across channels using the Squeeze-and-Excitation (SE) mechanism.
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        """Initialize the channel attention module.
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Reduction ratio for the fully connected layers
        """
        super(ChannelAttention, self).__init__()
        
        # Calculate reduced channel dimension
        reduced_channels = max(in_channels // reduction_ratio, 1)
        
        # Squeeze and excitation network
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the channel attention module.
        
        Args:
            x: Input feature map of shape (batch_size, channels, height, width)
            
        Returns:
            Attended feature map of the same shape as input
        """
        # Apply channel attention
        attention = self.se(x)
        
        # Apply attention weights
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial attention module for feature maps.
    
    This module applies attention across the spatial dimensions of feature maps.
    """
    
    def __init__(self, kernel_size: int = 7):
        """Initialize the spatial attention module.
        
        Args:
            kernel_size: Size of the convolutional kernel for spatial attention
        """
        super(SpatialAttention, self).__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the spatial attention module.
        
        Args:
            x: Input feature map of shape (batch_size, channels, height, width)
            
        Returns:
            Attended feature map of the same shape as input
        """
        # Calculate max and mean along channel dimension
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        
        # Concatenate pooled features
        spatial = torch.cat([max_pool, avg_pool], dim=1)
        
        # Apply spatial attention
        spatial = self.conv(spatial)
        spatial = self.sigmoid(spatial)
        
        # Apply attention weights
        return x * spatial


class CBAM(nn.Module):
    """Convolutional Block Attention Module (CBAM).
    
    This module combines channel and spatial attention for enhanced feature refinement.
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16, kernel_size: int = 7):
        """Initialize the CBAM module.
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Reduction ratio for channel attention
            kernel_size: Kernel size for spatial attention
        """
        super(CBAM, self).__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CBAM module.
        
        Args:
            x: Input feature map of shape (batch_size, channels, height, width)
            
        Returns:
            Attended feature map of the same shape as input
        """
        # Apply channel attention then spatial attention
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        
        return x