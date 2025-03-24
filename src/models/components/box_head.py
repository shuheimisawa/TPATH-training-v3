# src/models/components/box_head.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class CascadeBoxHead(nn.Module):
    """Box head for cascade stages."""
    
    def __init__(self, in_channels, representation_size, roi_size):
        """Initialize box head.
        
        Args:
            in_channels: Number of input channels
            representation_size: Size of box feature representation
            roi_size: Size of RoI features
        """
        super(CascadeBoxHead, self).__init__()
        
        # Flatten RoI features
        self.flatten = nn.Flatten()
        
        # Box head layers
        self.fc6 = nn.Linear(in_channels * roi_size[0] * roi_size[1], representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        
        # Initialize weights
        for layer in [self.fc6, self.fc7]:
            nn.init.kaiming_uniform_(layer.weight, a=1)
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        """Forward pass of box head.
        
        Args:
            x: RoI features
            
        Returns:
            Box features
        """
        # Flatten features
        x = self.flatten(x)
        
        # Box head
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        
        return x


class CascadeBoxPredictor(nn.Module):
    """Box predictor for cascade stages."""
    
    def __init__(self, in_channels, num_classes):
        """Initialize box predictor.
        
        Args:
            in_channels: Number of input channels
            num_classes: Number of classes including background
        """
        super(CascadeBoxPredictor, self).__init__()
        
        # Box classification layer
        self.cls_score = nn.Linear(in_channels, num_classes)
        
        # Box regression layer
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        
        # Initialize weights
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for layer in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(layer.bias, 0)
    
    def forward(self, x):
        """Forward pass of box predictor.
        
        Args:
            x: Box features
            
        Returns:
            Class logits and box regression
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        
        # Get predictions
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        
        return scores, bbox_deltas 