# New file: src/models/glomeruli_classifier.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Tuple


class GlomeruliClassifier(nn.Module):
    """
    Classifier for glomeruli types (normal vs. sclerotic).
    
    This model takes pre-detected glomeruli patches and classifies them
    into different categories.
    """
    
    def __init__(self, num_classes: int = 4, in_channels: int = 3, feature_dim: int = 256):
        """
        Initialize glomeruli classifier.
        
        Args:
            num_classes: Number of glomeruli classes
            in_channels: Number of input channels
            feature_dim: Dimension of feature vectors
        """
        super(GlomeruliClassifier, self).__init__()
        
        # Feature extraction backbone
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, num_classes)
        )
        
        # Attention mechanism for feature highlighting
        self.attention = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Feature fusion for manual and deep features
        self.feature_fusion = nn.Sequential(
            nn.Linear(feature_dim + 128, feature_dim),  # 128 for manual features
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor, manual_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input image batch
            manual_features: Optional tensor of manually extracted features
            
        Returns:
            Dictionary with classification outputs
        """
        # Feature extraction
        x = self.features[:14](x)  # Up to block 3
        
        # Apply attention
        attention_map = self.attention(x)
        x = x * attention_map + x  # Residual attention
        
        # Continue feature extraction
        deep_features = self.features[14:](x)  # Block 4 and pooling
        
        # Flatten features
        flat_features = deep_features.view(deep_features.size(0), -1)
        
        # Combine with manual features if provided
        if manual_features is not None:
            # Combine deep and manual features
            combined_features = torch.cat([flat_features, manual_features], dim=1)
            fused_features = self.feature_fusion(combined_features)
        else:
            fused_features = flat_features
        
        # Classification
        logits = self.classifier(fused_features)
        
        return {
            'logits': logits,
            'features': fused_features,
            'attention_map': attention_map
        }


class GlomeruliFeatureExtractor:
    """
    Extract deep features from glomeruli classifier.
    
    This class wraps the GlomeruliClassifier and provides methods
    to extract deep features from images.
    """
    
    def __init__(self, model: GlomeruliClassifier, device: torch.device):
        """
        Initialize feature extractor.
        
        Args:
            model: Trained GlomeruliClassifier model
            device: Device to run inference on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def extract_features(self, image: torch.Tensor, manual_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract features from an image.
        
        Args:
            image: Input image tensor
            manual_features: Optional manual features
            
        Returns:
            Feature vector
        """
        with torch.no_grad():
            # Move to device
            image = image.to(self.device)
            if manual_features is not None:
                manual_features = manual_features.to(self.device)
            
            # Forward pass to extract features
            output = self.model(image, manual_features)
            
            return output['features']