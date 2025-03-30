# src/models/backbones/resnet.py
import torch
import torch.nn as nn
import torchvision
from typing import List, Dict, Tuple


class ResNetBackbone(nn.Module):
    """ResNet backbone with Feature Pyramid Network (FPN) integration."""
    
    def __init__(self, 
                 name: str = 'resnet50',
                 pretrained: bool = True,
                 freeze_stages: int = 1,
                 norm_eval: bool = True,
                 out_indices: Tuple[int, ...] = (0, 1, 2, 3)):
        """Initialize ResNet backbone.
        
        Args:
            name: Name of the ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152')
            pretrained: Whether to use pretrained weights
            freeze_stages: Number of stages to freeze (1: stem, 2: stem+layer1, ...)
            norm_eval: Whether to set BN layers to eval mode during training
            out_indices: Indices of stages to output
        """
        super(ResNetBackbone, self).__init__()
        
        self.name = name
        self.freeze_stages = freeze_stages
        self.norm_eval = norm_eval
        self.out_indices = out_indices
        
        # Create ResNet model
        model_func = getattr(torchvision.models, name)
        
        # Get the correct weights class based on the model name
        if pretrained:
            if name == 'resnet18':
                weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            elif name == 'resnet34':
                weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1
            elif name == 'resnet50':
                weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V1
            elif name == 'resnet101':
                weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V1
            elif name == 'resnet152':
                weights = torchvision.models.ResNet152_Weights.IMAGENET1K_V1
            else:
                raise ValueError(f"Unsupported ResNet variant: {name}")
        else:
            weights = None
            
        resnet = model_func(weights=weights)
        
        # Remove fully connected layer and average pooling
        self.resnet = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        )
        
        # Get output channels for each stage
        if name in ['resnet18', 'resnet34']:
            self.out_channels = [64, 128, 256, 512]
        else:  # ResNet50, ResNet101, ResNet152
            self.out_channels = [256, 512, 1024, 2048]
        
        # Freeze stages if needed
        self._freeze_stages()
    
    def _freeze_stages(self):
        """Freeze stages of the ResNet."""
        if self.freeze_stages >= 1:
            # Freeze stem
            for param in self.resnet[0].parameters():
                param.requires_grad = False
            for param in self.resnet[1].parameters():
                param.requires_grad = False
        
        if self.freeze_stages >= 2:
            # Freeze layer1
            for param in self.resnet[4].parameters():
                param.requires_grad = False
        
        if self.freeze_stages >= 3:
            # Freeze layer2
            for param in self.resnet[5].parameters():
                param.requires_grad = False
        
        if self.freeze_stages >= 4:
            # Freeze layer3
            for param in self.resnet[6].parameters():
                param.requires_grad = False
        
        if self.freeze_stages >= 5:
            # Freeze layer4
            for param in self.resnet[7].parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass of the ResNet backbone.
        
        Args:
            x: Input tensor
            
        Returns:
            List of feature maps
        """
        outs = []
        
        # Stem
        x = self.resnet[0](x)  # conv1
        x = self.resnet[1](x)  # bn1
        x = self.resnet[2](x)  # relu
        x = self.resnet[3](x)  # maxpool
        
        # Layers
        x = self.resnet[4](x)  # layer1
        if 0 in self.out_indices:
            outs.append(x)
        
        x = self.resnet[5](x)  # layer2
        if 1 in self.out_indices:
            outs.append(x)
        
        x = self.resnet[6](x)  # layer3
        if 2 in self.out_indices:
            outs.append(x)
        
        x = self.resnet[7](x)  # layer4
        if 3 in self.out_indices:
            outs.append(x)
        
        return tuple(outs)
    
    def train(self, mode: bool = True):
        """Set the module in training mode.
        
        Args:
            mode: Whether to set training mode (True) or evaluation mode (False)
        """
        super(ResNetBackbone, self).train(mode)
        
        if mode and self.norm_eval:
            # Set BN layers to eval mode
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    for param in m.parameters():
                        param.requires_grad = False