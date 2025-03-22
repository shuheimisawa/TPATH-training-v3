import os
import sys
import unittest
import torch

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.model_config import ModelConfig
from src.models.backbones.resnet import ResNetBackbone
from src.models.components.fpn import FPN
from src.models.cascade_mask_rcnn import CascadeMaskRCNN, CascadeBoxHead, CascadeBoxPredictor, MaskRCNNHead


class TestBackbone(unittest.TestCase):
    """Test cases for the backbone model."""
    
    def test_resnet_backbone(self):
        """Test ResNet backbone."""
        backbone = ResNetBackbone(
            name='resnet18',
            pretrained=False,
            freeze_stages=0
        )
        
        # Test forward pass
        x = torch.randn(2, 3, 224, 224)
        features = backbone(x)
        
        # Check output
        self.assertEqual(len(features), 4)
        self.assertEqual(features[0].shape, (2, 64, 56, 56))
        self.assertEqual(features[1].shape, (2, 128, 28, 28))
        self.assertEqual(features[2].shape, (2, 256, 14, 14))
        self.assertEqual(features[3].shape, (2, 512, 7, 7))


class TestFPN(unittest.TestCase):
    """Test cases for the FPN component."""
    
    def test_fpn(self):
        """Test FPN forward pass."""
        fpn = FPN(
            in_channels=[64, 128, 256, 512],
            out_channels=256,
            num_outs=5
        )
        
        # Create dummy features
        c2 = torch.randn(2, 64, 56, 56)
        c3 = torch.randn(2, 128, 28, 28)
        c4 = torch.randn(2, 256, 14, 14)
        c5 = torch.randn(2, 512, 7, 7)
        features = [c2, c3, c4, c5]
        
        # Test forward pass
        outputs = fpn(features)
        
        # Check output
        self.assertEqual(len(outputs), 5)
        self.assertEqual(outputs[0].shape, (2, 256, 56, 56))
        self.assertEqual(outputs[1].shape, (2, 256, 28, 28))
        self.assertEqual(outputs[2].shape, (2, 256, 14, 14))
        self.assertEqual(outputs[3].shape, (2, 256, 7, 7))
        self.assertEqual(outputs[4].shape, (2, 256, 4, 4))  # Extra level


class TestCascadeMaskRCNN(unittest.TestCase):
    """Test cases for the Cascade Mask R-CNN model."""
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_model_cuda(self):
        """Test model on CUDA."""
        model_config = ModelConfig()
        model = CascadeMaskRCNN(model_config)
        model = model.cuda()
        
        # Test that model parameters are on CUDA
        self.assertTrue(next(model.parameters()).is_cuda)
    
    def test_box_head(self):
        """Test CascadeBoxHead."""
        box_head = CascadeBoxHead(
            in_channels=256,
            representation_size=1024,
            roi_size=(7, 7)
        )
        
        # Test forward pass
        x = torch.randn(10, 256, 7, 7)
        features = box_head(x)
        
        # Check output
        self.assertEqual(features.shape, (10, 1024))
    
    def test_box_predictor(self):
        """Test CascadeBoxPredictor."""
        box_predictor = CascadeBoxPredictor(
            in_channels=1024,
            num_classes=4
        )
        
        # Test forward pass
        x = torch.randn(10, 1024)
        cls_scores, bbox_preds = box_predictor(x)
        
        # Check output
        self.assertEqual(cls_scores.shape, (10, 4))  # 4 classes
        self.assertEqual(bbox_preds.shape, (10, 16))  # 4 classes * 4 coords
    
    def test_mask_head(self):
        """Test MaskRCNNHead."""
        mask_head = MaskRCNNHead(
            in_channels=256,
            layers=(256, 256, 256, 256),
            dilation=1,
            roi_size=(14, 14),
            num_classes=4
        )
        
        # Test forward pass with labels
        x = torch.randn(10, 256, 14, 14)
        labels = torch.randint(1, 4, (10,))
        mask_logits = mask_head(x, labels)
        
        # Check output
        self.assertEqual(mask_logits.shape, (10, 1, 14, 14))
    
    def test_model_initialization(self):
        """Test model initialization."""
        model_config = ModelConfig()
        model = CascadeMaskRCNN(model_config)
        
        # Check model attributes
        self.assertEqual(len(model.cascade_stages), model_config.cascade.num_stages)
        self.assertEqual(model.transform.image_mean, [0.485, 0.456, 0.406])
        self.assertEqual(model.transform.image_std, [0.229, 0.224, 0.225])