import os
import sys
import unittest
import torch
import torch.nn as nn
import shutil
from tempfile import TemporaryDirectory

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.training_config import TrainingConfig
from src.training.loss import MaskRCNNLoss
from src.training.optimization import create_optimizer, create_lr_scheduler
from src.training.callbacks import ModelCheckpoint, EarlyStopping


class DummyModel(nn.Module):
    """Dummy model for testing."""
    
    def __init__(self):
        super(DummyModel, self).__init__()
        self.conv = nn.Conv2d(3, 10, kernel_size=3, padding=1)
        self.fc = nn.Linear(10, 4)
    
    def forward(self, x):
        x = self.conv(x)
        x = x.mean(dim=(2, 3))
        x = self.fc(x)
        return x


class TestOptimization(unittest.TestCase):
    """Test cases for optimization utilities."""
    
    def test_create_optimizer(self):
        """Test optimizer creation."""
        model = DummyModel()
        
        # Test SGD
        config = {'type': 'sgd', 'lr': 0.01, 'momentum': 0.9, 'weight_decay': 0.0001}
        optimizer = create_optimizer(model.parameters(), config)
        self.assertIsInstance(optimizer, torch.optim.SGD)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.01)
        
        # Test Adam
        config = {'type': 'adam', 'lr': 0.001, 'weight_decay': 0.0001}
        optimizer = create_optimizer(model.parameters(), config)
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.001)
        
        # Test AdamW
        config = {'type': 'adamw', 'lr': 0.0001, 'weight_decay': 0.01}
        optimizer = create_optimizer(model.parameters(), config)
        self.assertIsInstance(optimizer, torch.optim.AdamW)
        self.assertEqual(optimizer.param_groups[0]['lr'], 0.0001)
    
    def test_create_lr_scheduler(self):
        """Test learning rate scheduler creation."""
        model = DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Test StepLR
        config = {'type': 'step', 'step_size': 10, 'gamma': 0.1}
        scheduler = create_lr_scheduler(optimizer, config)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.StepLR)
        
        # Test MultiStepLR
        config = {'type': 'multistep', 'milestones': [10, 20, 30], 'gamma': 0.1}
        scheduler = create_lr_scheduler(optimizer, config)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.MultiStepLR)
        
        # Test CosineAnnealingLR
        config = {'type': 'cosine', 't_max': 100, 'eta_min': 0}
        scheduler = create_lr_scheduler(optimizer, config)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        
        # Test ReduceLROnPlateau
        config = {'type': 'plateau', 'mode': 'min', 'factor': 0.1, 'patience': 10}
        scheduler = create_lr_scheduler(optimizer, config)
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


class TestLoss(unittest.TestCase):
    """Test cases for loss functions."""
    
    def test_mask_rcnn_loss(self):
        """Test MaskRCNNLoss."""
        loss_fn = MaskRCNNLoss(
            rpn_cls_weight=1.0,
            rpn_bbox_weight=1.0,
            rcnn_cls_weight=1.0,
            rcnn_bbox_weight=1.0,
            mask_weight=1.0
        )
        
        # Create dummy predictions and targets
        predictions = {
            'rpn_cls_loss': torch.tensor(0.1),
            'rpn_bbox_loss': torch.tensor(0.2),
            'rcnn_cls_logits': torch.randn(10, 4),
            'rcnn_bbox_pred': torch.randn(10, 16),
            'mask_pred': torch.randn(10, 10, 28, 28)
        }
        
        targets = {
            'labels': torch.randint(0, 4, (10,)),
            'bbox_targets': torch.randn(10, 16),
            'masks': torch.randn(10, 10, 28, 28)
        }
        
        # Calculate loss
        loss_dict = loss_fn(predictions, targets)
        
        # Check loss dictionary
        self.assertIn('rpn_cls_loss', loss_dict)
        self.assertIn('rpn_bbox_loss', loss_dict)
        self.assertIn('rcnn_cls_loss', loss_dict)
        self.assertIn('rcnn_bbox_loss', loss_dict)
        self.assertIn('mask_loss', loss_dict)
        self.assertIn('total_loss', loss_dict)


class TestCallbacks(unittest.TestCase):
    """Test cases for callbacks."""
    
    def setUp(self):
        """Set up test case."""
        self.temp_dir = TemporaryDirectory()
        self.checkpoint_dir = os.path.join(self.temp_dir.name, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up after test."""
        self.temp_dir.cleanup()
    
    def test_model_checkpoint(self):
        """Test ModelCheckpoint callback."""
        model = DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        # Create callback
        filepath = os.path.join(self.checkpoint_dir, 'checkpoint_{epoch:02d}.pth')
        callback = ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=True
        )
        
        # Test on_epoch_end
        logs = {
            'val_loss': 0.5,
            'model': model,
            'optimizer': optimizer
        }
        callback.on_epoch_end(epoch=0, logs=logs)
        
        # Check if checkpoint was saved
        self.assertTrue(os.path.exists(os.path.join(self.checkpoint_dir, 'checkpoint_01.pth')))
        
        # Test saving best only
        logs['val_loss'] = 0.3  # Better loss
        callback.on_epoch_end(epoch=1, logs=logs)
        self.assertTrue(os.path.exists(os.path.join(self.checkpoint_dir, 'checkpoint_02.pth')))
        
        # Test not saving when loss is worse
        logs['val_loss'] = 0.4  # Worse loss
        callback.on_epoch_end(epoch=2, logs=logs)
        self.assertFalse(os.path.exists(os.path.join(self.checkpoint_dir, 'checkpoint_03.pth')))
    
    def test_early_stopping(self):
        """Test EarlyStopping callback."""
        callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.01,
            patience=2,
            mode='min',
            verbose=True
        )
        
        # Test improvement
        logs = {'val_loss': 0.5}
        callback.on_epoch_end(epoch=0, logs=logs)
        self.assertEqual(callback.wait, 0)
        self.assertEqual(callback.best, 0.5)
        
        # Test small improvement (less than min_delta)
        logs = {'val_loss': 0.495}  # Improvement of 0.005, less than min_delta
        callback.on_epoch_end(epoch=1, logs=logs)
        self.assertEqual(callback.wait, 1)
        self.assertEqual(callback.best, 0.5)
        
        # Test another non-improvement
        logs = {'val_loss': 0.51}  # Worse
        callback.on_epoch_end(epoch=2, logs=logs)
        self.assertEqual(callback.wait, 2)
        self.assertEqual(callback.best, 0.5)
        
        # Test early stopping triggered
        logs = {'val_loss': 0.52, 'stop_training': False}
        callback.on_epoch_end(epoch=3, logs=logs)
        self.assertTrue(callback.stopped)
        self.assertTrue(logs['stop_training'])
        
        # Test large improvement
        callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.01,
            patience=2,
            mode='min',
            verbose=True
        )
        logs = {'val_loss': 0.5}
        callback.on_epoch_end(epoch=0, logs=logs)
        
        logs = {'val_loss': 0.48}  # Improvement of 0.02, more than min_delta
        callback.on_epoch_end(epoch=1, logs=logs)
        self.assertEqual(callback.wait, 0)
        self.assertEqual(callback.best, 0.48)


if __name__ == '__main__':
    unittest.main()