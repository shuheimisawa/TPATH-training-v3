import os
import sys
import unittest
import torch
import numpy as np
import albumentations as A
from PIL import Image
from tempfile import TemporaryDirectory
import json
import shutil

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import GlomeruliDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.loader import create_dataloaders, collate_fn


class TestGlomeruliDataset(unittest.TestCase):
    """Test cases for the GlomeruliDataset class."""
    
    def setUp(self):
        """Set up test case."""
        # Create a temporary directory for test data
        self.temp_dir = TemporaryDirectory()
        self.data_dir = self.temp_dir.name
        
        # Create test data structure
        os.makedirs(os.path.join(self.data_dir), exist_ok=True)
        
        # Create a dummy image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        self.image_path = os.path.join(self.data_dir, 'test_image.jpg')
        Image.fromarray(img).save(self.image_path)
        
        # Create dummy annotations
        self.annotations = {
            'test_image': {
                'file_path': 'test_image.jpg',
                'stain_type': 'PASM',
                'annotations': [
                    {
                        'bbox': [10, 10, 50, 50],  # x, y, width, height
                        'category': 'GN',
                        'segmentation': [
                            [10, 10, 60, 10, 60, 60, 10, 60]  # Polygon points
                        ]
                    }
                ]
            }
        }
        
        # Save annotations
        with open(os.path.join(self.data_dir, 'train_annotations.json'), 'w') as f:
            json.dump(self.annotations, f)
    
    def tearDown(self):
        """Clean up after test."""
        self.temp_dir.cleanup()
    
    def test_dataset_initialization(self):
        """Test dataset initialization."""
        dataset = GlomeruliDataset(
            data_dir=self.data_dir,
            mode='train'
        )
        
        self.assertEqual(len(dataset), 1)
        self.assertEqual(dataset.CLASSES, ['background', 'Normal', 'Partially_sclerotic', 'Sclerotic', 'Uncertain'])
    
    def test_dataset_getitem(self):
        """Test dataset __getitem__ method."""
        # Create transforms
        transforms = A.Compose(
            [
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                A.pytorch.ToTensorV2()
            ],
            bbox_params=A.BboxParams(
                format='pascal_voc',
                label_fields=['labels']
            )
        )
        
        dataset = GlomeruliDataset(
            data_dir=self.data_dir,
            transform=transforms,
            mode='train'
        )
        
        sample = dataset[0]
        
        # Check keys
        self.assertIn('image', sample)
        self.assertIn('target', sample)
        self.assertIn('image_id', sample)
        self.assertIn('stain_type', sample)
        
        # Check target
        target = sample['target']
        self.assertIn('boxes', target)
        self.assertIn('labels', target)
        self.assertIn('masks', target)
        
        # Check types
        self.assertIsInstance(sample['image'], torch.Tensor)
        self.assertIsInstance(target['boxes'], torch.Tensor)
        self.assertIsInstance(target['labels'], torch.Tensor)
        self.assertIsInstance(target['masks'], torch.Tensor)
        
        # Check shapes
        self.assertEqual(sample['image'].shape[0], 3)  # 3 channels
        self.assertEqual(len(target['boxes'].shape), 2)
        self.assertEqual(target['boxes'].shape[1], 4)  # x1, y1, x2, y2
    
    def test_transforms(self):
        """Test data transforms."""
        config = {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'use_augmentation': True,
            'augmentations': {
                'horizontal_flip': {'p': 0.5},
                'vertical_flip': {'p': 0.5}
            }
        }
        
        train_transforms = get_train_transforms(config)
        val_transforms = get_val_transforms(config)
        
        self.assertIsInstance(train_transforms, A.Compose)
        self.assertIsInstance(val_transforms, A.Compose)
    
    def test_dataloader(self):
        """Test data loader creation."""
        config = {
            'data': {
                'train_path': self.data_dir,
                'val_path': self.data_dir,
                'test_path': self.data_dir,
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            },
            'batch_size': 2,
            'workers': 1
        }
        
        dataloaders = create_dataloaders(config)
        
        self.assertIn('train', dataloaders)
        self.assertIn('val', dataloaders)
        self.assertIn('test', dataloaders)