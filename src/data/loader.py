# src/data/loader.py
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Dict, List, Tuple, Union, Optional, Any
import traceback

from .dataset import GlomeruliDataset
from .transforms import get_train_transforms, get_val_transforms


def collate_fn(batch):
    """Custom collate function for batching data."""
    images = []
    targets = []
    image_ids = []
    image_paths = []
    
    for item in batch:
        images.append(item['image'])
        targets.append(item['target'])
        image_ids.append(item['image_id'])
        image_paths.append(item['image_path'])
    
    # Stack images
    images = torch.stack(images, 0)
    
    return {
        'images': images,
        'targets': targets,
        'image_ids': image_ids,
        'image_paths': image_paths
    }


def create_dataloaders(config):
    """Create training, validation and test dataloaders.
    
    Args:
        config: Configuration dictionary containing data paths and training parameters
        
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        test_loader: Test dataloader (optional)
    """
    # Get data paths from config
    train_dir = config.data.train_path
    val_dir = config.data.val_path
    test_dir = config.data.test_path
    
    if not train_dir or not val_dir:
        raise ValueError("Training and validation paths must be provided in config")
    
    # Get training parameters
    batch_size = config.training.batch_size
    num_workers = config.training.num_workers
    
    # Create datasets
    train_dataset = GlomeruliDataset(
        data_dir=train_dir,
        mode='train',
        transform=get_train_transforms()
    )
    
    val_dataset = GlomeruliDataset(
        data_dir=val_dir,
        mode='val',
        transform=get_val_transforms()
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create test loader if test directory is provided
    test_loader = None
    if test_dir:
        test_dataset = GlomeruliDataset(
            data_dir=test_dir,
            mode='test',
            transform=get_val_transforms()
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader