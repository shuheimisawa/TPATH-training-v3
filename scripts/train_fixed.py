# scripts/train_fixed.py
import os
import sys
import json
import argparse
import torch
import traceback
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add the parent directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.models.cascade_mask_rcnn import CascadeMaskRCNN
from src.data.dataset import GlomeruliDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.loader import collate_fn
from src.training.trainer import Trainer
from src.training.callbacks import ModelCheckpoint, EarlyStopping
from src.utils.logger import get_logger
from src.utils.directml_adapter import get_dml_device, is_available, empty_cache

def parse_args():
    parser = argparse.ArgumentParser(description='Train Cascade Mask R-CNN with fixes')
    
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default='training_output',
                        help='Path to output directory')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='Backbone model to use (resnet18, resnet34, resnet50, resnet101, resnet152)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage even if GPU is available')
    
    return parser.parse_args()

def create_safe_dataset(data_dir, mode, transform=None):
    """Create dataset with error handling."""
    try:
        dataset = GlomeruliDataset(data_dir=data_dir, mode=mode, transform=transform)
        print(f"Created {mode} dataset with {len(dataset)} samples")
        return dataset
    except Exception as e:
        print(f"Error creating {mode} dataset: {e}")
        print(traceback.format_exc())
        # Create minimal dataset as fallback
        return GlomeruliDataset(data_dir=data_dir, mode=mode, transform=transform)

def main():
    # Parse arguments
    args = parse_args()
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    log_dir = os.path.join(args.output_dir, 'logs')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = get_logger(
        name="train_fixed",
        log_file=os.path.join(log_dir, "train.log")
    )
    
    # Validate backbone name
    valid_backbones = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    if args.backbone not in valid_backbones:
        logger.warning(f"Invalid backbone name '{args.backbone}'. Must be one of {valid_backbones}")
        logger.warning(f"Using default backbone: resnet50")
        args.backbone = 'resnet50'
    
    # Set device
    if args.cpu:
        device = torch.device("cpu")
        logger.info("Forcing CPU usage")
    elif is_available() and not args.cpu:
        device = get_dml_device()
        logger.info("Using DirectML device")
    elif torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda:0")
        logger.info("Using CUDA device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    try:
        # Create model config (simplified)
        model_config = ModelConfig()
        model_config.backbone.name = args.backbone
        model_config.backbone.freeze_stages = 1
        model_config.fpn.num_outs = 4
        model_config.cascade.num_stages = 2  # Reduced for stability
        model_config.use_bifpn = False
        model_config.use_attention = False
        
        # Create training config
        train_config = TrainingConfig()
        train_config.epochs = args.epochs
        train_config.batch_size = args.batch_size
        train_config.checkpoint_dir = checkpoint_dir
        train_config.log_dir = log_dir
        
        # Data paths
        train_dir = os.path.join(args.data_dir, 'train')
        val_dir = os.path.join(args.data_dir, 'val')
        
        # Check if data directories exist
        if not os.path.exists(train_dir):
            logger.warning(f"Training directory not found: {train_dir}")
            # Try alternative paths
            alternatives = [
                'data/train',
                'processed/train',
                'data_test/processed/train',
                '../data/train'
            ]
            for path in alternatives:
                if os.path.exists(path):
                    train_dir = path
                    logger.info(f"Found alternative training path: {train_dir}")
                    break
        
        if not os.path.exists(val_dir):
            logger.warning(f"Validation directory not found: {val_dir}")
            # Try alternative paths
            alternatives = [
                'data/val',
                'processed/val',
                'data_test/processed/val',
                '../data/val'
            ]
            for path in alternatives:
                if os.path.exists(path):
                    val_dir = path
                    logger.info(f"Found alternative validation path: {val_dir}")
                    break
        
        logger.info(f"Using data directories: train={train_dir}, val={val_dir}")
        
        # Create transforms
        train_transform = get_train_transforms({
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'use_augmentation': True
        })
        
        val_transform = get_val_transforms({
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        })
        
        # Create datasets safely
        train_dataset = create_safe_dataset(train_dir, 'train', train_transform)
        val_dataset = create_safe_dataset(val_dir, 'val', val_transform)
        
        if len(train_dataset) == 0:
            logger.error("Training dataset is empty. Cannot proceed.")
            return 1
            
        if len(val_dataset) == 0:
            logger.warning("Validation dataset is empty. Using a subset of training data for validation.")
            # Use a subset of training data for validation
            if len(train_dataset) > 1:
                train_size = int(0.8 * len(train_dataset))
                val_size = len(train_dataset) - train_size
                train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False
        )
        
        # Create model
        logger.info(f"Creating model with backbone: {model_config.backbone.name}")
        model = CascadeMaskRCNN(model_config)
        model.to(device)
        
        # Create callbacks
        callbacks = []
        
        # Model checkpoint callback
        checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_epoch_{epoch:02d}.pth')
        checkpoint_callback = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_map',
            save_best_only=True,
            mode='max',
            verbose=True
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_map',
            min_delta=0.001,
            patience=5,
            mode='max',
            verbose=True
        )
        callbacks.append(early_stopping)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=train_config,
            device=device,
            callbacks=callbacks
        )
        
        # Start training
        logger.info(f"Starting training for {args.epochs} epochs")
        training_summary = trainer.train(resume_from=args.resume)
        
        logger.info(f"Training completed. Best validation mAP: {training_summary['best_val_map']:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == '__main__':
    sys.exit(main())