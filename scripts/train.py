# scripts/train.py
import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader
import traceback

from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.models.cascade_mask_rcnn import CascadeMaskRCNN
from src.data import GlomeruliDataset, get_train_transforms, get_val_transforms, create_dataloaders
from src.training.trainer import Trainer
from src.training.callbacks import ModelCheckpoint, EarlyStopping
from src.utils.logger import get_logger
from src.utils.io import load_yaml
from src.utils.directml_adapter import get_dml_device, is_available


def parse_args():
    parser = argparse.ArgumentParser(description='Train Cascade Mask R-CNN for glomeruli segmentation')
    
    parser.add_argument('--config', type=str, default='experiments/configs/baseline.yaml',
                        help='Path to training configuration file')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--checkpoint-dir', type=str, default='experiments/checkpoints',
                        help='Path to checkpoint directory')
    parser.add_argument('--log-dir', type=str, default='experiments/logs',
                        help='Path to log directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def update_config_from_dict(config_obj, config_dict):
    """Update a config object with values from a dictionary, respecting nested dataclasses."""
    for key, value in config_dict.items():
        if hasattr(config_obj, key):
            attr = getattr(config_obj, key)
            # Check if attribute is a dataclass or a similar object with __dict__
            if hasattr(attr, '__dict__') and not isinstance(attr, (str, int, float, bool, list, dict)):
                # Recursively update nested config object
                if isinstance(value, dict):
                    update_config_from_dict(attr, value)
                else:
                    setattr(config_obj, key, value)
            else:
                # Direct attribute update
                setattr(config_obj, key, value)
    return config_obj


def create_model(config, device, checkpoint_path=None):
    """Factory function to create and initialize model.
    
    Args:
        config: Model configuration
        device: Device to run the model on
        checkpoint_path: Optional path to checkpoint to load
        
    Returns:
        Initialized model
    """
    model = CascadeMaskRCNN(config)
    model.to(device)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded checkpoint from {checkpoint_path}")
            else:
                # Try loading as direct state dict
                model.load_state_dict(checkpoint)
                print(f"Loaded state dictionary from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    
    return model


def train_worker(args, model_config, config, device=None):
    """Training worker function."""
    # Get logger
    logger = get_logger(name="train_worker")
    
    try:
        # Initialize device if not provided
        if device is None:
            if is_available():
                device = get_dml_device()
                logger.info("Using DirectML device for AMD GPU")
            elif torch.cuda.is_available():
                device = torch.device("cuda:0")
                logger.info("Using CUDA device")
            else:
                device = torch.device("cpu")
                logger.info("Using CPU device")
        
        # Create model
        logger.info("Creating model")
        model = create_model(model_config, device, args.resume)
        
        # Create data loaders
        logger.info("Creating dataloaders")
        # Convert config to dict format for dataloaders
        config_dict = {
            'transforms': {
                'train': {
                    'use_augmentation': config.data.use_augmentation,
                    'augmentations': config.data.augmentations,
                    'mean': config.data.mean,
                    'std': config.data.std
                },
                'val': {
                    'mean': config.data.mean,
                    'std': config.data.std
                }
            },
            'data': {
                # Make sure these keys match what loader.py is expecting
                'train_dir': config.data.train_path,
                'val_dir': config.data.val_path
            },
            'training': {
                'batch_size': config.batch_size,
                'num_workers': config.workers
            }
        }
        
        # Print the config_dict for debugging
        logger.info(f"Data directory config: {config_dict['data']}")
        
        dataloaders = create_dataloaders(config_dict, distributed=False)
        
        # Create callbacks
        callbacks = []
        
        # Model checkpoint callback
        checkpoint_path = os.path.join(config.checkpoint_dir, 'checkpoint_epoch_{epoch:02d}.pth')
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
        logger.info("Creating trainer")
        trainer = Trainer(
            model=model,
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            config=config,
            device=device,
            callbacks=callbacks
        )
        
        # Train model
        logger.info("Starting training")
        training_summary = trainer.train(resume_from=args.resume)
        logger.info("Training completed")
        
        # Log training summary
        logger.info(f"Best validation mAP: {training_summary['best_val_map']:.4f}")
        logger.info(f"Total training time: {training_summary['total_training_time']}")
    
    except Exception as e:
        logger.error(f"Training error: {e}")
        logger.error(traceback.format_exc())
        raise


def main():
    try:
        # Parse arguments
        args = parse_args()
        
        # Create logger
        os.makedirs(args.log_dir, exist_ok=True)
        logger = get_logger(
            name="train",
            log_file=os.path.join(args.log_dir, "train.log")
        )
        
        # Set random seed
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        
        # Load configuration from YAML file
        logger.info(f"Loading configuration from {args.config}")
        if not os.path.exists(args.config):
            logger.error(f"Config file not found: {args.config}")
            return 1
            
        try:
            config_dict = load_yaml(args.config)
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return 1
        
        # Create model and training configs
        model_config = ModelConfig()
        config = TrainingConfig()
        
        # Update configs from YAML
        if 'model' in config_dict:
            update_config_from_dict(model_config, config_dict['model'])
        if 'training' in config_dict:
            update_config_from_dict(config, config_dict['training'])
        
        # Update paths from arguments
        if args.data_dir:
            # Use the data directory as is, since it already contains 'processed'
            config.data.train_path = os.path.join(args.data_dir, 'train')
            config.data.val_path = os.path.join(args.data_dir, 'val')
            config.data.test_path = os.path.join(args.data_dir, 'test')
        
        config.checkpoint_dir = args.checkpoint_dir
        config.log_dir = args.log_dir
        
        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        
        # Start training
        train_worker(args, model_config, config)
        
        return 0
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())