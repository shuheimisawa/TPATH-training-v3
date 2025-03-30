# scripts/train_enhanced.py
import os
import sys
import json
import argparse
import torch
import traceback
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import datetime

# Add the parent directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig, OptimizerConfig, LRSchedulerConfig
from src.models.cascade_mask_rcnn import CascadeMaskRCNN
from src.data.dataset import GlomeruliDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.loader import create_dataloaders, collate_fn
from src.training.trainer import Trainer
from src.training.callbacks import ModelCheckpoint, EarlyStopping
from src.utils.logger import get_logger
from src.utils.io import load_yaml, save_json
from src.utils.directml_adapter import get_dml_device, is_available, empty_cache, get_device_info


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train enhanced Cascade Mask R-CNN for glomeruli segmentation')
    
    parser.add_argument('--config', type=str, default='experiments/configs/optimized.yaml',
                        help='Path to training configuration file')
    parser.add_argument('--data-dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Path to checkpoint directory (overrides config)')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='Path to log directory (overrides config)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--backbone', type=str, default=None,
                        help='Backbone model to use (overrides config)')
    parser.add_argument('--cpu', action='store_true',
                        help='Force using CPU even if GPU is available')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with fewer samples')
    
    return parser.parse_args()


def update_config_from_args(config, args):
    """Update config with command line arguments."""
    if args.checkpoint_dir:
        config.checkpoint_dir = args.checkpoint_dir
    if args.log_dir:
        config.log_dir = args.log_dir
    if args.seed is not None:
        config.seed = args.seed
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.debug:
        config.debug = True
        config.debug_samples = 10  # Use only 10 samples in debug mode
    
    # Update data paths if data directory is provided
    if args.data_dir:
        # Convert relative path to absolute path
        data_dir = os.path.abspath(args.data_dir)
        config.data.train_path = os.path.join(data_dir, 'train')
        config.data.val_path = os.path.join(data_dir, 'val')
        config.data.test_path = os.path.join(data_dir, 'test')
        
    return config


def update_model_config_from_args(config, args):
    """Update model config with command line arguments."""
    if args.backbone:
        # Validate backbone name
        valid_backbones = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
        if args.backbone not in valid_backbones:
            print(f"Warning: Invalid backbone name '{args.backbone}'. Must be one of {valid_backbones}")
            print(f"Using default backbone: {config.backbone.name}")
        else:
            config.backbone.name = args.backbone
    
    return config


def update_config_from_dict(config_obj, config_dict):
    """Update a config object with values from a dictionary."""
    if config_dict is None:
        return config_obj
        
    for key, value in config_dict.items():
        if hasattr(config_obj, key):
            attr = getattr(config_obj, key)
            # Check if attribute is a dataclass or a similar object
            if hasattr(attr, '__dict__') and not isinstance(attr, (str, int, float, bool, list, dict)):
                # Recursively update nested config object
                if isinstance(value, dict):
                    update_config_from_dict(attr, value)
                else:
                    setattr(config_obj, key, value)
            else:
                # Direct attribute update
                try:
                    # Handle type conversion for basic types
                    if isinstance(attr, bool) and isinstance(value, str):
                        value = value.lower() in ('true', 'yes', '1', 'y')
                    elif isinstance(attr, int) and not isinstance(value, bool):
                        value = int(value)
                    elif isinstance(attr, float) and not isinstance(value, bool):
                        value = float(value)
                    
                    setattr(config_obj, key, value)
                except Exception as e:
                    print(f"Warning: Error setting attribute '{key}' to '{value}': {e}")
    
    return config_obj


def create_model(config, device, logger, checkpoint_path=None):
    """Create model and initialize it."""
    logger.info(f"Creating model with backbone: {config.backbone.name}")
    
    try:
        # Create model
        model = CascadeMaskRCNN(config)
        
        # Move model to device
        model = model.to(device)
        
        # Load checkpoint if provided
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                logger.info(f"Loading checkpoint from {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=device)
                
                if 'model_state_dict' in checkpoint:
                    # Handle potential key mismatches
                    try:
                        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                        logger.info("Checkpoint loaded with strict matching")
                    except Exception as e:
                        logger.warning(f"Strict loading failed: {e}")
                        logger.warning("Trying non-strict loading...")
                        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                        logger.info("Checkpoint loaded with non-strict matching")
                else:
                    # Try loading as direct state dict
                    try:
                        model.load_state_dict(checkpoint, strict=True)
                        logger.info("State dictionary loaded with strict matching")
                    except Exception as e:
                        logger.warning(f"Strict loading failed: {e}")
                        logger.warning("Trying non-strict loading...")
                        model.load_state_dict(checkpoint, strict=False)
                        logger.info("State dictionary loaded with non-strict matching")
            except Exception as e:
                logger.error(f"Error loading checkpoint: {e}")
                logger.error(traceback.format_exc())
        
        return model
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        logger.error(traceback.format_exc())
        raise


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load configuration from YAML file
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        return 1
        
    try:
        config_dict = load_yaml(args.config)
    except Exception as e:
        print(f"Failed to load config: {e}")
        return 1
    
    # Create model and training configs
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Update configs from YAML
    if 'model' in config_dict:
        update_config_from_dict(model_config, config_dict['model'])
    if 'training' in config_dict:
        update_config_from_dict(training_config, config_dict['training'])
    
    # Update configs from command line arguments
    training_config = update_config_from_args(training_config, args)
    model_config = update_model_config_from_args(model_config, args)
    
    # Create directories
    os.makedirs(training_config.checkpoint_dir, exist_ok=True)
    os.makedirs(training_config.log_dir, exist_ok=True)
    
    # Create logger
    logger = get_logger(
        name="train_enhanced",
        log_file=os.path.join(training_config.log_dir, "train.log")
    )
    
    # Log versions
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    try:
        import cv2
        logger.info(f"OpenCV version: {cv2.__version__}")
    except:
        logger.info("OpenCV not available")
    
    # Set random seed
    if training_config.seed is not None:
        torch.manual_seed(training_config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(training_config.seed)
        logger.info(f"Using random seed: {training_config.seed}")
    
    # Log config information
    logger.info(f"Model configuration: {model_config}")
    logger.info(f"Training configuration: {training_config}")
    
    try:
        # Set device
        if args.cpu:
            device = torch.device("cpu")
            logger.info("Forcing CPU usage as requested")
        elif is_available() and not args.cpu:
            device = get_dml_device(args.gpu)
            logger.info(f"Using DirectML device: {get_device_info()}")
        elif torch.cuda.is_available() and not args.cpu:
            device = torch.device(f"cuda:{args.gpu}")
            logger.info(f"Using CUDA device {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        
        # Create model
        model = create_model(model_config, device, logger, args.resume)
        
        # Create data loaders
        logger.info("Creating data loaders")
        
        # Format config for data loaders
        dataloader_config = {
            'data': {
                'train_dir': training_config.data.train_path,
                'val_dir': training_config.data.val_path,
                'test_dir': training_config.data.test_path
            },
            'transforms': {
                'train': {
                    'mean': training_config.data.mean,
                    'std': training_config.data.std,
                    'use_augmentation': training_config.data.use_augmentation,
                    'augmentations': training_config.data.augmentations
                },
                'val': {
                    'mean': training_config.data.mean,
                    'std': training_config.data.std
                }
            },
            'training': {
                'batch_size': training_config.batch_size,
                'num_workers': training_config.workers
            }
        }
        
        # Create data loaders
        dataloaders = create_dataloaders(dataloader_config, distributed=training_config.distributed)
        
        train_loader = dataloaders['train']
        val_loader = dataloaders['val']
        
        # Create callbacks
        callbacks = []
        
        # Model checkpoint callback
        checkpoint_path = os.path.join(training_config.checkpoint_dir, 'checkpoint_epoch_{epoch:03d}.pth')
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
            patience=10,  # Increased patience for enhanced model
            mode='max',
            verbose=True
        )
        callbacks.append(early_stopping)
        
        # Create trainer
        logger.info("Creating trainer")
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            device=device,
            callbacks=callbacks
        )
        
        # Train model
        logger.info(f"Starting training for {training_config.epochs} epochs")
        start_time = time.time()
        training_summary = trainer.train(resume_from=args.resume)
        end_time = time.time()
        
        # Calculate total training time
        training_time = str(datetime.timedelta(seconds=int(end_time - start_time)))
        logger.info(f"Training completed in {training_time}")
        logger.info(f"Best validation mAP: {training_summary['best_val_map']:.4f}")
        
        # Save training summary
        summary_path = os.path.join(training_config.log_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2)
        logger.info(f"Saved training summary to {summary_path}")
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())