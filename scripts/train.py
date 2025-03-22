# scripts/train.py
import os
import json
import argparse
import torch
from torch.utils.data import DataLoader

from src.config.model_config import ModelConfig
from src.config.training_config import TrainingConfig
from src.models.cascade_mask_rcnn import CascadeMaskRCNN
from src.data.dataset import GlomeruliDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.loader import create_dataloaders
from src.training.trainer import Trainer
from src.training.callbacks import ModelCheckpoint, EarlyStopping
from src.utils.logger import get_logger
from src.utils.io import load_yaml
from src.utils.distributed import setup_distributed, cleanup_distributed, run_distributed


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
    parser.add_argument('--distributed', action='store_true',
                        help='Use distributed training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
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
    config_dict = load_yaml(args.config)
    
    # Create configuration objects and populate from YAML
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Update model config with values from YAML
    for section, values in config_dict.get('model', {}).items():
        if hasattr(model_config, section):
            if isinstance(getattr(model_config, section), dict):
                getattr(model_config, section).update(values)
            else:
                setattr(model_config, section, values)
    
    # Update training config with values from YAML
    for section, values in config_dict.get('training', {}).items():
        if hasattr(training_config, section):
            if isinstance(getattr(training_config, section), dict):
                getattr(training_config, section).update(values)
            else:
                setattr(training_config, section, values)
    
    # Update config with command line arguments
    training_config.checkpoint_dir = args.checkpoint_dir
    training_config.log_dir = args.log_dir
    training_config.seed = args.seed
    training_config.distributed = args.distributed
    
    # Create directories
    os.makedirs(training_config.checkpoint_dir, exist_ok=True)
    os.makedirs(training_config.log_dir, exist_ok=True)
    
    # Handle distributed training if requested
    if args.distributed:
        logger.info("Starting distributed training")
        world_size = len(training_config.gpu_ids)
        run_distributed(train_worker, world_size, args, model_config, training_config)
    else:
        # Set device
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Call training function directly
        train_worker(0, args, model_config, training_config, device=device)


def train_worker(rank, args, model_config, config, device=None):
    """Training worker function for distributed training."""
    # Setup distributed training if needed
    if args.distributed:
        setup_distributed(rank, len(config.gpu_ids))
        device = torch.device(f'cuda:{rank}')
    
    # Get logger
    logger = get_logger(name=f"train_worker_{rank}")
    
    # Create model
    logger.info("Creating model")
    model = CascadeMaskRCNN(model_config)
    model.to(device)
    
    # Load checkpoint if resuming
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from {args.resume}")
        
        # Ensure parameters are synchronized before DDP wrapping
        if args.distributed:
            # Broadcast model parameters from rank 0
            for param in model.parameters():
                torch.distributed.broadcast(param.data, 0)
    
    # Wrap model with DDP if distributed
    if args.distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[rank])
    
    # Create data loaders
    logger.info("Creating dataloaders")
    dataloaders = create_dataloaders(config, distributed=args.distributed)
    
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
    if rank == 0:
        logger.info(f"Best validation mAP: {training_summary['best_val_map']:.4f}")
        logger.info(f"Total training time: {training_summary['total_training_time']}")
    
    # Clean up distributed training
    if args.distributed:
        cleanup_distributed()


if __name__ == '__main__':
    main()