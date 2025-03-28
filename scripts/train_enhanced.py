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
from src.config.training_config import TrainingConfig
from src.models.cascade_mask_rcnn import CascadeMaskRCNN
from src.data.dataset import GlomeruliDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.data.loader import create_dataloaders
from src.training.trainer import Trainer
from src.training.callbacks import ModelCheckpoint, EarlyStopping
from src.training.enhanced_loss import EnhancedMaskRCNNLoss
from src.utils.logger import get_logger
from src.utils.io import load_yaml, save_json
from src.utils.directml_adapter import get_dml_device, is_available, empty_cache
from src.models.components.enhanced_mask_head import MaskRCNNHeadWithBoundary
from src.utils.stain_normalization import StainNormalizationTransform


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
    
    # Update data paths if data directory is provided
    if args.data_dir:
        config.data.train_path = os.path.join(args.data_dir, 'train')
        config.data.val_path = os.path.join(args.data_dir, 'val')
        config.data.test_path = os.path.join(args.data_dir, 'test')
        
    return config


def update_config_from_dict(config_obj, config_dict):
    """Update a config object with values from a dictionary."""
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
                setattr(config_obj, key, value)
    return config_obj


def create_model(config, device, logger, checkpoint_path=None):
    """Create model and initialize it."""
    logger.info("Creating model")
    
    # Create model with enhanced mask head if using boundary-aware segmentation
    model = CascadeMaskRCNN(config)
    
    # Replace default mask head with enhanced mask head if specified
    if getattr(config, 'use_enhanced_mask_head', True):
        logger.info("Using enhanced mask head with boundary awareness")
        
        # Save original mask head for reference
        original_mask_head = model.mask_head
        
        # Create enhanced mask head
        enhanced_mask_head = MaskRCNNHeadWithBoundary(
            in_channels=config.mask.in_channels,
            roi_size=config.mask.roi_size,
            num_classes=config.num_classes,
            use_attention=config.use_attention,
            attention_type=config.attention_type
        )
        
        # Replace mask head
        model.mask_head = enhanced_mask_head
    
    # Move model to device
    model = model.to(device)
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        try:
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            if 'model_state_dict' in checkpoint:
                # Handle potential key mismatches due to mask head replacement
                state_dict = checkpoint['model_state_dict']
                
                # Try to load with strict=False first
                model.load_state_dict(state_dict, strict=False)
                logger.info("Checkpoint loaded with non-strict matching")
            else:
                # Try loading as direct state dict
                model.load_state_dict(checkpoint, strict=False)
                logger.info("State dictionary loaded with non-strict matching")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            logger.error(traceback.format_exc())
    
    return model


def create_loss_function(config, device, logger):
    """Create enhanced loss function."""
    logger.info("Creating enhanced loss function")
    
    # Extract loss config
    loss_config = getattr(config, 'loss', {})
    
    # Check if enhanced loss should be used
    use_enhanced_loss = getattr(loss_config, 'use_enhanced_loss', True)
    
    if use_enhanced_loss:
        # Create class weights tensor
        class_weights = [1.0]  # Background weight
        for cls_name in config.data.classes:
            weight = getattr(config.data.class_weights, cls_name, 1.0)
            class_weights.append(weight)
        
        class_weights_tensor = torch.tensor(class_weights).to(device)
        
        # Create enhanced loss function
        loss_fn = EnhancedMaskRCNNLoss(
            rpn_cls_weight=getattr(loss_config, 'rpn_cls_loss_weight', 1.0),
            rpn_bbox_weight=getattr(loss_config, 'rpn_bbox_loss_weight', 1.0),
            rcnn_cls_weight=getattr(loss_config, 'rcnn_cls_weight', 1.2),
            rcnn_bbox_weight=getattr(loss_config, 'rcnn_bbox_weight', 1.0),
            mask_weight=getattr(loss_config, 'mask_loss_weight', 1.5),
            focal_tversky_weight=getattr(loss_config, 'focal_tversky_weight', 0.7),
            boundary_weight=getattr(loss_config, 'boundary_weight', 0.3),
            class_weights=class_weights_tensor
        )
        
        logger.info("Using enhanced loss function with focal Tversky and boundary components")
    else:
        # Fall back to standard loss
        from src.training.loss import MaskRCNNLoss
        
        # Create class weights tensor
        class_weights = [1.0]  # Background weight
        for cls_name in config.data.classes:
            weight = getattr(config.data.class_weights, cls_name, 1.0)
            class_weights.append(weight)
        
        class_weights_tensor = torch.tensor(class_weights).to(device)
        
        # Create standard loss function
        loss_fn = MaskRCNNLoss(
            rpn_cls_weight=getattr(loss_config, 'rpn_cls_loss_weight', 1.0),
            rpn_bbox_weight=getattr(loss_config, 'rpn_bbox_loss_weight', 1.0),
            rcnn_cls_weight=getattr(loss_config, 'rcnn_cls_weight', 1.0),
            rcnn_bbox_weight=getattr(loss_config, 'rcnn_bbox_weight', 1.0),
            mask_weight=getattr(loss_config, 'mask_loss_weight', 1.0),
            dice_weight=getattr(loss_config, 'dice_weight', 0.5),
            class_weights=class_weights_tensor
        )
        
        logger.info("Using standard loss function")
    
    return loss_fn


def create_transforms(config, logger):
    """Create data transforms with stain normalization."""
    logger.info("Creating data transforms")
    
    # Check if stain normalization should be used
    use_stain_norm = getattr(config.data, 'use_stain_normalization', False)
    
    if use_stain_norm:
        # Extract stain normalization config
        stain_norm_config = getattr(config.data, 'stain_normalization', {})
        
        # Create stain normalization transform
        method = getattr(stain_norm_config, 'method', 'macenko')
        target_image_path = getattr(stain_norm_config, 'target_image_path', None)
        params_path = getattr(stain_norm_config, 'params_path', None)
        
        logger.info(f"Using {method} stain normalization")
        
        # Handle target image path
        if target_image_path and not os.path.exists(target_image_path):
            logger.warning(f"Target image not found: {target_image_path}")
            target_image_path = None
        
        # Create stain normalization transform
        stain_norm_transform = StainNormalizationTransform(
            method=method,
            target_image_path=target_image_path,
            params_path=params_path
        )
        
        # Create normalization params directory if needed
        if target_image_path and params_path is None:
            norm_dir = os.path.join(config.log_dir, 'stain_normalization')
            os.makedirs(norm_dir, exist_ok=True)
            
            # Save normalization parameters for future use
            params_path = os.path.join(norm_dir, f"{method}_params.npz")
            stain_norm_transform.normalizer.save(params_path)
            logger.info(f"Saved stain normalization parameters to {params_path}")
    else:
        stain_norm_transform = None
        logger.info("Stain normalization disabled")
    
    # Get standard transforms
    train_transform = get_train_transforms(
        config=vars(config.data),
        custom_transforms=[stain_norm_transform] if stain_norm_transform else None
    )
    
    val_transform = get_val_transforms(
        config=vars(config.data),
        custom_transforms=[stain_norm_transform] if stain_norm_transform else None
    )
    
    return train_transform, val_transform


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
    
    # Update config from command line arguments
    training_config = update_config_from_args(training_config, args)
    
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
    torch.manual_seed(training_config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(training_config.seed)
    logger.info(f"Using random seed: {training_config.seed}")
    
    # Log config information
    logger.info(f"Model configuration: {model_config}")
    logger.info(f"Training configuration: {training_config}")
    
    try:
        # Set device
        if is_available():
            device = get_dml_device(args.gpu)
            logger.info("Using DirectML device")
        elif torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu}")
            logger.info(f"Using CUDA device {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU device")
        
        # Create transforms
        train_transform, val_transform = create_transforms(training_config, logger)
        
        # Create model
        model = create_model(model_config, device, logger, args.resume)
        
        # Create loss function
        loss_fn = create_loss_function(training_config, device, logger)
        
        # Create data loaders
        logger.info("Creating data loaders")
        dataloader_config = {
            'batch_size': training_config.batch_size,
            'num_workers': training_config.workers,
            'pin_memory': getattr(training_config, 'pin_memory', True)
        }
        
        train_dataset = GlomeruliDataset(
            data_dir=training_config.data.train_path,
            mode='train',
            transform=train_transform
        )
        
        val_dataset = GlomeruliDataset(
            data_dir=training_config.data.val_path,
            mode='val',
            transform=val_transform
        )
        
        logger.info(f"Created training dataset with {len(train_dataset)} samples")
        logger.info(f"Created validation dataset with {len(val_dataset)} samples")
        
        # Create data loaders
        from src.data.loader import collate_fn
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=dataloader_config['batch_size'],
            shuffle=True,
            num_workers=dataloader_config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=dataloader_config['pin_memory'],
            drop_last=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=dataloader_config['batch_size'],
            shuffle=False,
            num_workers=dataloader_config['num_workers'],
            collate_fn=collate_fn,
            pin_memory=dataloader_config['pin_memory'],
            drop_last=False
        )
        
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
        
        # Create optimizer
        from src.training.optimization import create_optimizer, create_lr_scheduler
        
        optimizer = create_optimizer(
            model.parameters(),
            vars(training_config.optimizer)
        )
        
        # Create LR scheduler
        lr_scheduler = create_lr_scheduler(
            optimizer,
            vars(training_config.lr_scheduler)
        )
        
        # Overwrite Trainer's loss function with our enhanced one
        class EnhancedTrainer(Trainer):
            def __init__(self, *args, **kwargs):
                custom_loss_fn = kwargs.pop('loss_fn', None)
                super(EnhancedTrainer, self).__init__(*args, **kwargs)
                if custom_loss_fn:
                    self.loss_fn = custom_loss_fn
                
            def _train_epoch(self, epoch):
                """Enhanced training epoch with gradient clipping."""
                self.model.train()
                total_loss = 0.0
                num_batches = len(self.train_loader)
                
                # Get gradient clipping value
                grad_clip_val = getattr(self.config, 'gradient_clip_val', None)
                
                # Create progress bar
                pbar = tqdm(self.train_loader, desc=f"Training (Epoch {epoch+1}/{self.config.epochs})")
                
                for batch_idx, batch in enumerate(pbar):
                    try:
                        # Handle both tuple and dictionary batch formats
                        if isinstance(batch, tuple) and len(batch) == 2:
                            images, targets = batch
                        elif isinstance(batch, dict):
                            images = batch['image']
                            targets = batch['target']
                        else:
                            raise ValueError(f"Unexpected batch format: {type(batch)}")
                        
                        # Move data to device
                        images = [img.to(self.device) for img in images]
                        targets = [{k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                                for k, v in t.items()} for t in targets]
                        
                        # Check for mixed precision training
                        use_amp = getattr(self.config, 'mixed_precision', False)
                        
                        if use_amp and hasattr(torch.cuda, 'amp'):
                            # Use automatic mixed precision
                            from torch.cuda.amp import autocast, GradScaler
                            scaler = GradScaler()
                            
                            # Forward pass with autocast
                            with autocast():
                                loss_dict = self.model(images, targets)
                                losses = sum(loss for loss in loss_dict.values())
                            
                            # Backward pass with scaler
                            self.optimizer.zero_grad()
                            scaler.scale(losses).backward()
                            
                            # Gradient clipping if enabled
                            if grad_clip_val:
                                scaler.unscale_(self.optimizer)
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_val)
                            
                            # Update weights with scaler
                            scaler.step(self.optimizer)
                            scaler.update()
                        else:
                            # Standard forward pass
                            loss_dict = self.model(images, targets)
                            losses = sum(loss for loss in loss_dict.values())
                            
                            # Backward pass
                            self.optimizer.zero_grad()
                            losses.backward()
                            
                            # Gradient clipping if enabled
                            if grad_clip_val:
                                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_val)
                            
                            # Update weights
                            self.optimizer.step()
                        
                        # Update progress bar
                        total_loss += losses.item()
                        pbar.set_postfix({
                            'loss': losses.item(),
                            'avg_loss': total_loss / (batch_idx + 1)
                        })
                        
                        # Free memory
                        del images, targets, loss_dict, losses
                        if batch_idx % 5 == 0:
                            empty_cache()
                        
                    except Exception as e:
                        self.logger.error(f"Error in batch {batch_idx} of epoch {epoch+1}: {e}")
                        self.logger.error(traceback.format_exc())
                        continue
                
                # Calculate average loss
                avg_loss = total_loss / num_batches
                self.logger.info(f"Epoch {epoch+1}/{self.config.epochs} - Average loss: {avg_loss:.4f}")
                
                return avg_loss
        
        # Create trainer
        trainer = EnhancedTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            device=device,
            callbacks=callbacks,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            loss_fn=loss_fn
        )
        
        # Train model
        logger.info("Starting training")
        start_time = time.time()
        training_summary = trainer.train(resume_from=args.resume)
        end_time = time.time()
        
        # Calculate total training time
        training_time = str(datetime.timedelta(seconds=int(end_time - start_time)))
        logger.info(f"Training completed in {training_time}")
        logger.info(f"Best validation mAP: {training_summary['best_val_map']:.4f}")
        
        # Save training summary
        summary_path = os.path.join(training_config.log_dir, 'training_summary.json')
        save_json(training_summary, summary_path)
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