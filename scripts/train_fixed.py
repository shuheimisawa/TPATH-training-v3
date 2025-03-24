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
        # Return empty dataset as fallback
        return torch.utils.data.TensorDataset(torch.tensor([]))

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
    
    # Set device
    if is_available():
        device = get_dml_device()
        logger.info("Using DirectML device")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info("Using CUDA device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    try:
        # Create model config (simplified)
        model_config = ModelConfig()
        model_config.backbone.name = 'resnet50'
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
        logger.info("Creating model...")
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
        
        # Create trainer with a max iterations safeguard
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=train_config,
            device=device,
            callbacks=callbacks
        )
        
        # Monkey patch the _train_epoch method to add safety checks
        import types
        trainer._original_train_epoch = trainer._train_epoch
        
        def safe_train_epoch(self, epoch):
            self.model.train()
            
            total_loss = 0.0
            processed_batches = 0
            max_batches = len(self.train_loader)  # Safety check
            
            epoch_iterator = tqdm(self.train_loader, desc=f"Training (Epoch {epoch+1})")
            
            for i, batch in enumerate(epoch_iterator):
                # Safety check to prevent infinite loops
                if processed_batches >= max_batches:
                    self.logger.warning(f"Reached maximum number of batches ({max_batches}). Breaking loop.")
                    break
                    
                try:
                    # Process batch (simplified)
                    images, targets = batch
                    
                    # Check for empty batch
                    if not images or not targets:
                        self.logger.warning(f"Empty batch at index {i}. Skipping.")
                        continue
                    
                    # Move data to device
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    
                    # Zero gradients
                    self.optimizer.zero_grad()
                    
                    # Forward pass
                    losses = self.model(images, targets)
                    
                    # Calculate total loss
                    if isinstance(losses, dict):
                        loss = sum(loss for loss in losses.values() if isinstance(loss, torch.Tensor))
                    else:
                        loss = losses
                    
                    # Backward pass
                    loss.backward()
                    
                    # Check for NaN loss
                    if torch.isnan(loss) or torch.isinf(loss):
                        self.logger.error(f"NaN or Inf loss detected. Skipping batch.")
                        continue
                    
                    # Update weights
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    processed_batches += 1
                    
                    # Update display
                    epoch_iterator.set_postfix({
                        'loss': loss.item(),
                        'processed': f"{processed_batches}/{max_batches}"
                    })
                    
                    # Clear memory every few batches
                    if i % 5 == 0:
                        empty_cache()
                    
                except Exception as e:
                    self.logger.error(f"Error in batch {i}: {e}")
                    self.logger.error(traceback.format_exc())
                    continue
            
            avg_loss = total_loss / max(1, processed_batches)
            self.logger.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}, Processed {processed_batches}/{max_batches} batches")
            return avg_loss
        
        trainer._train_epoch = types.MethodType(safe_train_epoch, trainer)
        
        # Start training
        logger.info("Starting training...")
        training_summary = trainer.train(resume_from=args.resume)
        
        logger.info(f"Training completed. Best validation mAP: {training_summary['best_val_map']:.4f}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
        return 1

if __name__ == '__main__':
    sys.exit(main())