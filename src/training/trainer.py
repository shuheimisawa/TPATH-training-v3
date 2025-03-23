# src/training/trainer.py
import os
import time
import datetime
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from ..utils.logger import get_logger
from ..utils.distributed import is_main_process
from ..config.training_config import TrainingConfig
from ..evaluation.evaluator import Evaluator
from .optimization import create_optimizer, create_lr_scheduler
from .loss import MaskRCNNLoss
# Import the DirectML adapter
from ..utils.directml_adapter import empty_cache, is_available


class Trainer:
    """Trainer for Cascade Mask R-CNN model."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: torch.device,
        callbacks: Optional[List] = None
    ):
        """Initialize the trainer.
        
        Args:
            model: The model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: Training configuration
            device: Device to use for training
            callbacks: Optional list of callbacks
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.callbacks = callbacks or []
        
        # Check if using DirectML and warn about distributed training
        if config.distributed and "directml" in str(device):
            print("Warning: Distributed training not fully supported with DirectML. Using single GPU mode.")
            config.distributed = False
        
        # Create optimizer
        self.optimizer = create_optimizer(
            self.model.parameters(),
            vars(self.config.optimizer)
        )
        
        # Create learning rate scheduler
        self.lr_scheduler = create_lr_scheduler(
            self.optimizer,
            vars(self.config.lr_scheduler)
        )
        
        # Create loss function with class weights
        self.loss_fn = self._init_loss_function(config)
        
        # Create evaluator
        self.evaluator = Evaluator(device=device)
        
        # Set up logger
        self.logger = get_logger(name="trainer")
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Initialize training state
        self.start_epoch = 0
        self.best_val_map = 0.0
        self.global_step = 0
        
        # Summary metrics
        self.train_loss_history = []
        self.val_map_history = []
        self.learning_rates = []
    
    def _init_loss_function(self, config):
        """Initialize the loss function with class weights.
        
        Args:
            config: Training configuration
            
        Returns:
            Loss function instance
        """
        # Extract class weights from config
        if hasattr(config, 'class_weights'):
            # Create class weights tensor
            class_weights = [
                config.class_weights.background,
                config.class_weights.Normal,
                config.class_weights.Sclerotic,
                config.class_weights.Partially_sclerotic,
                config.class_weights.Uncertain
            ]
            class_weights_tensor = torch.tensor(class_weights).to(self.device)
        else:
            class_weights_tensor = None
        
        # Create loss function with class weights
        loss_fn = MaskRCNNLoss(
            rpn_cls_weight=config.rpn_cls_loss_weight,
            rpn_bbox_weight=config.rpn_bbox_loss_weight,
            rcnn_cls_weight=config.rcnn_cls_loss_weight,
            rcnn_bbox_weight=config.rcnn_bbox_loss_weight,
            mask_weight=config.mask_loss_weight,
            dice_weight=getattr(config, 'dice_weight', 0.5),
            class_weights=class_weights_tensor
        )
        
        return loss_fn
    
    def train(self, resume_from: Optional[str] = None) -> Dict:
        """Train the model.
        
        Args:
            resume_from: Path to checkpoint to resume from
            
        Returns:
            Dictionary with training summary
        """
        if resume_from:
            self._load_checkpoint(resume_from)
        
        self.logger.info("Starting training")
        self.logger.info(f"Training for {self.config.epochs} epochs")
        
        # Start time for total training time
        start_time = time.time()
        
        # Call on_train_begin for callbacks
        logs = {}
        for callback in self.callbacks:
            callback.on_train_begin(logs)
        
        try:
            for epoch in range(self.start_epoch, self.config.epochs):
                # Call on_epoch_begin for callbacks
                epoch_logs = {}
                for callback in self.callbacks:
                    callback.on_epoch_begin(epoch, epoch_logs)
                
                self.logger.info(f"Epoch {epoch+1}/{self.config.epochs}")
                
                # Train for one epoch
                try:
                    epoch_loss = self._train_epoch(epoch)
                except Exception as e:
                    self.logger.error(f"Error in training epoch {epoch+1}: {e}")
                    # Save emergency checkpoint
                    self._save_checkpoint(epoch, is_emergency=True)
                    raise
                
                # Update learning rate
                if self.lr_scheduler is not None:
                    # Store current learning rate
                    self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
                    
                    if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        # This scheduler needs the validation metric
                        pass  # will be updated after validation
                    else:
                        self.lr_scheduler.step()
                
                # Record train loss
                self.train_loss_history.append(epoch_loss)
                epoch_logs['train_loss'] = epoch_loss
                
                # Evaluate on validation set
                if (epoch + 1) % self.config.eval_freq == 0:
                    try:
                        val_metrics = self._validate_epoch(epoch)
                        val_map = val_metrics['mAP']
                        self.val_map_history.append(val_map)
                        epoch_logs.update(val_metrics)
                        epoch_logs['val_map'] = val_map
                        
                        # Update LR scheduler if it's ReduceLROnPlateau
                        if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step(val_map)
                    except Exception as e:
                        self.logger.error(f"Error in validation epoch {epoch+1}: {e}")
                        # Use previous metrics or default values
                        val_map = self.val_map_history[-1] if self.val_map_history else 0.0
                        self.val_map_history.append(val_map)
                        epoch_logs['val_map'] = val_map
                
                # Add model and optimizer to logs for callbacks
                epoch_logs['model'] = self.model
                epoch_logs['optimizer'] = self.optimizer
                epoch_logs['lr_scheduler'] = self.lr_scheduler
                
                # Call on_epoch_end for callbacks
                for callback in self.callbacks:
                    callback.on_epoch_end(epoch, epoch_logs)
                
                # Check if training should be stopped (from callbacks)
                if epoch_logs.get('stop_training', False):
                    self.logger.info("Early stopping triggered")
                    break
                
                # Save periodic checkpoint (if not saved by callback)
                if (epoch + 1) % self.config.save_freq == 0:
                    self._save_checkpoint(epoch)
                
                # Clear GPU cache after each epoch to avoid memory issues
                empty_cache()
            
            # Training completed, save final checkpoint
            self._save_checkpoint(self.config.epochs - 1, is_final=True)
        
        except Exception as e:
            self.logger.error(f"Training interrupted: {e}")
            # Save emergency checkpoint
            self._save_checkpoint(self.start_epoch + len(self.train_loss_history), is_emergency=True)
            raise
        
        finally:
            # Calculate total training time
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            
            # Call on_train_end for callbacks
            end_logs = {'total_time': total_time_str}
            for callback in self.callbacks:
                callback.on_train_end(end_logs)
            
            self.logger.info(f"Training completed in {total_time_str}")
            self.logger.info(f"Best validation mAP: {self.best_val_map:.4f}")
            
            # Create and return training summary
            training_summary = {
                'best_val_map': self.best_val_map,
                'train_loss_history': self.train_loss_history,
                'val_map_history': self.val_map_history,
                'learning_rates': self.learning_rates,
                'total_training_time': total_time_str
            }
            
            # Save training summary
            summary_path = os.path.join(self.config.log_dir, 'training_summary.json')
            try:
                with open(summary_path, 'w') as f:
                    json.dump(training_summary, f, indent=2)
            except Exception as e:
                self.logger.error(f"Failed to save training summary: {e}")
            
            return training_summary
    
    def _train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        epoch_iterator = tqdm(self.train_loader, desc=f"Training (Epoch {epoch+1})")
        
        # Default accumulation steps (1 = no accumulation)
        accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 1)
        
        for i, batch in enumerate(epoch_iterator):
            try:
                # Call on_batch_begin for callbacks
                batch_logs = {'batch': i, 'size': len(batch[0])}
                for callback in self.callbacks:
                    callback.on_batch_begin(i, batch_logs)
                
                # Get images and targets
                images, targets = batch
                
                # Move data to device
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                
                # Forward pass
                losses = self.model(images, targets)
                
                # Calculate total loss
                loss = sum(loss for loss in losses.values())
                
                # Scale the loss for gradient accumulation
                loss = loss / accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Check for NaN loss
                if torch.isnan(loss) or torch.isinf(loss):
                    self.logger.error(f"NaN or Inf loss detected: {loss.item()}")
                    self.logger.error(f"Loss breakdown: {losses}")
                    raise RuntimeError("NaN or Inf loss detected")
                
                # Only update weights after accumulation_steps
                if (i + 1) % accumulation_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                
                # For logging, use the unscaled loss
                batch_loss = loss.item() * accumulation_steps
                total_loss += batch_loss
                
                # Update metrics display
                current_lr = self.optimizer.param_groups[0]['lr']
                epoch_iterator.set_postfix({
                    'loss': batch_loss,
                    'lr': current_lr
                })
                
                self.global_step += 1
                
                # Update batch_logs for callbacks
                batch_logs.update({
                    'loss': batch_loss,
                    'lr': current_lr
                })
                
                # Call on_batch_end for callbacks
                for callback in self.callbacks:
                    callback.on_batch_end(i, batch_logs)
                
                # Log batch metrics
                if i % self.config.log_freq == 0:
                    self.logger.info(f"Epoch: {epoch+1}, Batch: {i}, Loss: {batch_loss:.4f}, LR: {current_lr:.6f}")
                
                # Clear GPU memory every few batches to prevent OOM
                if i % 10 == 0:
                    empty_cache()
            
            except Exception as e:
                self.logger.error(f"Error in batch {i} of epoch {epoch+1}: {e}")
                # Skip this batch and continue with next
                continue
        
        # Calculate average loss
        avg_loss = total_loss / max(1, len(self.train_loader))
        self.logger.info(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def _validate_epoch(self, epoch: int) -> Dict:
        """Validate the model on the validation set."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc=f"Validation (Epoch {epoch+1})"):
                try:
                    # Get images and targets
                    images, targets = batch
                    
                    # Move data to device
                    images = [img.to(self.device) for img in images]
                    targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                    
                    # Get predictions
                    predictions = self.model(images)
                    
                    # Convert targets to CPU for consistent evaluation
                    cpu_targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
                    cpu_predictions = []
                    for pred in predictions:
                        cpu_pred = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                                for k, v in pred.items()}
                        cpu_predictions.append(cpu_pred)
                    
                    # Store predictions and targets
                    all_predictions.extend(cpu_predictions)
                    all_targets.extend(cpu_targets)
                    
                    # Clear GPU memory to avoid OOM
                    empty_cache()
                except Exception as e:
                    self.logger.error(f"Error in validation batch: {e}")
                    # Skip this batch and continue with next
                    continue
        
        # Skip evaluation if no predictions gathered
        if not all_predictions:
            self.logger.warning("No predictions gathered during validation")
            return {'mAP': 0.0}
        
        # Evaluate predictions
        try:
            metrics = self.evaluator.evaluate(all_predictions, all_targets)
        except Exception as e:
            self.logger.error(f"Error evaluating predictions: {e}")
            # Return default metrics
            return {'mAP': 0.0}
        
        # Check if this is the best model
        if metrics['mAP'] > self.best_val_map:
            self.best_val_map = metrics['mAP']
            self._save_checkpoint(epoch, is_best=True)
        
        self.logger.info(f"Epoch {epoch+1} - Validation mAP: {metrics['mAP']:.4f}")
        
        # Clear GPU memory after validation
        empty_cache()
        
        return metrics
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False, 
                        is_final: bool = False, is_emergency: bool = False) -> None:
        """Save a checkpoint of the model."""
        # Only save checkpoints from main process in distributed training
        if not is_main_process():
            return
        
        try:
            # Ensure checkpoint directory exists with proper path handling
            os.makedirs(self.config.checkpoint_dir, exist_ok=True)
            
            # Get model state dict
            if hasattr(self.model, 'module'):
                # If using DDP, get underlying model
                model_state_dict = self.model.module.state_dict()
            else:
                model_state_dict = self.model.state_dict()
            
            # Move state dict to CPU before saving
            model_state_dict = {k: v.cpu() for k, v in model_state_dict.items()}
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model_state_dict,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                'best_val_map': self.best_val_map,
                'global_step': self.global_step,
                'train_loss_history': self.train_loss_history,
                'val_map_history': self.val_map_history
            }
            
            # Save regular checkpoint
            checkpoint_path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Checkpoint saved to {checkpoint_path}")
            
            # Save best model
            if is_best:
                best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
                torch.save(checkpoint, best_path)
                self.logger.info(f"Best model saved to {best_path}")
            
            # Save final model
            if is_final:
                final_path = os.path.join(self.config.checkpoint_dir, 'final_model.pth')
                torch.save(checkpoint, final_path)
                self.logger.info(f"Final model saved to {final_path}")
            
            # Save emergency checkpoint
            if is_emergency:
                emergency_path = os.path.join(self.config.checkpoint_dir, f'emergency_epoch_{epoch+1}.pth')
                torch.save(checkpoint, emergency_path)
                self.logger.info(f"Emergency checkpoint saved to {emergency_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
            # Try to save a minimal checkpoint with just the model state
            try:
                minimal_path = os.path.join(self.config.checkpoint_dir, f'minimal_epoch_{epoch+1}.pth')
                if hasattr(self.model, 'module'):
                    torch.save(self.model.module.state_dict(), minimal_path)
                else:
                    torch.save(self.model.state_dict(), minimal_path)
                self.logger.info(f"Minimal checkpoint saved to {minimal_path}")
            except Exception as e2:
                self.logger.error(f"Failed to save minimal checkpoint: {e2}")
    
    def _load_checkpoint(self, checkpoint_path: str) -> None:
        """Load a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint
        """
        if not os.path.exists(checkpoint_path):
            self.logger.error(f"Checkpoint {checkpoint_path} does not exist")
            return
        
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model
            if hasattr(self.model, 'module'):
                # If using DDP, load to module
                self.model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
            # Load optimizer
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load LR scheduler if available
            if checkpoint['lr_scheduler_state_dict'] and self.lr_scheduler:
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            
            # Load training state
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_val_map = checkpoint['best_val_map']
            self.global_step = checkpoint['global_step']
            self.train_loss_history = checkpoint['train_loss_history']
            self.val_map_history = checkpoint.get('val_map_history', [])
            
            self.logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {e}")
            raise