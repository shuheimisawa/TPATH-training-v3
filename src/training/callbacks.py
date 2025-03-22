# src/training/callbacks.py
import os
import torch
from typing import Dict, Any, Callable, List, Optional


class Callback:
    """Base class for callbacks."""
    
    def on_train_begin(self, logs: Optional[Dict] = None) -> None:
        """Called at the beginning of training."""
        pass
    
    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_begin(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Called at the beginning of an epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Called at the end of an epoch."""
        pass
    
    def on_batch_begin(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Called at the beginning of a batch."""
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict] = None) -> None:
        """Called at the end of a batch."""
        pass


class ModelCheckpoint(Callback):
    """Callback to save model checkpoints."""
    
    def __init__(self, 
                 filepath: str, 
                 monitor: str = 'val_map',
                 save_best_only: bool = False,
                 save_weights_only: bool = False,
                 mode: str = 'max',
                 period: int = 1,
                 verbose: bool = False):
        """Initialize ModelCheckpoint callback.
        
        Args:
            filepath: Path to save the model file
            monitor: Quantity to monitor
            save_best_only: Whether to save only the best model
            save_weights_only: Whether to save only weights (not optimizer)
            mode: 'min' or 'max'
            period: Interval (number of epochs) between checkpoints
            verbose: Whether to print messages
        """
        super(ModelCheckpoint, self).__init__()
        
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.mode = mode
        self.period = period
        self.verbose = verbose
        
        # Initialize best metric value
        self.best = float('inf') if mode == 'min' else float('-inf')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Keep track of epochs since last save
        self.epochs_since_last_save = 0
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Save the model at the end of an epoch."""
        logs = logs or {}
        self.epochs_since_last_save += 1
        
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            
            if self.save_best_only:
                # Get current metric value
                current = logs.get(self.monitor)
                
                if current is None:
                    if self.verbose:
                        print(f"Can't save best model: {self.monitor} not in logs")
                    return
                
                # Check if current model is better
                if (self.mode == 'min' and current < self.best) or \
                   (self.mode == 'max' and current > self.best):
                    if self.verbose:
                        print(f"Epoch {epoch+1}: {self.monitor} improved from {self.best:.4f} to {current:.4f}")
                    
                    self.best = current
                    self._save_model(epoch, logs)
                else:
                    if self.verbose:
                        print(f"Epoch {epoch+1}: {self.monitor} did not improve from {self.best:.4f}")
            else:
                # Always save
                self._save_model(epoch, logs)
    
    def _save_model(self, epoch: int, logs: Dict) -> None:
        """Save the model."""
        filepath = self.filepath.format(epoch=epoch + 1, **logs)
        
        if self.verbose:
            print(f"Saving model to {filepath}")
        
        if self.save_weights_only:
            torch.save(logs['model'].state_dict(), filepath)
        else:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': logs['model'].state_dict(),
                'optimizer_state_dict': logs['optimizer'].state_dict(),
                'best': self.best,
                'metrics': {k: v for k, v in logs.items() if k not in ['model', 'optimizer']}
            }
            
            if 'lr_scheduler' in logs:
                checkpoint['lr_scheduler_state_dict'] = logs['lr_scheduler'].state_dict()
            
            torch.save(checkpoint, filepath)


class EarlyStopping(Callback):
    """Callback for early stopping."""
    
    def __init__(self, 
                 monitor: str = 'val_loss',
                 min_delta: float = 0,
                 patience: int = 0,
                 mode: str = 'min',
                 verbose: bool = False):
        """Initialize EarlyStopping callback.
        
        Args:
            monitor: Quantity to monitor
            min_delta: Minimum change to qualify as improvement
            patience: Number of epochs with no improvement to stop training
            mode: 'min' or 'max'
            verbose: Whether to print messages
        """
        super(EarlyStopping, self).__init__()
        
        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        
        # Initialize variables
        self.wait = 0
        self.stopped_epoch = 0
        self.best = float('inf') if mode == 'min' else float('-inf')
        self.stopped = False
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        """Check if training should be stopped at the end of an epoch."""
        logs = logs or {}
        current = logs.get(self.monitor)
        
        if current is None:
            if self.verbose:
                print(f"Can't check early stopping: {self.monitor} not in logs")
            return
        
        # Check if current model is better
        if (self.mode == 'min' and current < self.best - self.min_delta) or \
           (self.mode == 'max' and current > self.best + self.min_delta):
            if self.verbose:
                print(f"Epoch {epoch+1}: {self.monitor} improved from {self.best:.4f} to {current:.4f}")
            
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.verbose:
                print(f"Epoch {epoch+1}: {self.monitor} did not improve from {self.best:.4f}, wait: {self.wait}/{self.patience}")
            
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.stopped = True
                if 'stop_training' in logs:
                    logs['stop_training'] = True
                if self.verbose:
                    print(f"Epoch {epoch+1}: Early stopping triggered")
    
    def on_train_end(self, logs: Optional[Dict] = None) -> None:
        """Report when early stopping occurred."""
        if self.stopped_epoch > 0 and self.verbose:
            print(f"Early stopping occurred at epoch {self.stopped_epoch+1}")