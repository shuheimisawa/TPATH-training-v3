import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
import logging

from ..models.unet import UNet
from .losses import CombinedLoss

class SegmentationTrainer:
    def __init__(
        self,
        model: UNet,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        loss_weight: float = 0.5,
        output_dir: str = 'experiments'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer and loss
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = CombinedLoss(dice_weight=loss_weight)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc='Training')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validation'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        return total_loss / len(self.val_loader), np.array(all_preds), np.array(all_targets)

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.output_dir / 'latest_checkpoint.pth')
        
        # Save best model if applicable
        if is_best:
            torch.save(checkpoint, self.output_dir / 'best_model.pth')
            self.logger.info(f'Saved best model with validation loss: {val_loss:.4f}')

    def train(self, num_epochs, early_stopping_patience=10):
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.logger.info(f'Epoch {epoch+1}/{num_epochs}')
            
            # Training phase
            train_loss = self.train_epoch()
            self.logger.info(f'Training Loss: {train_loss:.4f}')
            
            # Validation phase
            val_loss, val_preds, val_targets = self.validate()
            self.logger.info(f'Validation Loss: {val_loss:.4f}')
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            self.save_checkpoint(epoch, val_loss, is_best)
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                self.logger.info(f'Early stopping triggered after {epoch+1} epochs')
                break

# Example usage:
# trainer = SegmentationTrainer(
#     model=model,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     device='cuda',
#     learning_rate=1e-4,
#     output_dir='experiments'
# )
# trainer.train(num_epochs=100) 