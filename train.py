import os
import torch
from torch.utils.data import DataLoader
from src.config.training_config import TrainingConfig
from src.data.dataset import GlomeruliDataset
from src.models.cascade_mask_rcnn import CascadeMaskRCNN
from src.training.trainer import Trainer

def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create training configuration
    config = TrainingConfig(
        epochs=12,
        batch_size=2,
        data=TrainingConfig.data(
            train_path="data_test/processed/train",
            val_path="data_test/processed/val",
            img_size=(512, 512)
        )
    )
    
    # Create datasets
    train_dataset = GlomeruliDataset(
        data_dir=config.data.train_path,
        mode='train'
    )
    
    val_dataset = GlomeruliDataset(
        data_dir=config.data.val_path,
        mode='val'
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.workers,
        collate_fn=lambda x: tuple(zip(*x))
    )
    
    # Create model
    model = CascadeMaskRCNN(
        num_classes=len(config.data.classes) + 1,  # +1 for background
        pretrained=True
    )
    model.to(device)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()