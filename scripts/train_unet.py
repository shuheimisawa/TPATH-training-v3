import argparse
import torch
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.unet import UNet
from src.data.dataset import SegmentationDataset
from src.training.segmentation_trainer import SegmentationTrainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train U-Net for glomeruli segmentation')
    parser.add_argument('--data_dir', type=str, required=True,
                      help='Directory containing processed data')
    parser.add_argument('--output_dir', type=str, default='experiments',
                      help='Directory to save model checkpoints and logs')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100,
                      help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                      help='Learning rate for optimizer')
    parser.add_argument('--val_split', type=float, default=0.2,
                      help='Fraction of data to use for validation')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='cuda',
                      help='Device to use for training (cuda or cpu)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize dataset
    dataset = SegmentationDataset(
        data_dir=args.data_dir,
        transform=None  # Add transforms if needed
    )
    
    # Split dataset into train and validation
    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    model = UNet(
        n_channels=3,  # RGB images
        n_classes=5,   # Background + 4 glomeruli types
        bilinear=True
    )
    
    # Initialize trainer
    trainer = SegmentationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir
    )
    
    # Train the model
    trainer.train(
        num_epochs=args.num_epochs,
        early_stopping_patience=10
    )

if __name__ == '__main__':
    main() 