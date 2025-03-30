# New file: scripts/train_classifier.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import argparse
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from src.models.glomeruli_classifier import GlomeruliClassifier
from src.data.dataset import GlomeruliDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.utils.logger import get_logger
from src.utils.io import save_json
from src.config.model_config import ModelConfig


def train_one_epoch(model, dataloader, criterion, optimizer, device, logger):
    """Train model for one epoch."""
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Get data
        images = batch['image'].to(device)
        targets = batch['target']
        
        # Get labels
        labels = targets['labels'].to(device)
        
        # Forward pass
        outputs = model(images)
        logits = outputs['logits']
        
        # Compute loss
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        epoch_loss += loss.item()
        
        # Compute accuracy
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    # Compute epoch metrics
    epoch_loss /= len(dataloader)
    epoch_acc = correct / total
    
    logger.info(f"Training Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}")
    
    return {'loss': epoch_loss, 'acc': epoch_acc}


def validate(model, dataloader, criterion, device, logger, class_names=None):
    """Validate model."""
    model.eval()
    val_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Get data
            images = batch['image'].to(device)
            targets = batch['target']
            
            # Get labels
            labels = targets['labels'].to(device)
            
            # Forward pass
            outputs = model(images)
            logits = outputs['logits']
            
            # Compute loss
            loss = criterion(logits, labels)
            
            # Update metrics
            val_loss += loss.item()
            
            # Get predictions
            _, predictions = torch.max(logits, 1)
            
            # Store for metrics
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute validation metrics
    val_loss /= len(dataloader)
    
    # Compute classification metrics
    class_report = classification_report(
        all_labels, all_preds, 
        target_names=class_names if class_names else None,
        output_dict=True
    )
    
    # Compute accuracy
    accuracy = class_report['accuracy']
    
    logger.info(f"Validation Loss: {val_loss:.4f}, Acc: {accuracy:.4f}")
    
    # Print classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(all_labels, all_preds, 
                                    target_names=class_names if class_names else None))
    
    return {
        'loss': val_loss, 
        'acc': accuracy, 
        'report': class_report,
        'predictions': all_preds,
        'ground_truth': all_labels
    }


def plot_confusion_matrix(ground_truth, predictions, class_names, output_path):
    """Plot confusion matrix."""
    # Compute confusion matrix
    cm = confusion_matrix(ground_truth, predictions)
    
    # Create figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_metrics(train_metrics, val_metrics, output_dir):
    """Plot training and validation metrics."""
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_metrics['loss'], label='Train')
    plt.plot(val_metrics['loss'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_metrics['acc'], label='Train')
    plt.plot(val_metrics['acc'], label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics.png'), dpi=300)
    plt.close()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train glomeruli classifier')
    
    parser.add_argument('--train-dir', type=str, required=True,
                        help='Path to training data directory')
    parser.add_argument('--val-dir', type=str, required=True,
                        help='Path to validation data directory')
    parser.add_argument('--output-dir', type=str, default='experiments/classifier',
                        help='Path to output directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--class-names', type=str, nargs='+',
                        default=['Normal', 'Sclerotic', 'Partially_sclerotic', 'Uncertain'],
                        help='Class names')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create logger
    logger = get_logger(
        name="train_classifier",
        log_file=os.path.join(args.output_dir, "train.log")
    )
    
    # Log arguments
    logger.info(f"Arguments: {args}")
    
    # Set device
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset = GlomeruliDataset(
        data_dir=args.train_dir,
        mode='train',
        transform=get_train_transforms()
    )
    
    val_dataset = GlomeruliDataset(
        data_dir=args.val_dir,
        mode='val',
        transform=get_val_transforms()
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model_config = ModelConfig()
    
    model = GlomeruliClassifier(
        num_classes=len(args.class_names),
        in_channels=3,
        feature_dim=model_config.classification.feature_dim
    )
    
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.1, 
        patience=5,
        verbose=True
    )
    
    # Train model
    logger.info("Starting training...")
    
    best_val_acc = 0
    best_epoch = 0
    
    # Metrics tracking
    train_metrics = {'loss': [], 'acc': []}
    val_metrics = {'loss': [], 'acc': []}
    
    for epoch in range(args.epochs):
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        
        # Train one epoch
        train_results = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            logger=logger
        )
        
        # Validate
        val_results = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            logger=logger,
            class_names=args.class_names
        )
        
        # Update learning rate
        scheduler.step(val_results['loss'])
        
        # Save metrics
        train_metrics['loss'].append(train_results['loss'])
        train_metrics['acc'].append(train_results['acc'])
        val_metrics['loss'].append(val_results['loss'])
        val_metrics['acc'].append(val_results['acc'])
        
        # Save model if best
        if val_results['acc'] > best_val_acc:
            best_val_acc = val_results['acc']
            best_epoch = epoch
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'val_loss': val_results['loss']
            }, os.path.join(args.output_dir, 'best_model.pth'))
            
            logger.info(f"Saved best model with validation accuracy: {best_val_acc:.4f}")
            
            # Plot confusion matrix
            plot_confusion_matrix(
                ground_truth=val_results['ground_truth'],
                predictions=val_results['predictions'],
                class_names=args.class_names,
                output_path=os.path.join(args.output_dir, 'confusion_matrix.png')
            )
            
            # Save validation report
            save_json(
                val_results['report'],
                os.path.join(args.output_dir, 'validation_report.json')
            )
        
        # Save latest model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_results['acc'],
            'val_loss': val_results['loss']
        }, os.path.join(args.output_dir, 'latest_model.pth'))
    
    # Plot metrics
    plot_metrics(train_metrics, val_metrics, args.output_dir)
    
    # Save final metrics
    save_json({
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc
    }, os.path.join(args.output_dir, 'metrics.json'))
    
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}")


if __name__ == '__main__':
    main()