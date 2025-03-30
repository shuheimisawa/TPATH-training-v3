# src/utils/visualization.py
import os
import matplotlib.pyplot as plt

def plot_learning_curves(train_metrics, val_metrics, output_dir):
    """
    Plot training and validation learning curves.
    
    Args:
        train_metrics: Dictionary of training metrics
        val_metrics: Dictionary of validation metrics
        output_dir: Output directory for saving plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    plt.plot(train_metrics['loss'], 'b-', label='Training Loss')
    
    if val_metrics['loss']:
        # Adjust for validation frequency
        x_val = list(range(0, len(train_metrics['loss']), len(train_metrics['loss']) // len(val_metrics['loss'])))
        if len(x_val) > len(val_metrics['loss']):
            x_val = x_val[:len(val_metrics['loss'])]
        plt.plot(x_val, val_metrics['loss'], 'r-', label='Validation Loss')
    
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_curve.png'))
    plt.close()
    
    # Plot component losses
    plt.figure(figsize=(15, 10))
    
    # RPN Box Loss
    plt.subplot(2, 3, 1)
    plt.plot(train_metrics['loss_rpn_box'], 'b-', label='Train')
    if val_metrics['loss_rpn_box']:
        plt.plot(x_val, val_metrics['loss_rpn_box'], 'r-', label='Val')
    plt.title('RPN Box Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # RPN Classification Loss
    plt.subplot(2, 3, 2)
    plt.plot(train_metrics['loss_rpn_cls'], 'b-', label='Train')
    if val_metrics['loss_rpn_cls']:
        plt.plot(x_val, val_metrics['loss_rpn_cls'], 'r-', label='Val')
    plt.title('RPN Classification Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Box Regression Loss
    plt.subplot(2, 3, 3)
    plt.plot(train_metrics['loss_box_reg'], 'b-', label='Train')
    if val_metrics['loss_box_reg']:
        plt.plot(x_val, val_metrics['loss_box_reg'], 'r-', label='Val')
    plt.title('Box Regression Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Classification Loss
    plt.subplot(2, 3, 4)
    plt.plot(train_metrics['loss_cls'], 'b-', label='Train')
    if val_metrics['loss_cls']:
        plt.plot(x_val, val_metrics['loss_cls'], 'r-', label='Val')
    plt.title('Classification Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    # Mask Loss
    plt.subplot(2, 3, 5)
    plt.plot(train_metrics['loss_mask'], 'b-', label='Train')
    if val_metrics['loss_mask']:
        plt.plot(x_val, val_metrics['loss_mask'], 'r-', label='Val')
    plt.title('Mask Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'component_loss_curves.png'))
    plt.close()