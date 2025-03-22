# scripts/evaluate.py
import os
import json
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np

from src.config.model_config import ModelConfig
from src.models.cascade_mask_rcnn import CascadeMaskRCNN
from src.data.dataset import GlomeruliDataset
from src.data.transforms import get_val_transforms
from src.evaluation.evaluator import Evaluator
from src.evaluation.visualization import visualize_batch_predictions
from src.utils.logger import get_logger
from src.utils.io import load_image, save_image


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Cascade Mask R-CNN for glomeruli segmentation')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data-dir', type=str, default='data/test',
                        help='Path to test data directory')
    parser.add_argument('--output-dir', type=str, default='experiments/results',
                        help='Path to output directory')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for testing')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save predictions to file')
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Create logger
    logger = get_logger(
        name="evaluate",
        log_file=os.path.join(args.output_dir, "evaluate.log")
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model_config = ModelConfig()
    model = CascadeMaskRCNN(model_config)
    
    # Load model checkpoint
    logger.info(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create dataset and data loader
    test_transform = get_val_transforms({
        'mean': (0.485, 0.456, 0.406),
        'std': (0.229, 0.224, 0.225)
    })
    
    test_dataset = GlomeruliDataset(
        data_dir=args.data_dir,
        transform=test_transform,
        mode='test'
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda batch: tuple(zip(*batch))
    )
    
    # Create evaluator
    evaluator = Evaluator(device=device)
    
    # Evaluate model
    logger.info("Starting evaluation")
    
    all_predictions = []
    all_targets = []
    all_images = []
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(test_loader):
            logger.info(f"Processing batch {i+1}/{len(test_loader)}")
            
            # Move data to device
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Get predictions
            predictions = model(images)
            
            # Store predictions, targets, and images
            all_predictions.extend(predictions)
            all_targets.extend(targets)
            
            # Convert images from tensor to numpy for visualization
            numpy_images = []
            for img in images:
                # Denormalize
                img = img.cpu().numpy().transpose(1, 2, 0)
                img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                img = (img * 255).astype(np.uint8)
                numpy_images.append(img)
            all_images.extend(numpy_images)
            
            # Visualize predictions
            if args.visualize and i < 5:  # Visualize first 5 batches
                visualize_batch_predictions(
                    numpy_images,
                    predictions,
                    class_names=['GN', 'GL', 'GS'],
                    save_path=os.path.join(args.output_dir, f'batch_{i+1}_predictions.png')
                )
    
    # Calculate metrics
    metrics = evaluator.evaluate(all_predictions, all_targets)
    
    # Print metrics
    logger.info("Evaluation metrics:")
    for name, value in metrics.items():
        logger.info(f"{name}: {value:.4f}")
    
    # Save metrics
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions
    if args.save_predictions:
        predictions_dict = {
            f"image_{i}": {
                "boxes": pred["boxes"].cpu().numpy().tolist(),
                "labels": pred["labels"].cpu().numpy().tolist(),
                "scores": pred["scores"].cpu().numpy().tolist()
            }
            for i, pred in enumerate(all_predictions)
        }
        
        with open(os.path.join(args.output_dir, 'predictions.json'), 'w') as f:
            json.dump(predictions_dict, f, indent=2)
    
    logger.info("Evaluation completed")


if __name__ == '__main__':
    main()
