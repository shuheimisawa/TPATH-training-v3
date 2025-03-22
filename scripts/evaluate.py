# scripts/evaluate.py
import os
import sys
import json
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import traceback

from src.config.model_config import ModelConfig
from src.models.cascade_mask_rcnn import CascadeMaskRCNN
from src.data.dataset import GlomeruliDataset
from src.data.transforms import get_val_transforms
from src.evaluation.evaluator import Evaluator
from src.evaluation.visualization import visualize_batch_predictions
from src.utils.logger import get_logger
from src.utils.io import load_image, save_image, save_json


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
    parser.add_argument('--class-names', type=str, nargs='+', 
                        default=['Normal', 'Sclerotic', 'Partially_sclerotic', 'Uncertain'],
                        help='Names of the classes')
    
    return parser.parse_args()


def create_model(config, device, checkpoint_path):
    """Factory function to create and initialize model.
    
    Args:
        config: Model configuration
        device: Device to run the model on
        checkpoint_path: Path to checkpoint to load
        
    Returns:
        Initialized model
    """
    model = CascadeMaskRCNN(config)
    model.to(device)
    
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Try loading as direct state dict
                model.load_state_dict(checkpoint)
            print(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    return model


def main():
    try:
        # Parse arguments
        args = parse_args()
        
        # Create logger
        os.makedirs(args.output_dir, exist_ok=True)
        logger = get_logger(
            name="evaluate",
            log_file=os.path.join(args.output_dir, "evaluate.log")
        )
        
        # Set device
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Create model
        model_config = ModelConfig()
        
        try:
            model = create_model(model_config, device, args.model_path)
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            return 1
        
        # Set model to evaluation mode
        model.eval()
        
        # Create dataset and data loader
        try:
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
            
            logger.info(f"Created test dataset with {len(test_dataset)} samples")
        except Exception as e:
            logger.error(f"Failed to create dataset: {e}")
            return 1
        
        # Create evaluator
        evaluator = Evaluator(device=device)
        
        # Evaluate model
        logger.info("Starting evaluation")
        
        all_predictions = []
        all_targets = []
        all_images = []
        all_image_ids = []
        
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                try:
                    logger.info(f"Processing batch {i+1}/{len(test_loader)}")
                    
                    # Extract data
                    images, targets = batch
                    
                    # Extract image IDs
                    batch_image_ids = [t.get('image_id', f"img_{i}_{j}") for j, t in enumerate(targets)]
                    
                    # Move data to device
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    
                    # Get predictions
                    predictions = model(images)
                    
                    # Store predictions, targets, and images
                    all_predictions.extend(predictions)
                    all_targets.extend(targets)
                    all_image_ids.extend(batch_image_ids)
                    
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
                            class_names=args.class_names,
                            save_path=os.path.join(args.output_dir, f'batch_{i+1}_predictions.png')
                        )
                except Exception as e:
                    logger.error(f"Error processing batch {i+1}: {e}")
                    continue
        
        # Calculate metrics
        try:
            metrics = evaluator.evaluate(all_predictions, all_targets)
            
            # Calculate per-class metrics
            class_metrics = evaluator.evaluate_classifications(all_predictions, all_targets)
            
            # Combine metrics
            combined_metrics = {
                "overall": metrics,
                "per_class": class_metrics
            }
            
            # Print metrics
            logger.info("Evaluation metrics:")
            for name, value in metrics.items():
                logger.info(f"{name}: {value:.4f}")
            
            # Save metrics
            metrics_path = os.path.join(args.output_dir, 'metrics.json')
            save_json(combined_metrics, metrics_path)
            logger.info(f"Metrics saved to {metrics_path}")
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
        
        # Save predictions
        if args.save_predictions:
            try:
                predictions_dict = {}
                
                for i, (pred, img_id) in enumerate(zip(all_predictions, all_image_ids)):
                    entry = {
                        "boxes": pred["boxes"].cpu().numpy().tolist() if "boxes" in pred else [],
                        "labels": pred["labels"].cpu().numpy().tolist() if "labels" in pred else [],
                        "scores": pred["scores"].cpu().numpy().tolist() if "scores" in pred else []
                    }
                    
                    # Add masks dimensions but not the actual data (too large)
                    if "masks" in pred and pred["masks"] is not None:
                        entry["masks_shape"] = list(pred["masks"].shape)
                    
                    predictions_dict[str(img_id)] = entry
                
                # Save predictions
                predictions_path = os.path.join(args.output_dir, 'predictions.json')
                save_json(predictions_dict, predictions_path)
                logger.info(f"Predictions saved to {predictions_path}")
            except Exception as e:
                logger.error(f"Error saving predictions: {e}")
        
        logger.info("Evaluation completed")
        return 0
    
    except Exception as e:
        print(f"Critical error: {e}")
        print(traceback.format_exc())
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)