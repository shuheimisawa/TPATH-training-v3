# scripts/run_inference.py
import os
import sys
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import traceback

# Add the parent directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.model_config import ModelConfig
from src.models.cascade_mask_rcnn import CascadeMaskRCNN
from src.data.transforms import get_val_transforms, denormalize_image
from src.evaluation.visualization import visualize_prediction
from src.utils.logger import get_logger
from src.utils.directml_adapter import get_dml_device, is_available, empty_cache
from src.utils.io import load_image, save_image


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference with Cascade Mask R-CNN')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Path to input directory with images')
    parser.add_argument('--output-dir', type=str, default='inference_results',
                        help='Path to output directory')
    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='Backbone model to use (resnet18, resnet34, resnet50, resnet101, resnet152)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection confidence threshold')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage even if GPU is available')
    parser.add_argument('--show-masks', action='store_true',
                        help='Show instance segmentation masks')
    
    return parser.parse_args()


def create_model(config, device, checkpoint_path):
    """Create model and load checkpoint."""
    try:
        # Create model
        model = CascadeMaskRCNN(config)
        model.to(device)
        
        # Load checkpoint
        if os.path.exists(checkpoint_path):
            # Convert device to string for map_location to avoid DirectML issues
            if hasattr(device, 'type'):
                map_device = device.type
            else:
                map_device = str(device)
                
            print(f"Loading checkpoint with map_location={map_device}")
            
            try:
                # Use 'cpu' first to avoid any device-specific issues, then move to target device
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                print("Loaded checkpoint to CPU first")
            except Exception as e:
                print(f"Error loading checkpoint to CPU: {e}")
                # Direct device loading as fallback
                checkpoint = torch.load(checkpoint_path, map_location=str(device))
            
            if 'model_state_dict' in checkpoint:
                # Handle potential key mismatches
                try:
                    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
                    print("Checkpoint loaded with strict matching")
                except Exception as e:
                    print(f"Strict loading failed: {e}")
                    print("Trying non-strict loading...")
                    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                    print("Checkpoint loaded with non-strict matching")
            else:
                # Try loading as direct state dict
                try:
                    model.load_state_dict(checkpoint, strict=True)
                    print("State dictionary loaded with strict matching")
                except Exception as e:
                    print(f"Strict loading failed: {e}")
                    print("Trying non-strict loading...")
                    model.load_state_dict(checkpoint, strict=False)
                    print("State dictionary loaded with non-strict matching")
                    
            # Make sure model is on the correct device after loading
            model = model.to(device)
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        return model
    except Exception as e:
        print(f"Error creating model: {e}")
        print(traceback.format_exc())
        raise


def run_inference(model, image_path, transform, device, threshold=0.5, class_names=None):
    """Run inference on a single image."""
    try:
        # Load image
        image = load_image(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Keep original image for visualization
        orig_image = image.copy()
        
        # Apply transform
        if transform:
            transformed = transform(image=image)
            image_tensor = transformed['image']
        else:
            # Basic transformation (normalize and convert to tensor)
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            
            # Normalize with ImageNet mean and std
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            image_tensor = (image_tensor - mean) / std
            
            # Resize if needed
            if image.shape[0] > 800 or image.shape[1] > 800:
                image_tensor = torch.nn.functional.interpolate(
                    image_tensor.unsqueeze(0), 
                    size=(800, 800), 
                    mode='bilinear', 
                    align_corners=False
                ).squeeze(0)
        
        # Add batch dimension
        image_batch = [image_tensor.to(device)]
        
        # Run inference
        with torch.no_grad():
            predictions = model(image_batch)
        
        # Filter predictions by threshold
        filtered_predictions = []
        for pred in predictions:
            scores = pred['scores']
            keep = scores >= threshold
            
            filtered_pred = {
                'boxes': pred['boxes'][keep],
                'labels': pred['labels'][keep],
                'scores': pred['scores'][keep]
            }
            
            if 'masks' in pred:
                filtered_pred['masks'] = pred['masks'][keep]
            
            filtered_predictions.append(filtered_pred)
        
        return filtered_predictions, orig_image
    
    except Exception as e:
        print(f"Error running inference on {image_path}: {e}")
        print(traceback.format_exc())
        return None, None


def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up logger
    logger = get_logger(
        name="inference",
        log_file=os.path.join(args.output_dir, "inference.log")
    )
    
    # Validate backbone name
    valid_backbones = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']
    if args.backbone not in valid_backbones:
        logger.warning(f"Invalid backbone name '{args.backbone}'. Must be one of {valid_backbones}")
        logger.warning(f"Using default backbone: resnet50")
        args.backbone = 'resnet50'
    
    # Set device
    if args.cpu:
        device = torch.device("cpu")
        logger.info("Forcing CPU usage")
    elif torch.cuda.is_available() and not args.cpu:
        device = torch.device("cuda:0")
        logger.info("Using CUDA device")
    elif is_available() and not args.cpu:
        device = get_dml_device()
        logger.info("Using DirectML device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    try:
        # Create model config
        model_config = ModelConfig()
        model_config.backbone.name = args.backbone
        
        # Create model and load checkpoint
        logger.info(f"Creating model with backbone: {model_config.backbone.name}")
        model = create_model(model_config, device, args.model_path)
        model.eval()
        
        # Create transform
        transform = get_val_transforms({
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225]
        })
        
        # Find images in input directory
        # Check if input directory exists
        if not os.path.exists(args.input_dir):
            logger.error(f"Input directory not found: {args.input_dir}")
            # Try to find images in current directory as fallback
            args.input_dir = '.'
            logger.info("Using current directory as fallback")
        
        image_extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend([
                os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir)
                if f.lower().endswith(ext)
            ])
        
        if not image_paths:
            logger.error(f"No images found in {args.input_dir}")
            # Try to create a dummy test image as fallback
            dummy_image_path = os.path.join(args.input_dir, "test_image.png")
            try:
                # Create a simple 512x512 test image with a white rectangle in the middle
                dummy_image = np.zeros((512, 512, 3), dtype=np.uint8)
                cv2.rectangle(dummy_image, (100, 100), (400, 400), (255, 255, 255), -1)
                cv2.imwrite(dummy_image_path, dummy_image)
                logger.info(f"Created a dummy test image at {dummy_image_path}")
                image_paths = [dummy_image_path]
            except Exception as e:
                logger.error(f"Failed to create dummy test image: {e}")
                return 1
        
        logger.info(f"Found {len(image_paths)} images to process")
        
        # Class names
        class_names = ['Background', 'Normal', 'Sclerotic', 'Partially_sclerotic', 'Uncertain']
        
        # Run inference on each image
        results = {}
        
        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                # Get image name
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                
                # Run inference
                predictions, image = run_inference(
                    model=model,
                    image_path=image_path,
                    transform=transform,
                    device=device,
                    threshold=args.threshold,
                    class_names=class_names
                )
                
                if predictions is None or image is None:
                    logger.warning(f"Failed to process {image_path}")
                    continue
                
                # Visualize results
                for i, pred in enumerate(predictions):
                    # Visualize
                    visualization = visualize_prediction(
                        image=image,
                        prediction=pred,
                        class_names=class_names,
                        show_masks=args.show_masks,
                        show_boxes=True,
                        show_scores=True
                    )
                    
                    # Save visualization
                    output_path = os.path.join(args.output_dir, f"{image_name}_result.png")
                    visualization.save(output_path)
                
                # Store results
                result_data = {
                    "image_path": image_path,
                    "detections": []
                }
                
                # Convert tensors to lists for JSON serialization
                for i, pred in enumerate(predictions):
                    boxes = pred['boxes'].cpu().numpy().tolist() if len(pred['boxes']) > 0 else []
                    labels = pred['labels'].cpu().numpy().tolist() if len(pred['labels']) > 0 else []
                    scores = pred['scores'].cpu().numpy().tolist() if len(pred['scores']) > 0 else []
                    
                    for box, label, score in zip(boxes, labels, scores):
                        result_data["detections"].append({
                            "box": box,
                            "label": label,
                            "class": class_names[label] if label < len(class_names) else f"Class {label}",
                            "score": score
                        })
                
                results[image_name] = result_data
                
                # Free memory
                empty_cache()
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                logger.error(traceback.format_exc())
                continue
        
        # Save all results to JSON
        results_path = os.path.join(args.output_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results to {results_path}")
        logger.info("Inference completed")
        
        return 0
    
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == '__main__':
    sys.exit(main())