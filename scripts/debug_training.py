# scripts/debug_training.py
import os
import sys
import torch
import argparse
import traceback
import json
import numpy as np
from PIL import Image
import cv2

# Add the parent directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.model_config import ModelConfig
from src.models.cascade_mask_rcnn import CascadeMaskRCNN
from src.data.dataset import GlomeruliDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.utils.logger import get_logger
from src.utils.directml_adapter import get_dml_device, is_available, empty_cache

def parse_args():
    parser = argparse.ArgumentParser(description='Debug training issues')
    
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default='debug_output',
                        help='Path to output directory')
    
    return parser.parse_args()

def check_images_in_directory(dir_path, logger):
    """Check images in a directory for potential issues."""
    if not os.path.exists(dir_path):
        logger.error(f"Directory does not exist: {dir_path}")
        return
    
    image_files = [f for f in os.listdir(dir_path) if f.endswith('.png') or f.endswith('.jpg')]
    logger.info(f"Found {len(image_files)} image files in {dir_path}")
    
    # Check a few images
    sample_files = image_files[:5] if len(image_files) > 5 else image_files
    
    for img_file in sample_files:
        img_path = os.path.join(dir_path, img_file)
        try:
            # Open with PIL
            img = Image.open(img_path)
            img_np = np.array(img)
            
            logger.info(f"Image {img_file}: Size={img.size}, Mode={img.mode}, Shape={img_np.shape}")
            
            # Check for potential issues
            if img_np.max() == 0:
                logger.warning(f"Image {img_file} appears to be all black (max value = 0)")
            
            if img_np.min() == 255 and img_np.max() == 255:
                logger.warning(f"Image {img_file} appears to be all white")
                
            # Check if this is a visualization image
            if "_vis" in img_file:
                logger.info(f"Image {img_file} appears to be a visualization image")
            
        except Exception as e:
            logger.error(f"Error checking image {img_file}: {e}")

def examine_annotation_structure(annotation_file, logger):
    """Examine the structure of the annotation file"""
    try:
        with open(annotation_file, 'r') as f:
            annotations = json.load(f)
        
        # Check the structure
        logger.info(f"Annotation file contains {len(annotations)} entries")
        
        # Check first entry
        first_key = list(annotations.keys())[0]
        first_entry = annotations[first_key]
        
        if isinstance(first_entry, list):
            logger.info(f"Annotation format: List of annotations per image")
            logger.info(f"Sample entry ({first_key}): List with {len(first_entry)} items")
            
            # Check first annotation
            if first_entry:
                logger.info(f"First annotation structure: {list(first_entry[0].keys())}")
        elif isinstance(first_entry, dict):
            logger.info(f"Annotation format: Dictionary per image")
            logger.info(f"Sample entry ({first_key}): Dictionary with keys {list(first_entry.keys())}")
            
            # Check annotations
            if 'annotations' in first_entry:
                annotations_list = first_entry['annotations']
                logger.info(f"Contains {len(annotations_list)} annotation entries")
                if annotations_list:
                    logger.info(f"First annotation structure: {list(annotations_list[0].keys())}")
        else:
            logger.warning(f"Unexpected annotation format: {type(first_entry)}")
    
    except Exception as e:
        logger.error(f"Error examining annotation file: {e}")
        logger.error(traceback.format_exc())

def main():
    # Parse arguments
    args = parse_args()
    
    # Create logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(
        name="debug_training",
        log_file=os.path.join(args.output_dir, "debug.log")
    )
    
    # Set device
    if is_available():
        device = get_dml_device()
        logger.info("Using DirectML device")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info("Using CUDA device")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    
    try:
        # Step 1: Verify data directories
        train_dir = os.path.join(args.data_dir, 'train')
        val_dir = os.path.join(args.data_dir, 'val')
        test_dir = os.path.join(args.data_dir, 'test')
        
        logger.info(f"Checking data directories:")
        for dir_path in [train_dir, val_dir, test_dir]:
            if os.path.exists(dir_path):
                logger.info(f"  {dir_path}: exists")
                # Count files
                image_files = [f for f in os.listdir(dir_path) if f.endswith('.png') and not f.endswith('_vis.png')]
                vis_files = [f for f in os.listdir(dir_path) if f.endswith('_vis.png')]
                logger.info(f"    Found {len(image_files)} image files and {len(vis_files)} visualization files")
                
                # Check for potential image issues
                check_images_in_directory(dir_path, logger)
            else:
                logger.error(f"  {dir_path}: does not exist")
        
        # Step 2: Check annotation files
        logger.info(f"Checking annotation files:")
        for mode, dir_path in [('train', train_dir), ('val', val_dir), ('test', test_dir)]:
            annotation_file = os.path.join(dir_path, f"{mode}_annotations.json")
            if os.path.exists(annotation_file):
                logger.info(f"  {annotation_file}: exists")
                
                # Examine annotation structure
                examine_annotation_structure(annotation_file, logger)
            else:
                logger.error(f"  {annotation_file}: does not exist")
        
        # Step 3: Test dataset loading
        logger.info("Testing dataset loading...")
        try:
            transform = get_train_transforms({
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            })
            
            train_dataset = GlomeruliDataset(data_dir=train_dir, mode='train', transform=transform)
            logger.info(f"Successfully created training dataset with {len(train_dataset)} samples")
            
            # Check a few samples
            if len(train_dataset) > 0:
                logger.info("Testing sample access...")
                for i in range(min(3, len(train_dataset))):
                    try:
                        sample = train_dataset[i]
                        logger.info(f"  Sample {i}: image shape {sample['image'].shape}")
                        
                        # Log info about boxes
                        boxes = sample['target']['boxes']
                        labels = sample['target']['labels']
                        logger.info(f"    Found {len(boxes)} boxes with labels: {labels.tolist()}")
                        
                        # Save sample image for inspection
                        img = sample['image']
                        if isinstance(img, torch.Tensor):
                            # Convert tensor to numpy
                            img_np = img.permute(1, 2, 0).numpy()
                            # Denormalize
                            img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                            img_np = (img_np * 255).astype(np.uint8)
                        else:
                            img_np = img
                            
                        Image.fromarray(img_np).save(os.path.join(args.output_dir, f"sample_{i}.png"))
                        
                    except Exception as e:
                        logger.error(f"  Error accessing sample {i}: {e}")
                        logger.error(traceback.format_exc())
        except Exception as e:
            logger.error(f"Error testing dataset: {e}")
            logger.error(traceback.format_exc())
        
        # Step 4: Test model initialization
        logger.info("Testing model initialization...")
        try:
            model_config = ModelConfig()
            # Simplify the model to avoid memory issues
            model_config.backbone.name = 'resnet50'
            model_config.fpn.num_outs = 3
            model_config.cascade.num_stages = 2
            model_config.use_bifpn = False
            model_config.use_attention = False
            
            model = CascadeMaskRCNN(model_config)
            model.to(device)
            logger.info("Successfully created model")
            
            # Test with a properly formatted dummy input
            logger.info("Testing forward pass with properly formatted dummy input...")
            # Create a 3D tensor (C, H, W) without batch dimension
            dummy_input = torch.randn(3, 512, 512, device=device)
            
            # Create a list containing this tensor (model expects a list of images)
            dummy_input_list = [dummy_input]
            
            # Set model to eval mode for testing
            model.eval()
            
            with torch.no_grad():
                try:
                    outputs = model(dummy_input_list)
                    logger.info(f"Forward pass successful")
                    # Log detected objects
                    boxes = outputs[0]['boxes']
                    scores = outputs[0]['scores']
                    labels = outputs[0]['labels']
                    logger.info(f"  Detected {len(boxes)} objects")
                    if len(boxes) > 0:
                        logger.info(f"  First box: {boxes[0].tolist()}, Score: {scores[0].item()}, Label: {labels[0].item()}")
                except Exception as e:
                    logger.error(f"Error in forward pass: {e}")
                    logger.error(traceback.format_exc())
        except Exception as e:
            logger.error(f"Error testing model: {e}")
            logger.error(traceback.format_exc())
        
        # Step 5: Check CUDA/memory status
        logger.info("Checking CUDA/memory status...")
        try:
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                current_device = torch.cuda.current_device()
                device_name = torch.cuda.get_device_name(current_device)
                
                logger.info(f"CUDA available: True")
                logger.info(f"Device count: {device_count}")
                logger.info(f"Current device: {current_device}")
                logger.info(f"Device name: {device_name}")
                
                # Get memory info
                memory_allocated = torch.cuda.memory_allocated(current_device) / (1024**2)  # MB
                memory_reserved = torch.cuda.memory_reserved(current_device) / (1024**2)    # MB
                memory_max_allocated = torch.cuda.max_memory_allocated(current_device) / (1024**2)  # MB
                
                logger.info(f"Memory allocated: {memory_allocated:.2f} MB")
                logger.info(f"Memory reserved: {memory_reserved:.2f} MB")
                logger.info(f"Max memory allocated: {memory_max_allocated:.2f} MB")
            else:
                logger.info("CUDA not available")
        except Exception as e:
            logger.error(f"Error checking CUDA status: {e}")
            
        logger.info("Debug completed")
        
    except Exception as e:
        logger.error(f"Debugging failed: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()