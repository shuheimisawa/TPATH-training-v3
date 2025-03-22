import os
import argparse
import torch
import torch.nn as nn
import torch.onnx
import json

from src.config.model_config import ModelConfig
from src.models.cascade_mask_rcnn import CascadeMaskRCNN
from src.utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Export Cascade Mask R-CNN model')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output-path', type=str, default='experiments/exported_models',
                        help='Path to output directory')
    parser.add_argument('--format', type=str, choices=['onnx', 'torchscript'], default='torchscript',
                        help='Export format')
    
    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Create logger
    logger = get_logger(
        name="export_model",
        log_file=os.path.join(args.output_path, "export.log")
    )
    
    # Create output directory
    os.makedirs(args.output_path, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    
    # Export model
    if args.format == 'torchscript':
        # Create a dummy input
        dummy_input = torch.randn(1, 3, 1024, 1024, device=device)
        
        # Export to TorchScript
        output_path = os.path.join(args.output_path, 'model.pt')
        try:
            traced_script_module = torch.jit.trace(model, dummy_input)
            traced_script_module.save(output_path)
            logger.info(f"Model exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export model: {e}")
            # Try scripting instead of tracing
            try:
                scripted_module = torch.jit.script(model)
                scripted_module.save(output_path)
                logger.info(f"Model exported to {output_path} using script")
            except Exception as e2:
                logger.error(f"Failed to export model using script: {e2}")
    
    elif args.format == 'onnx':
        # Create a dummy input
        dummy_input = torch.randn(1, 3, 1024, 1024, device=device)
        
        # Export to ONNX
        output_path = os.path.join(args.output_path, 'model.onnx')
        try:
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=12,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['boxes', 'labels', 'scores', 'masks'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'boxes': {0: 'batch_size'},
                    'labels': {0: 'batch_size'},
                    'scores': {0: 'batch_size'},
                    'masks': {0: 'batch_size'}
                }
            )
            logger.info(f"Model exported to {output_path}")
        except Exception as e:
            logger.error(f"Failed to export model to ONNX: {e}")
    
    logger.info("Export completed")


if __name__ == '__main__':
    main()
