# scripts/process_svs.py
import os
import sys
import argparse
import torch
import glob
import json
import traceback
from tqdm import tqdm

from src.config.model_config import ModelConfig
from src.models.cascade_mask_rcnn import CascadeMaskRCNN
from src.utils.logger import get_logger
from src.utils.io import save_json
from src.inference.slide_inference import SlideInference
from src.utils.directml_adapter import get_dml_device, is_available

def parse_args():
    parser = argparse.ArgumentParser(description='Process SVS/WSI files for glomeruli segmentation')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Path to directory with SVS/WSI files')
    parser.add_argument('--output-dir', type=str, default='experiments/results',
                        help='Path to output directory')
    parser.add_argument('--tile-size', type=int, default=1024,
                        help='Size of tiles (width and height)')
    parser.add_argument('--overlap', type=int, default=256,
                        help='Overlap between adjacent tiles in pixels')
    parser.add_argument('--level', type=int, default=0,
                        help='Magnification level to process (0 is highest)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size for inference')
    parser.add_argument('--confidence-threshold', type=float, default=0.5,
                        help='Confidence threshold for detections')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use')
    parser.add_argument('--filter-background', action='store_true',
                        help='Filter out background tiles')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize predictions')
    parser.add_argument('--stitch', action='store_true',
                        help='Stitch tile predictions into a whole slide image')
    parser.add_argument('--slide-pattern', type=str, default='*',
                        help='Pattern to match slide filenames (e.g., "patient_*")')
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
            name="process_svs",
            log_file=os.path.join(args.output_dir, "process_svs.log")
        )
        
        # Set device
        if is_available():
            device = get_dml_device(args.gpu)
            logger.info(f"Using DirectML device for AMD GPU")
        else:
            device = torch.device('cpu')
            logger.info("Using CPU (No DirectML device available)")
        
        # Create model
        try:
            model_config = ModelConfig()
            model = create_model(model_config, device, args.model_path)
            model.eval()
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            return 1
        
        # Create slide inference
        slide_inference = SlideInference(
            model=model,
            device=device,
            tile_size=args.tile_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
            level=args.level,
            confidence_threshold=args.confidence_threshold,
            filter_background=args.filter_background
        )
        
        # Find SVS/WSI files
        slide_extensions = ['.svs', '.ndpi', '.tif', '.tiff']
        slide_paths = []
        
        for ext in slide_extensions:
            slide_paths.extend(glob.glob(os.path.join(args.input_dir, f'{args.slide_pattern}{ext}')))
            # Also check subdirectories if needed
            slide_paths.extend(glob.glob(os.path.join(args.input_dir, '**', f'{args.slide_pattern}{ext}'), recursive=True))
        
        logger.info(f"Found {len(slide_paths)} slide files")
        
        if len(slide_paths) == 0:
            logger.warning(f"No slides found in {args.input_dir} with pattern {args.slide_pattern}")
            return 1
        
        # Process each slide
        for slide_path in tqdm(slide_paths, desc="Processing slides"):
            try:
                logger.info(f"Processing {slide_path}")
                
                # Create output directory for this slide
                slide_name = os.path.splitext(os.path.basename(slide_path))[0]
                slide_output_dir = os.path.join(args.output_dir, slide_name)
                os.makedirs(slide_output_dir, exist_ok=True)
                
                # Create directory for tile visualizations
                tile_output_dir = os.path.join(slide_output_dir, 'tiles') if args.visualize else None
                
                # Run inference
                results = slide_inference.run_inference(
                    slide_path=slide_path,
                    output_dir=tile_output_dir
                )
                
                # Save results
                if 'error' in results:
                    logger.error(f"Error processing slide {slide_path}: {results['error']}")
                    continue
                
                # Save results as JSON
                results_path = os.path.join(slide_output_dir, 'results.json')
                
                # Convert tensors to lists for JSON serialization
                serializable_results = {
                    'slide_path': results['slide_path'],
                    'results': []
                }
                
                for tile_result in results['results']:
                    prediction = tile_result['prediction']
                    
                    serializable_result = {
                        'x': tile_result['x'],
                        'y': tile_result['y'],
                        'width': tile_result['width'],
                        'height': tile_result['height'],
                        'level': tile_result['level'],
                        'index': tile_result['index'],
                        'prediction': {
                            'boxes': prediction['boxes'].cpu().numpy().tolist() if 'boxes' in prediction else [],
                            'labels': prediction['labels'].cpu().numpy().tolist() if 'labels' in prediction else [],
                            'scores': prediction['scores'].cpu().numpy().tolist() if 'scores' in prediction else []
                        }
                    }
                    
                    if 'masks' in prediction and prediction['masks'] is not None:
                        # Masks are too large to store in JSON
                        # Save a reference to where they would be stored
                        serializable_result['prediction']['masks_shape'] = list(prediction['masks'].shape)
                    
                    serializable_results['results'].append(serializable_result)
                
                # Save results
                save_json(serializable_results, results_path)
                logger.info(f"Saved results to {results_path}")
                
                # Visualize slide results
                if args.visualize:
                    try:
                        # Create visualization of detections on slide thumbnail
                        vis_path = os.path.join(slide_output_dir, 'visualization.png')
                        slide_inference.visualize_slide_results(
                            slide_path=slide_path,
                            results=results,
                            output_path=vis_path,
                            class_names=args.class_names
                        )
                        logger.info(f"Saved visualization to {vis_path}")
                    except Exception as e:
                        logger.error(f"Error visualizing slide results: {e}")
                
                # Stitch tile predictions
                if args.stitch:
                    try:
                        # Create stitched image
                        stitch_path = os.path.join(slide_output_dir, 'stitched.png')
                        slide_inference.stitch_tile_predictions(
                            slide_path=slide_path,
                            results=results,
                            output_path=stitch_path
                        )
                        logger.info(f"Saved stitched image to {stitch_path}")
                    except Exception as e:
                        logger.error(f"Error stitching slide predictions: {e}")
            
            except Exception as e:
                logger.error(f"Error processing slide {slide_path}: {e}")
                logger.error(traceback.format_exc())
                continue
        
        logger.info("Processing completed")
        return 0
    
    except Exception as e:
        print(f"Critical error: {e}")
        print(traceback.format_exc())
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)