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


# Update to scripts/process_svs.py

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


def create_classifier(config, device, checkpoint_path):
    """Factory function to create and initialize classifier.
    
    Args:
        config: Model configuration
        device: Device to run the model on
        checkpoint_path: Path to checkpoint to load
        
    Returns:
        Initialized classifier
    """
    from src.models.glomeruli_classifier import GlomeruliClassifier
    
    model = GlomeruliClassifier(
        num_classes=config.classification.num_classes,
        in_channels=config.classification.in_channels,
        feature_dim=config.classification.feature_dim
    )
    model.to(device)
    
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Try loading as direct state dict
                model.load_state_dict(checkpoint)
            print(f"Loaded classifier checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Error loading classifier checkpoint: {e}")
            raise
    else:
        raise FileNotFoundError(f"Classifier checkpoint not found: {checkpoint_path}")
    
    return model


def parse_args():
    parser = argparse.ArgumentParser(description='Process SVS/WSI files for glomeruli segmentation')
    
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to detection model checkpoint')
    parser.add_argument('--classifier-path', type=str, 
                        help='Path to classifier model checkpoint (for two-stage pipeline)')
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
    parser.add_argument('--two-stage', action='store_true',
                        help='Use two-stage pipeline (detection + classification)')
    parser.add_argument('--normalization-method', type=str, default='vahadane',
                        choices=['macenko', 'reinhard', 'vahadane'],
                        help='Stain normalization method')
    parser.add_argument('--reference-image', type=str,
                        help='Path to reference image for stain normalization')
    
    return parser.parse_args()


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
        
        # Create configuration
        model_config = ModelConfig()
        
        # Update config from args
        model_config.two_stage = args.two_stage
        model_config.normalization.method = args.normalization_method
        model_config.normalization.reference_image_path = args.reference_image
        model_config.detection.score_threshold = args.confidence_threshold
        
        # Create models
        try:
            # Create detection model
            detection_model = create_model(model_config, device, args.model_path)
            detection_model.eval()
            
            # Create classifier model if two-stage pipeline
            classification_model = None
            if args.two_stage:
                if not args.classifier_path:
                    logger.error("Classifier path required for two-stage pipeline")
                    return 1
                
                classification_model = create_classifier(model_config, device, args.classifier_path)
                classification_model.eval()
                logger.info("Created two-stage pipeline")
            
        except Exception as e:
            logger.error(f"Failed to create model: {e}")
            return 1
        
        # Create inference pipeline
        if args.two_stage:
            from src.inference.two_stage_pipeline import TwoStagePipeline
            
            # Convert model_config to dict for TwoStagePipeline
            config_dict = {
                'normalization': {
                    'method': model_config.normalization.method,
                    'reference_image_path': model_config.normalization.reference_image_path,
                    'params_path': model_config.normalization.params_path,
                },
                'feature_extraction': {
                    'use_texture_features': model_config.feature_extraction.use_texture_features,
                    'gabor_frequencies': model_config.feature_extraction.gabor_frequencies,
                    'gabor_orientations': model_config.feature_extraction.gabor_orientations,
                    'lbp_radius': model_config.feature_extraction.lbp_radius,
                    'lbp_points': model_config.feature_extraction.lbp_points,
                    'use_color_features': model_config.feature_extraction.use_color_features,
                    'color_bins': model_config.feature_extraction.color_bins,
                    'use_morphological_features': model_config.feature_extraction.use_morphological_features
                },
                'detection': {
                    'score_threshold': model_config.detection.score_threshold,
                    'nms_threshold': model_config.detection.nms_threshold
                },
                'classification': {
                    'patch_size': model_config.classification.patch_size,
                    'confidence_threshold': model_config.classification.confidence_threshold
                }
            }
            
            pipeline = TwoStagePipeline(
                detection_model=detection_model,
                classification_model=classification_model,
                device=device,
                config=config_dict,
                class_names=args.class_names
            )
        else:
            # Use original SlideInference
            pipeline = SlideInference(
                model=detection_model,
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
                if args.two_stage:
                    # Process with two-stage pipeline
                    results = pipeline.process_slide(
                        slide_path=slide_path,
                        tile_size=args.tile_size,
                        overlap=args.overlap,
                        level=args.level,
                        filter_background=args.filter_background
                    )
                    
                    # Visualize results if requested
                    if args.visualize:
                        # Make sure tile_output_dir exists
                        os.makedirs(tile_output_dir, exist_ok=True)
                        
                        # Visualize each tile
                        for tile_result in results['results']:
                            # Read tile
                            slide_reader = SlideReader(slide_path)
                            tile = slide_reader.read_region(
                                location=(tile_result['x'], tile_result['y']),
                                level=tile_result['level'],
                                size=(tile_result['width'], tile_result['height'])
                            )
                            tile_array = np.array(tile)
                            
                            # Visualize
                            vis_path = os.path.join(tile_output_dir, f"tile_{tile_result['index']:06d}.png")
                            pipeline.visualize_results(
                                image=tile_array,
                                results=tile_result,
                                output_path=vis_path
                            )
                            
                            slide_reader.close()
                else:
                    # Process with original SlideInference
                    results = pipeline.run_inference(
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
                    if args.two_stage:
                        # Two-stage pipeline results
                        serializable_result = {
                            'x': tile_result['x'],
                            'y': tile_result['y'],
                            'width': tile_result['width'],
                            'height': tile_result['height'],
                            'level': tile_result['level'],
                            'index': tile_result['index'],
                            'detection': {
                                'boxes': tile_result['detection']['boxes'].tolist() if 'boxes' in tile_result['detection'] else [],
                                'labels': tile_result['detection']['labels'].tolist() if 'labels' in tile_result['detection'] else [],
                                'scores': tile_result['detection']['scores'].tolist() if 'scores' in tile_result['detection'] else []
                            },
                            'classification': {
                                'class_labels': tile_result['classification']['class_labels'].tolist() if 'class_labels' in tile_result['classification'] else [],
                                'class_scores': tile_result['classification']['class_scores'].tolist() if 'class_scores' in tile_result['classification'] else []
                            },
                            'combined': {
                                'boxes': tile_result['combined']['boxes'].tolist() if 'boxes' in tile_result['combined'] else [],
                                'labels': tile_result['combined']['labels'].tolist() if 'labels' in tile_result['combined'] else [],
                                'scores': tile_result['combined']['scores'].tolist() if 'scores' in tile_result['combined'] else [],
                                'class_names': tile_result['combined']['class_names'] if 'class_names' in tile_result['combined'] else []
                            }
                        }
                    else:
                        # Original pipeline results
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
                    
                    # Masks are too large to store in JSON
                    # Skip or save reference
                    
                    serializable_results['results'].append(serializable_result)
                
                # Save results
                save_json(serializable_results, results_path)
                logger.info(f"Saved results to {results_path}")
                
                # Visualize slide results
                if args.visualize:
                    try:
                        # Create visualization of detections on slide thumbnail
                        vis_path = os.path.join(slide_output_dir, 'visualization.png')
                        
                        if args.two_stage:
                            # Open slide to get a thumbnail
                            slide_reader = SlideReader(slide_path)
                            thumb_size = (1024, 1024)
                            thumb = slide_reader.get_slide_thumbnail(thumb_size)
                            thumb_array = np.array(thumb)
                            
                            # Collect all detections
                            boxes = []
                            labels = []
                            scores = []
                            class_names = []
                            
                            for tile_result in results['results']:
                                tile_boxes = tile_result['combined']['boxes']
                                tile_labels = tile_result['combined']['labels']
                                tile_scores = tile_result['combined']['scores']
                                tile_class_names = tile_result['combined']['class_names']
                                
                                # Scale boxes to thumbnail size
                                slide_width, slide_height = slide_reader.dimensions
                                scale_x = thumb_size[0] / slide_width
                                scale_y = thumb_size[1] / slide_height
                                
                                for box in tile_boxes:
                                    scaled_box = [
                                        int(box[0] * scale_x),
                                        int(box[1] * scale_y),
                                        int(box[2] * scale_x),
                                        int(box[3] * scale_y)
                                    ]
                                    boxes.append(scaled_box)
                                
                                labels.extend(tile_labels)
                                scores.extend(tile_scores)
                                class_names.extend(tile_class_names)
                            
                            # Draw on thumbnail
                            vis = thumb_array.copy()
                            
                            for i in range(len(boxes)):
                                # Get detection info
                                box = boxes[i]
                                label = labels[i]
                                score = scores[i]
                                class_name = class_names[i]
                                
                                # Generate color based on class
                                color_map = {
                                    'Normal': (0, 255, 0),  # Green
                                    'Sclerotic': (255, 0, 0),  # Red
                                    'Partially_sclerotic': (255, 255, 0),  # Yellow
                                    'Uncertain': (0, 0, 255)  # Blue
                                }
                                
                                color = color_map.get(class_name, (255, 255, 255))
                                
                                # Draw bounding box
                                x1, y1, x2, y2 = box
                                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                            
                            # Save visualization
                            cv2.imwrite(vis_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
                            
                            slide_reader.close()
                        else:
                            # Use original visualization method
                            pipeline.visualize_slide_results(
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
                        
                        if args.two_stage:
                            # Implement custom stitching for two-stage pipeline
                            # (This would be a more complex implementation than shown here)
                            logger.warning("Stitching not implemented for two-stage pipeline")
                        else:
                            # Use original stitching method
                            pipeline.stitch_tile_predictions(
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