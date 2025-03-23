"""Script to prepare the dataset for training."""

import os
import argparse
import json
import random
import gc
import traceback
from glob import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2
import openslide

# Disable PIL's size limit before any image operations
Image.MAX_IMAGE_PIXELS = None
Image.warnings.simplefilter('ignore', Image.DecompressionBombWarning)

from src.utils.logger import get_logger
from src.utils.slide_io import SlideReader
from src.utils.annotation_utils import load_annotations, process_annotations
from src.utils.tissue_detection import process_tile, visualize_tissue_detection


def prepare_dataset(input_dir, annotations_dir, output_dir, tile_size=512, overlap=64, 
                    train_ratio=0.7, val_ratio=0.15, max_tiles_per_batch=1000,
                    filter_background=True, background_threshold=240, min_tissue_percentage=0.08):
    """Prepare the dataset for training with improved memory handling.
    
    Args:
        input_dir: Directory containing slide images
        annotations_dir: Directory containing annotation files
        output_dir: Directory to save processed dataset
        tile_size: Size of tiles to extract
        overlap: Overlap between tiles
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        max_tiles_per_batch: Maximum number of tiles to process in a batch
        filter_background: Whether to filter out background tiles
        background_threshold: Threshold for background filtering (0-255)
        min_tissue_percentage: Minimum percentage of tissue required (0-1)
    """
    logger = get_logger(name="prepare_dataset", log_file=os.path.join(output_dir, "prepare_dataset.log"))
    
    logger.info("Starting dataset preparation with parameters:")
    logger.info(f"- Input directory: {input_dir}")
    logger.info(f"- Annotations directory: {annotations_dir}")
    logger.info(f"- Output directory: {output_dir}")
    logger.info(f"- Tile size: {tile_size}")
    logger.info(f"- Overlap: {overlap}")
    logger.info(f"- Train ratio: {train_ratio}")
    logger.info(f"- Val ratio: {val_ratio}")
    logger.info(f"- Max tiles per batch: {max_tiles_per_batch}")
    logger.info(f"- Filter background: {filter_background}")
    logger.info(f"- Background threshold: {background_threshold}")
    logger.info(f"- Minimum tissue percentage: {min_tissue_percentage}")
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    for split_dir in [train_dir, val_dir, test_dir]:
        os.makedirs(split_dir, exist_ok=True)
        logger.info(f"Created output directory: {split_dir}")
    
    # Find all slide files
    slide_files = []
    for ext in ['.svs', '.tiff', '.tif']:
        files = glob(os.path.join(input_dir, f"*{ext}"))
        slide_files.extend(files)
    
    logger.info(f"Found {len(slide_files)} slide files")
    
    if not slide_files:
        logger.error(f"No slide files found in {input_dir}")
        return
    
    # Check for previously processed splits
    split_info_path = os.path.join(output_dir, 'split_info.json')
    if os.path.exists(split_info_path):
        try:
            with open(split_info_path, 'r') as f:
                split_info = json.load(f)
            
            logger.info(f"Loaded existing split info from {split_info_path}")
            train_slides = split_info['train']
            val_slides = split_info['val']
            test_slides = split_info['test']
            
            # Verify all files still exist
            for slide_path in train_slides + val_slides + test_slides:
                if not os.path.exists(slide_path):
                    logger.warning(f"Slide file not found: {slide_path}")
                    raise FileNotFoundError("Some slide files are missing")
            
            logger.info(f"Using existing split: {len(train_slides)} training, {len(val_slides)} validation, {len(test_slides)} test slides")
        except Exception as e:
            logger.warning(f"Could not use existing split info: {e}")
            # Create new splits
            random.shuffle(slide_files)
            n_train = int(len(slide_files) * train_ratio)
            n_val = int(len(slide_files) * val_ratio)
            
            train_slides = slide_files[:n_train]
            val_slides = slide_files[n_train:n_train + n_val]
            test_slides = slide_files[n_train + n_val:]
            
            # Save split info
            split_info = {
                'train': train_slides,
                'val': val_slides,
                'test': test_slides
            }
            
            with open(split_info_path, 'w') as f:
                json.dump(split_info, f, indent=2)
            
            logger.info(f"Created new split: {len(train_slides)} training, {len(val_slides)} validation, {len(test_slides)} test slides")
    else:
        # Randomly split slides into train/val/test
        random.shuffle(slide_files)
        n_train = int(len(slide_files) * train_ratio)
        n_val = int(len(slide_files) * val_ratio)
        
        train_slides = slide_files[:n_train]
        val_slides = slide_files[n_train:n_train + n_val]
        test_slides = slide_files[n_train + n_val:]
        
        # Save split info
        split_info = {
            'train': train_slides,
            'val': val_slides,
            'test': test_slides
        }
        
        with open(split_info_path, 'w') as f:
            json.dump(split_info, f, indent=2)
        
        logger.info(f"Created new split: {len(train_slides)} training, {len(val_slides)} validation, {len(test_slides)} test slides")
    
    # Process each split
    split_annotations = {
        'train': {},
        'val': {},
        'test': {}
    }
    
    # Check for processing file to avoid reprocessing slides
    processed_slides_path = os.path.join(output_dir, 'processed_slides.json')
    processed_slides = set()
    
    if os.path.exists(processed_slides_path):
        try:
            with open(processed_slides_path, 'r') as f:
                processed_slides = set(json.load(f))
            logger.info(f"Found {len(processed_slides)} previously processed slides")
        except Exception as e:
            logger.warning(f"Could not load processed slides info: {e}")
    
    # Process each split
    for split, slides, split_dir in [('train', train_slides, train_dir), 
                                   ('val', val_slides, val_dir), 
                                   ('test', test_slides, test_dir)]:
        logger.info(f"Processing {split} split")
        
        # Load existing annotations if available
        split_annotations_path = os.path.join(split_dir, f"{split}_annotations.json")
        if os.path.exists(split_annotations_path):
            try:
                with open(split_annotations_path, 'r') as f:
                    split_annotations[split] = json.load(f)
                logger.info(f"Loaded {len(split_annotations[split])} existing annotations for {split} split")
            except Exception as e:
                logger.warning(f"Could not load existing annotations for {split} split: {e}")
        
        # Process each slide
        for slide_idx, slide_path in enumerate(tqdm(slides, desc=f"Processing {split} slides")):
            try:
                # Get slide name
                slide_name = os.path.splitext(os.path.basename(slide_path))[0]
                logger.info(f"Processing slide {slide_idx+1}/{len(slides)}: {slide_name}")
                
                # Skip if already processed
                if slide_path in processed_slides:
                    logger.info(f"Skipping {slide_name} - already processed")
                    continue
                
                # Load annotations
                annotation_path = os.path.join(annotations_dir, f"{slide_name}_annotations.json")
                if not os.path.exists(annotation_path):
                    logger.warning(f"No annotations found for {slide_name}")
                    # Add to processed slides to avoid checking again
                    processed_slides.add(slide_path)
                    continue
                
                logger.info(f"Loading annotations from {annotation_path}")
                annotations = load_annotations(annotation_path)
                logger.info(f"Loaded {len(annotations)} annotations")
                
                if not annotations:
                    logger.warning(f"No valid annotations found for {slide_name}")
                    # Add to processed slides to avoid checking again
                    processed_slides.add(slide_path)
                    continue
                
                # Create slide reader
                logger.info(f"Creating slide reader for {slide_path}")
                try:
                    slide = SlideReader(slide_path)
                    logger.info(f"Created slide reader - dimensions: {slide.width}x{slide.height}, levels: {slide.level_count}")
                except Exception as e:
                    logger.error(f"Could not open slide {slide_path}: {e}")
                    continue
                
                # Create slide-specific output directory
                slide_output_dir = os.path.join(split_dir, slide_name)
                os.makedirs(slide_output_dir, exist_ok=True)
                
                # Process annotations and extract tiles
                logger.info(f"Processing annotations and extracting tiles for {slide_name}")
                try:
                    tile_annotations = process_annotations(
                        slide=slide,
                        annotations=annotations,
                        output_dir=split_dir,
                        tile_size=tile_size,
                        overlap=overlap,
                        filter_background=filter_background,
                        background_threshold=background_threshold,
                        min_tissue_percentage=min_tissue_percentage
                    )
                    
                    # Add to split annotations
                    if tile_annotations:
                        split_annotations[split].update(tile_annotations)
                        logger.info(f"Added {len(tile_annotations)} tile annotations for {slide_name}")
                    
                    # Save updated split annotations
                    with open(split_annotations_path, 'w') as f:
                        json.dump(split_annotations[split], f, indent=2)
                    logger.info(f"Saved updated annotations for {split} split")
                    
                except Exception as e:
                    logger.error(f"Error processing annotations for {slide_name}: {e}")
                    logger.error(traceback.format_exc())
                
                # Close slide
                try:
                    slide.close()
                except Exception as e:
                    logger.warning(f"Error closing slide {slide_name}: {e}")
                
                # Add to processed slides
                processed_slides.add(slide_path)
                
                # Save processed slides info
                with open(processed_slides_path, 'w') as f:
                    json.dump(list(processed_slides), f, indent=2)
                
                # Force garbage collection
                gc.collect()
                
            except Exception as e:
                logger.error(f"Error processing slide {slide_name}: {e}")
                logger.error(traceback.format_exc())
                continue
    
    # Save final annotations
    for split in ['train', 'val', 'test']:
        split_annotations_path = os.path.join(output_dir, split, f"{split}_annotations.json")
        try:
            with open(split_annotations_path, 'w') as f:
                json.dump(split_annotations[split], f, indent=2)
            logger.info(f"Saved {len(split_annotations[split])} annotations for {split} split to {split_annotations_path}")
        except Exception as e:
            logger.error(f"Error saving annotations for {split} split: {e}")
    
    logger.info("Dataset preparation completed")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Prepare dataset for training')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing slide images')
    parser.add_argument('--annotations-dir', type=str, required=True,
                        help='Directory containing annotation files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save processed dataset')
    parser.add_argument('--tile-size', type=int, default=512,
                        help='Size of tiles to extract')
    parser.add_argument('--overlap', type=int, default=64,
                        help='Overlap between tiles')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Ratio of training data')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Ratio of validation data')
    parser.add_argument('--max-tiles-per-batch', type=int, default=1000,
                        help='Maximum number of tiles to process in a batch')
    parser.add_argument('--filter-background', action='store_true',
                        help='Filter out background tiles')
    parser.add_argument('--background-threshold', type=int, default=240,
                        help='Threshold for background filtering (0-255)')
    parser.add_argument('--min-tissue-percentage', type=float, default=0.08,
                        help='Minimum percentage of tissue required (0-1)')
    
    args = parser.parse_args()
    
    # Prepare dataset
    prepare_dataset(
        input_dir=args.input_dir,
        annotations_dir=args.annotations_dir,
        output_dir=args.output_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_tiles_per_batch=args.max_tiles_per_batch,
        filter_background=args.filter_background,
        background_threshold=args.background_threshold,
        min_tissue_percentage=args.min_tissue_percentage
    )


if __name__ == '__main__':
    main()