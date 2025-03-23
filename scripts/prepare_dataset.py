"""Script to prepare the dataset for training."""

import os
import argparse
import json
import random
from glob import glob
from tqdm import tqdm
from PIL import Image
import numpy as np
import cv2

# Disable PIL image size limit before any image operations
Image.MAX_IMAGE_PIXELS = None
Image.warnings.simplefilter('ignore', Image.DecompressionBombWarning)

print("Starting dataset preparation script...")

from src.utils.logger import get_logger
from src.utils.slide_io import SlideReader
from src.utils.annotation_utils import load_annotations, process_annotations


def prepare_dataset(input_dir, annotations_dir, output_dir, tile_size=512, overlap=64, train_ratio=0.7, val_ratio=0.15):
    """Prepare the dataset for training.
    
    Args:
        input_dir: Directory containing slide images
        annotations_dir: Directory containing annotation files
        output_dir: Directory to save processed dataset
        tile_size: Size of tiles to extract
        overlap: Overlap between tiles
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
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
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    
    for split_dir in [train_dir, val_dir, test_dir]:
        os.makedirs(split_dir, exist_ok=True)
        logger.info(f"Created output directory: {split_dir}")
    
    # Find all slide files
    slide_files = []
    for ext in ['.tiff', '.tif', '.svs', '.ndpi']:
        files = glob(os.path.join(input_dir, f"*{ext}"))
        logger.info(f"Found {len(files)} files with extension {ext}")
        logger.info(f"Files with extension {ext}: {files}")
        slide_files.extend(files)
    
    logger.info(f"Found {len(slide_files)} total slide files")
    logger.info(f"All slide files: {slide_files}")
    
    if not slide_files:
        logger.error(f"No slide files found in {input_dir}")
        return
    
    # Randomly split slides into train/val/test
    random.shuffle(slide_files)
    n_train = int(len(slide_files) * train_ratio)
    n_val = int(len(slide_files) * val_ratio)
    
    train_slides = slide_files[:n_train]
    val_slides = slide_files[n_train:n_train + n_val]
    test_slides = slide_files[n_train + n_val:]
    
    logger.info(f"Split dataset into {len(train_slides)} training, {len(val_slides)} validation, and {len(test_slides)} test slides")
    logger.info(f"Training slides: {train_slides}")
    logger.info(f"Validation slides: {val_slides}")
    logger.info(f"Test slides: {test_slides}")
    
    # Process each split
    split_annotations = {
        'train': {},
        'val': {},
        'test': {}
    }
    
    for split, slides, split_dir in [('train', train_slides, train_dir), 
                                   ('val', val_slides, val_dir), 
                                   ('test', test_slides, test_dir)]:
        logger.info(f"Processing {split} split")
        
        # Process each slide
        for slide_path in tqdm(slides, desc=f"Processing {split} slides"):
            try:
                # Get slide name
                slide_name = os.path.splitext(os.path.basename(slide_path))[0]
                logger.info(f"Processing slide: {slide_name}")
                
                # Load annotations
                annotation_path = os.path.join(annotations_dir, f"{slide_name}_annotations.json")
                if not os.path.exists(annotation_path):
                    logger.warning(f"No annotations found for {slide_name}")
                    continue
                
                logger.info(f"Loading annotations from {annotation_path}")
                annotations = load_annotations(annotation_path)
                logger.info(f"Loaded {len(annotations)} annotations")
                
                # Create slide reader
                logger.info(f"Creating slide reader for {slide_path}")
                slide = SlideReader(slide_path)
                logger.info(f"Created slide reader")
                
                # Create slide-specific output directory
                slide_output_dir = os.path.join(split_dir, slide_name)
                os.makedirs(slide_output_dir, exist_ok=True)
                
                # Process annotations and extract tiles
                logger.info(f"Processing annotations and extracting tiles")
                tile_annotations = process_annotations(slide, annotations, slide_output_dir, tile_size, overlap)
                
                # Add to split annotations
                if tile_annotations:
                    split_annotations[split].update(tile_annotations)
                
                # Close slide
                slide.close()
                logger.info(f"Finished processing slide: {slide_name}")
                
            except Exception as e:
                logger.error(f"Error processing {slide_name}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                continue
    
    # Save split annotations
    for split, annotations in split_annotations.items():
        annotation_path = os.path.join(output_dir, split, f"{split}_annotations.json")
        try:
            with open(annotation_path, 'w') as f:
                json.dump(annotations, f, indent=2)
            logger.info(f"Saved {len(annotations)} annotations for {split} split to {annotation_path}")
        except Exception as e:
            logger.error(f"Error saving annotations for {split} split: {e}")
    
    logger.info("Dataset preparation completed")


def main():
    print("Entering main function...")
    
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
    
    args = parser.parse_args()
    print(f"Arguments parsed: {args}")
    
    # Prepare dataset
    prepare_dataset(args.input_dir, args.annotations_dir, args.output_dir,
                   args.tile_size, args.overlap, args.train_ratio, args.val_ratio)


if __name__ == '__main__':
    main()