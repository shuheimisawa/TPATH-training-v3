"""Script to convert SVS files to TIFF format."""

import os
import argparse
from glob import glob
from tqdm import tqdm
from PIL import Image
import numpy as np

from src.utils.logger import get_logger


def convert_svs_to_tiff(input_dir, output_dir):
    """Convert SVS files to TIFF format.
    
    Args:
        input_dir: Directory containing SVS files
        output_dir: Directory to save TIFF files
    """
    logger = get_logger(name="convert_svs")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all SVS files
    svs_files = glob(os.path.join(input_dir, "*.svs"))
    logger.info(f"Found {len(svs_files)} SVS files")
    
    # Process each file
    for svs_path in tqdm(svs_files, desc="Converting SVS files"):
        try:
            # Get filename without extension
            filename = os.path.splitext(os.path.basename(svs_path))[0]
            output_path = os.path.join(output_dir, f"{filename}.tiff")
            
            # Skip if output file already exists
            if os.path.exists(output_path):
                logger.info(f"Skipping {filename} - output file already exists")
                continue
            
            # Open SVS file
            logger.info(f"Opening {filename}")
            
            # Temporarily increase PIL's image size limit
            max_pixels = Image.MAX_IMAGE_PIXELS
            Image.MAX_IMAGE_PIXELS = None
            
            try:
                img = Image.open(svs_path)
                
                # Save as TIFF
                logger.info(f"Saving {filename} as TIFF")
                img.save(output_path, format="TIFF", compression="tiff_lzw")
                
                # Close image
                img.close()
                
                logger.info(f"Successfully converted {filename}")
                
            finally:
                # Restore PIL's image size limit
                Image.MAX_IMAGE_PIXELS = max_pixels
            
        except Exception as e:
            logger.error(f"Error converting {filename}: {e}")
            continue


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Convert SVS files to TIFF format')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing SVS files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save TIFF files')
    
    args = parser.parse_args()
    
    # Convert files
    convert_svs_to_tiff(args.input_dir, args.output_dir)


if __name__ == '__main__':
    main() 