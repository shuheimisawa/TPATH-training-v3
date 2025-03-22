# scripts/analyze_svs.py
import os
import argparse
import glob
from tqdm import tqdm
import openslide
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from src.utils.slide_io import SlideReader
from src.utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze SVS/WSI files in a directory')
    
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Path to directory with SVS/WSI files')
    parser.add_argument('--output-dir', type=str, default='analysis_results',
                        help='Path to output directory')
    parser.add_argument('--save-thumbnails', action='store_true',
                        help='Save thumbnails of slides')
    parser.add_argument('--thumbnail-size', type=int, default=1024,
                        help='Maximum dimension of thumbnails')
    
    return parser.parse_args()


def analyze_slide(slide_path, output_dir, save_thumbnail=False, thumbnail_size=1024):
    """Analyze a single slide file.
    
    Args:
        slide_path: Path to the SVS/WSI file
        output_dir: Directory to save analysis results
        save_thumbnail: Whether to save a thumbnail of the slide
        thumbnail_size: Maximum dimension of the thumbnail
        
    Returns:
        Dictionary with slide analysis results
    """
    logger = get_logger(name="analyze_svs")
    logger.info(f"Analyzing {slide_path}")
    
    slide_name = os.path.basename(slide_path)
    slide_dir = os.path.join(output_dir, os.path.splitext(slide_name)[0])
    os.makedirs(slide_dir, exist_ok=True)
    
    try:
        # Open slide
        slide_reader = SlideReader(slide_path)
        
        # Get slide properties
        slide_info = {
            'slide_name': slide_name,
            'width': slide_reader.width,
            'height': slide_reader.height,
            'level_count': slide_reader.level_count,
            'level_dimensions': slide_reader.level_dimensions,
            'level_downsamples': slide_reader.level_downsamples,
            'magnification': None,
            'vendor': None,
            'mpp_x': None,
            'mpp_y': None
        }
        
        # Extract additional properties from slide metadata
        try:
            properties = slide_reader.slide.properties
            if openslide.PROPERTY_NAME_OBJECTIVE_POWER in properties:
                slide_info['magnification'] = float(properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
            if openslide.PROPERTY_NAME_MPP_X in properties:
                slide_info['mpp_x'] = float(properties[openslide.PROPERTY_NAME_MPP_X])
            if openslide.PROPERTY_NAME_MPP_Y in properties:
                slide_info['mpp_y'] = float(properties[openslide.PROPERTY_NAME_MPP_Y])
            if openslide.PROPERTY_NAME_VENDOR in properties:
                slide_info['vendor'] = properties[openslide.PROPERTY_NAME_VENDOR]
        except Exception as e:
            logger.warning(f"Error extracting additional properties: {e}")
        
        # Calculate tissue percentage for each level
        slide_info['tissue_percentage'] = []
        
        for level in range(slide_reader.level_count):
            # Use a smaller level for large slides to avoid memory issues
            if level > 2:
                continue
                
            # Get level dimensions
            width, height = slide_reader.level_dimensions[level]
            
            # Read level image
            try:
                level_image = slide_reader.read_region((0, 0), level, (width, height))
                
                # Convert to numpy array
                level_array = np.array(level_image)
                
                # Convert to grayscale
                level_gray = (0.299 * level_array[:, :, 0] + 0.587 * level_array[:, :, 1] + 0.114 * level_array[:, :, 2]).astype(np.uint8)
                
                # Threshold to detect tissue (non-white pixels)
                # Adjust threshold as needed
                tissue_mask = level_gray < 220
                
                # Calculate tissue percentage
                tissue_percent = np.mean(tissue_mask) * 100
                
                slide_info['tissue_percentage'].append((level, tissue_percent))
                
                # Debugging: save tissue mask for first few levels
                if level <= 2:
                    mask_image = Image.fromarray(tissue_mask.astype(np.uint8) * 255)
                    mask_path = os.path.join(slide_dir, f"tissue_mask_level{level}.png")
                    mask_image.save(mask_path)
            except Exception as e:
                logger.warning(f"Error processing level {level}: {e}")
        
        # Save thumbnail if requested
        if save_thumbnail:
            try:
                thumbnail = slide_reader.get_slide_thumbnail((thumbnail_size, thumbnail_size))
                thumbnail_path = os.path.join(slide_dir, "thumbnail.png")
                thumbnail.save(thumbnail_path)
                logger.info(f"Saved thumbnail to {thumbnail_path}")
            except Exception as e:
                logger.warning(f"Error saving thumbnail: {e}")
        
        # Close slide
        slide_reader.close()
        
        return slide_info
    
    except Exception as e:
        logger.error(f"Error analyzing slide {slide_path}: {e}")
        return {
            'slide_name': slide_name,
            'error': str(e)
        }


def main():
    # Parse arguments
    args = parse_args()
    
    # Create logger
    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(
        name="analyze_svs",
        log_file=os.path.join(args.output_dir, "analyze_svs.log")
    )
    
    # Find SVS/WSI files
    slide_extensions = ['.svs', '.ndpi', '.tif', '.tiff']
    slide_paths = []
    
    for ext in slide_extensions:
        slide_paths.extend(glob.glob(os.path.join(args.input_dir, f'*{ext}')))
    
    logger.info(f"Found {len(slide_paths)} slide files")
    
    # Analyze slides
    results = []
    
    for slide_path in tqdm(slide_paths, desc="Analyzing slides"):
        slide_info = analyze_slide(
            slide_path=slide_path,
            output_dir=args.output_dir,
            save_thumbnail=args.save_thumbnails,
            thumbnail_size=args.thumbnail_size
        )
        
        results.append(slide_info)
    
    # Create summary DataFrame
    summary_data = []
    
    for result in results:
        if 'error' in result:
            summary_data.append({
                'slide_name': result['slide_name'],
                'width': None,
                'height': None,
                'level_count': None,
                'magnification': None,
                'vendor': None,
                'mpp_x': None,
                'mpp_y': None,
                'tissue_percentage': None,
                'status': 'error',
                'error': result['error']
            })
        else:
            tissue_percent = result['tissue_percentage'][0][1] if result['tissue_percentage'] else None
            
            summary_data.append({
                'slide_name': result['slide_name'],
                'width': result['width'],
                'height': result['height'],
                'level_count': result['level_count'],
                'magnification': result['magnification'],
                'vendor': result['vendor'],
                'mpp_x': result['mpp_x'],
                'mpp_y': result['mpp_y'],
                'tissue_percentage': tissue_percent,
                'status': 'success',
                'error': None
            })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary to CSV
    summary_path = os.path.join(args.output_dir, "slide_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary to {summary_path}")
    
    # Generate summary plots
    try:
        # Plot slide dimensions
        plt.figure(figsize=(10, 6))
        plt.scatter(summary_df['width'], summary_df['height'])
        plt.xlabel('Width (pixels)')
        plt.ylabel('Height (pixels)')
        plt.title('Slide Dimensions')
        plt.savefig(os.path.join(args.output_dir, "slide_dimensions.png"))
        plt.close()
        
        # Plot tissue percentage histogram
        plt.figure(figsize=(10, 6))
        plt.hist(summary_df['tissue_percentage'].dropna(), bins=20)
        plt.xlabel('Tissue Percentage')
        plt.ylabel('Count')
        plt.title('Tissue Percentage Distribution')
        plt.savefig(os.path.join(args.output_dir, "tissue_percentage.png"))
        plt.close()
        
        # Plot magnification histogram if available
        if summary_df['magnification'].notna().any():
            plt.figure(figsize=(10, 6))
            plt.hist(summary_df['magnification'].dropna(), bins=10)
            plt.xlabel('Magnification')
            plt.ylabel('Count')
            plt.title('Magnification Distribution')
            plt.savefig(os.path.join(args.output_dir, "magnification.png"))
            plt.close()
        
        logger.info("Generated summary plots")
    except Exception as e:
        logger.warning(f"Error generating summary plots: {e}")
    
    logger.info("Analysis completed")


if __name__ == '__main__':
    main()