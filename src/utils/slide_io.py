"""Utilities for reading slide images."""

import os
from PIL import Image
import numpy as np
import openslide
from src.utils.logger import get_logger
from typing import Tuple, Optional
import logging

# Disable PIL's size limit
Image.MAX_IMAGE_PIXELS = None

class SlideReader:
    """Class for reading slide images."""
    
    def __init__(self, slide_path: str):
        """Initialize a slide reader.
        
        Args:
            slide_path (str): Path to the slide file.
        """
        self.logger = get_logger(name="slide_reader")
        self.slide_path = slide_path
        
        # Check if file exists
        if not os.path.exists(slide_path):
            raise FileNotFoundError(f"Slide file not found: {slide_path}")
            
        # Check file extension
        ext = os.path.splitext(slide_path)[1].lower()
        if ext not in ['.svs', '.tiff', '.tif']:
            raise ValueError(f"Unsupported file format: {ext}")
            
        # Open slide with OpenSlide
        try:
            self.slide = openslide.OpenSlide(slide_path)
            self.width, self.height = self.slide.dimensions
            self.level_count = self.slide.level_count
            self.level_dimensions = self.slide.level_dimensions
            self.level_downsamples = self.slide.level_downsamples
            self.dimensions = (self.width, self.height)
            logging.info(f"Opened slide {slide_path} with dimensions {self.width}x{self.height}")
            
            # Get slide properties
            self.properties = self.slide.properties
            logging.info(f"Slide properties: {self.properties}")
            
        except Exception as e:
            logging.error(f"Error opening slide {slide_path}: {e}")
            raise
    
    def open(self):
        """Open the slide image."""
        try:
            self.logger.info(f"Slide dimensions: {self.width}x{self.height}")
        except Exception as e:
            self.logger.error(f"Error opening slide {self.slide_path}: {e}")
            raise
    
    def close(self):
        """Close the slide image."""
        if hasattr(self, 'slide'):
            self.slide.close()
            self.logger.info("Closed slide")
    
    def get_dimensions(self) -> Tuple[int, int]:
        """Get the dimensions of the slide.
        
        Returns:
            Tuple[int, int]: The width and height of the slide.
        """
        return self.dimensions
    
    def read_region(self, location: Tuple[int, int], level: int, size: Tuple[int, int]) -> Image.Image:
        """Read a region from the slide.
        
        Args:
            location (Tuple[int, int]): (x, y) coordinates of the top-left corner.
            level (int): Magnification level.
            size (Tuple[int, int]): (width, height) of the region to read.
            
        Returns:
            PIL.Image.Image: The image region as a PIL Image.
        """
        try:
            # Read region using OpenSlide
            region = self.slide.read_region(location, level, size)
            # Convert to RGB
            region = region.convert('RGB')
            return region
        except Exception as e:
            self.logger.error(f"Error reading region from {self.slide_path}: {str(e)}")
            raise
    
    def __del__(self):
        """Clean up resources."""
        self.close()
    
    def get_slide_thumbnail(self, size):
        """Get a thumbnail of the slide.
        
        Args:
            size: Tuple of (width, height) for the thumbnail
            
        Returns:
            PIL Image object
        """
        try:
            # Get the best level for thumbnail
            target_size = max(size)
            best_level = self.slide.get_best_level_for_downsample(self.width / target_size)
            
            # Read the whole image at this level
            thumb = self.slide.read_region((0, 0), best_level, self.level_dimensions[best_level])
            thumb = thumb.convert('RGB')
            
            # Resize to exact size
            thumb.thumbnail(size, Image.LANCZOS)
            return thumb
        except Exception as e:
            self.logger.error(f"Error getting thumbnail: {e}")
            return Image.new('RGB', size, color=(255, 255, 255))

    def get_best_level_for_downsample(self, downsample: float) -> int:
        """Get the best level for the given downsample factor.
        
        Args:
            downsample (float): The desired downsample factor.
            
        Returns:
            int: The best level for the given downsample factor.
        """
        return self.slide.get_best_level_for_downsample(downsample)


class TileExtractor:
    """Class for extracting tiles from a slide."""
    
    def __init__(self, tile_size: int = 256, overlap: int = 0, level: int = 0):
        """Initialize a tile extractor.
        
        Args:
            tile_size (int): Size of the tiles to extract.
            overlap (int): Overlap between adjacent tiles.
            level (int): Magnification level to extract tiles from.
        """
        self.logger = get_logger(name="tile_extractor")
        self.tile_size = tile_size
        self.overlap = overlap
        self.level = level
        logging.info(f"Initialized TileExtractor with tile_size={tile_size}, overlap={overlap}, level={level}")
    
    def extract_tiles(self, slide_reader: SlideReader, filter_background: bool = False, background_threshold: int = 220) -> list:
        """Extract tiles from a slide.
        
        Args:
            slide_reader (SlideReader): The slide reader to extract tiles from.
            filter_background (bool): Whether to filter out background tiles.
            background_threshold (int): Intensity threshold for background filtering.
            
        Returns:
            list: List of dictionaries containing tile information.
        """
        try:
            width, height = slide_reader.get_dimensions()
            self.logger.info(f"Extracting tiles from slide of size {width}x{height}")

            effective_tile_size = self.tile_size - 2 * self.overlap
            tiles = []

            for y in range(0, height, effective_tile_size):
                for x in range(0, width, effective_tile_size):
                    # Calculate actual tile coordinates with overlap
                    tile_x = max(0, x - self.overlap)
                    tile_y = max(0, y - self.overlap)
                    
                    # Adjust tile size for image boundaries
                    tile_w = min(self.tile_size, width - tile_x)
                    tile_h = min(self.tile_size, height - tile_y)

                    try:
                        tile = slide_reader.read_region((tile_x, tile_y), self.level, (tile_w, tile_h))
                        
                        # Filter background tiles if requested
                        if filter_background:
                            if np.mean(tile) > background_threshold:
                                self.logger.debug(f"Skipping background tile at ({tile_x}, {tile_y})")
                                continue

                        tiles.append({
                            'x': tile_x,
                            'y': tile_y,
                            'width': tile_w,
                            'height': tile_h,
                            'level': self.level
                        })
                    except Exception as e:
                        self.logger.error(f"Error extracting tile at ({tile_x}, {tile_y}): {str(e)}")
                        continue

            self.logger.info(f"Extracted {len(tiles)} tiles")
            return tiles

        except Exception as e:
            self.logger.error(f"Error during tile extraction: {str(e)}")
            raise


class TileStitcher:
    """Class for stitching tiles back into a single image."""
    
    def __init__(self, output_width, output_height, background_color=(255, 255, 255)):
        """Initialize the tile stitcher.
        
        Args:
            output_width: Width of the output image
            output_height: Height of the output image
            background_color: Background color for the output image
        """
        self.logger = get_logger(name="tile_stitcher")
        self.output_width = output_width
        self.output_height = output_height
        self.background_color = background_color
        
        # Create empty output image
        self.output_image = Image.new('RGB', (output_width, output_height), color=background_color)
        
        self.logger.info(f"Initialized tile stitcher with output size {output_width}x{output_height}")
    
    def add_tile(self, tile_image, x, y):
        """Add a tile to the output image.
        
        Args:
            tile_image: PIL Image object
            x: X coordinate in the output image
            y: Y coordinate in the output image
        """
        try:
            # Paste tile at the specified position
            self.output_image.paste(tile_image, (x, y))
        except Exception as e:
            self.logger.error(f"Error adding tile at ({x}, {y}): {e}")
    
    def get_output_image(self):
        """Get the stitched output image.
        
        Returns:
            PIL Image object
        """
        return self.output_image
    
    def save_output_image(self, output_path):
        """Save the stitched output image.
        
        Args:
            output_path: Path to save the output image
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save output image
            self.output_image.save(output_path)
            self.logger.info(f"Saved stitched image to {output_path}")
        except Exception as e:
            self.logger.error(f"Error saving stitched image: {e}")