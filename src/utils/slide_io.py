"""Utilities for reading slide images."""

import os
import logging
from PIL import Image
import numpy as np
import cv2

from src.utils.logger import get_logger

# Disable PIL image size limit before any image operations
Image.MAX_IMAGE_PIXELS = None
Image.warnings.simplefilter('ignore', Image.DecompressionBombWarning)


class SlideReader:
    """Class for reading slide images."""
    
    def __init__(self, slide_path):
        """Initialize the slide reader.
        
        Args:
            slide_path: Path to the slide file
        """
        self.logger = get_logger(name="slide_reader")
        self.slide_path = slide_path
        
        # Open slide
        self.logger.info(f"Opening slide: {slide_path}")
        try:
            # Use OpenCV for large images
            self.img = cv2.imread(slide_path, cv2.IMREAD_UNCHANGED)
            if self.img is None:
                raise ValueError(f"Failed to load image: {slide_path}")
            
            # Convert BGR to RGB if needed
            if len(self.img.shape) == 3:
                self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
            
            # Get image dimensions
            self.height, self.width = self.img.shape[:2]
            self.logger.info(f"Slide dimensions: {self.width}x{self.height}")
            
            # For now, we'll only use the highest resolution level
            self.n_levels = 1
            self.level_dimensions = [(self.width, self.height)]
            self.logger.info(f"Number of levels: {self.n_levels}")
            
            self.logger.info(f"Successfully opened slide: {slide_path}")
        except Exception as e:
            self.logger.error(f"Error opening slide {slide_path}: {e}")
            raise
    
    def get_dimensions(self, level=0):
        """Get slide dimensions at specified level.
        
        Args:
            level: Magnification level (0 is highest resolution)
            
        Returns:
            Tuple of (width, height)
        """
        if 0 <= level < len(self.level_dimensions):
            return self.level_dimensions[level]
        return self.width, self.height
    
    def read_region(self, location, level, size):
        """Read a region from the slide.
        
        Args:
            location: Tuple of (x, y) coordinates
            level: Magnification level (0 is highest resolution)
            size: Tuple of (width, height)
            
        Returns:
            PIL Image object
        """
        self.logger.info(f"Reading region at ({location[0]}, {location[1]}) with size {size} at level {level}")
        
        try:
            # Extract region using OpenCV
            x, y = location
            w, h = size
            region = self.img[y:y+h, x:x+w]
            
            # Convert to PIL Image
            region = Image.fromarray(region)
            
            self.logger.info("Successfully read region")
            return region
            
        except Exception as e:
            self.logger.error(f"Error reading region: {e}")
            raise
    
    def close(self):
        """Close the slide."""
        self.img = None
        self.logger.info("Closed slide")


class TileExtractor:
    """Class for extracting tiles from slides."""
    
    def __init__(self, tile_size=512, overlap=64, level=0):
        """Initialize the tile extractor.
        
        Args:
            tile_size: Size of tiles to extract
            overlap: Overlap between tiles
            level: Magnification level to extract tiles from
        """
        self.logger = get_logger(name="tile_extractor")
        self.tile_size = tile_size
        self.overlap = overlap
        self.level = level
    
    def extract_tiles(self, slide, filter_background=False, background_threshold=220):
        """Extract tiles from a slide.
        
        Args:
            slide: SlideReader object
            filter_background: Whether to filter out background tiles
            background_threshold: Threshold for background filtering
            
        Returns:
            List of (location, tile) tuples
        """
        self.logger.info(f"Extracting tiles from slide with size {slide.tile_size}")
        
        width, height = slide.get_dimensions(self.level)
        tiles = []
        
        # Calculate tile coordinates
        for y in range(0, height, self.tile_size - self.overlap):
            for x in range(0, width, self.tile_size - self.overlap):
                try:
                    # Extract tile
                    tile = slide.read_region((x, y), self.level, (self.tile_size, self.tile_size))
                    
                    # Filter background if requested
                    if filter_background:
                        # Convert to numpy array
                        tile_array = np.array(tile)
                        
                        # Calculate mean intensity
                        mean_intensity = np.mean(tile_array)
                        
                        # Skip if too bright (likely background)
                        if mean_intensity > background_threshold:
                            self.logger.info(f"Skipping background tile at ({x}, {y})")
                            continue
                    
                    tiles.append(((x, y), tile))
                    
                except Exception as e:
                    self.logger.error(f"Error extracting tile at ({x}, {y}): {e}")
                    continue
        
        self.logger.info(f"Extracted {len(tiles)} tiles")
        return tiles 