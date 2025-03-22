import os
import numpy as np
import openslide
from PIL import Image
from typing import Dict, List, Tuple, Optional, Union, Any

from .logger import get_logger


class SlideReader:
    """Utility for reading SVS/whole slide image files."""
    
    def __init__(self, slide_path: str):
        """Initialize the slide reader.
        
        Args:
            slide_path: Path to the SVS/WSI file
        """
        if not os.path.exists(slide_path):
            raise FileNotFoundError(f"Slide file not found at {slide_path}")
        
        # Check if OpenSlide can open the file
        if not openslide.is_openslide(slide_path):
            raise ValueError(f"File {slide_path} is not a valid slide file")
        
        self.slide_path = slide_path
        self.slide = None
        self.logger = get_logger(name="slide_reader")
        
        try:
            self.slide = openslide.OpenSlide(slide_path)
            
            # Get slide properties
            self.width, self.height = self.slide.dimensions
            self.level_count = self.slide.level_count
            self.level_dimensions = self.slide.level_dimensions
            self.level_downsamples = self.slide.level_downsamples
            
            # Get pixel size if available
            self.pixel_size_x = float(self.slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0))
            self.pixel_size_y = float(self.slide.properties.get(openslide.PROPERTY_NAME_MPP_Y, 0))
            
            self.logger.info(f"Loaded slide {os.path.basename(slide_path)}")
            self.logger.info(f"Dimensions: {self.width}x{self.height}")
            self.logger.info(f"Level count: {self.level_count}")
            self.logger.info(f"Level dimensions: {self.level_dimensions}")
            self.logger.info(f"Level downsamples: {self.level_downsamples}")
        except Exception as e:
            if self.slide is not None:
                self.slide.close()
            raise ValueError(f"Failed to open slide {slide_path}: {e}")
    
    def get_slide_thumbnail(self, max_size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
        """Get a thumbnail of the slide.
        
        Args:
            max_size: Maximum size of the thumbnail (width, height)
            
        Returns:
            PIL Image of the thumbnail
        """
        if self.slide is None:
            raise ValueError("Slide is not open")
        
        try:
            return self.slide.get_thumbnail(max_size)
        except Exception as e:
            self.logger.error(f"Error getting thumbnail: {e}")
            # Create a blank thumbnail as fallback
            return Image.new('RGB', max_size, color=(255, 255, 255))
    
    def read_region(self, 
                   location: Tuple[int, int], 
                   level: int, 
                   size: Tuple[int, int]) -> Image.Image:
        """Read a region from the slide.
        
        Args:
            location: (x, y) tuple giving the top left pixel in the level 0 reference frame
            level: The level number
            size: (width, height) tuple giving the region size
            
        Returns:
            PIL Image of the region
        """
        if self.slide is None:
            raise ValueError("Slide is not open")
        
        # Validate level
        if level < 0 or level >= self.level_count:
            raise ValueError(f"Invalid level: {level}, valid range is 0-{self.level_count-1}")
        
        # Validate location and size
        if location[0] < 0 or location[1] < 0:
            raise ValueError(f"Invalid location: {location}, coordinates must be non-negative")
        
        if size[0] <= 0 or size[1] <= 0:
            raise ValueError(f"Invalid size: {size}, dimensions must be positive")
        
        try:
            region = self.slide.read_region(location, level, size)
            # Convert to RGB (removing alpha channel)
            return region.convert("RGB")
        except Exception as e:
            self.logger.error(f"Error reading region: {e}")
            # Return a blank image of the requested size as fallback
            return Image.new('RGB', size, color=(255, 255, 255))
    
    def get_tile(self, 
                x: int, 
                y: int, 
                width: int, 
                height: int, 
                level: int = 0) -> Image.Image:
        """Get a tile from the slide.
        
        Args:
            x: X-coordinate of the top-left corner of the tile
            y: Y-coordinate of the top-left corner of the tile
            width: Width of the tile
            height: Height of the tile
            level: Magnification level (0 is highest resolution)
            
        Returns:
            PIL Image of the tile
        """
        return self.read_region((x, y), level, (width, height))
    
    def close(self) -> None:
        """Close the slide."""
        if self.slide is not None:
            self.slide.close()
            self.slide = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def __del__(self):
        self.close()


class TileExtractor:
    """Utility for extracting tiles from WSI files."""
    
    def __init__(self, 
                slide_reader: SlideReader, 
                tile_size: int = 1024, 
                overlap: int = 0,
                level: int = 0):
        """Initialize the tile extractor.
        
        Args:
            slide_reader: SlideReader instance
            tile_size: Size of tiles (width and height)
            overlap: Overlap between adjacent tiles in pixels
            level: Magnification level to extract tiles from (0 is highest)
        """
        self.slide_reader = slide_reader
        self.tile_size = tile_size
        self.overlap = overlap
        self.level = level
        
        # Calculate effective tile size (accounting for overlap)
        self.stride = tile_size - overlap
        
        # Get level dimensions
        self.level_width = self.slide_reader.level_dimensions[level][0]
        self.level_height = self.slide_reader.level_dimensions[level][1]
        
        # Calculate number of tiles in each dimension
        self.n_tiles_x = max(1, (self.level_width + self.stride - 1) // self.stride)
        self.n_tiles_y = max(1, (self.level_height + self.stride - 1) // self.stride)
        
        self.logger = get_logger(name="tile_extractor")
        self.logger.info(f"Initialized tile extractor")
        self.logger.info(f"Level dimensions: {self.level_width}x{self.level_height}")
        self.logger.info(f"Tile size: {tile_size}x{tile_size}")
        self.logger.info(f"Overlap: {overlap}")
        self.logger.info(f"Number of tiles: {self.n_tiles_x}x{self.n_tiles_y} = {self.n_tiles_x * self.n_tiles_y}")
    
    def extract_tile(self, tile_x: int, tile_y: int) -> Tuple[Image.Image, Dict]:
        """Extract a specific tile from the slide.
        
        Args:
            tile_x: Tile X index
            tile_y: Tile Y index
            
        Returns:
            Tuple of (tile_image, tile_info)
        """
        if tile_x < 0 or tile_x >= self.n_tiles_x or tile_y < 0 or tile_y >= self.n_tiles_y:
            raise ValueError(f"Tile indices out of range: ({tile_x}, {tile_y})")
        
        # Calculate pixel coordinates for this tile
        x = tile_x * self.stride
        y = tile_y * self.stride
        
        # Adjust width/height for edge tiles
        width = min(self.tile_size, self.level_width - x)
        height = min(self.tile_size, self.level_height - y)
        
        # Read the tile
        # Scale coordinates to level 0 if necessary
        if self.level > 0:
            downsample = self.slide_reader.level_downsamples[self.level]
            level0_x = int(x * downsample)
            level0_y = int(y * downsample)
            tile_image = self.slide_reader.read_region((level0_x, level0_y), self.level, (width, height))
        else:
            tile_image = self.slide_reader.get_tile(x, y, width, height, self.level)
        
        # Create tile info
        tile_info = {
            'index': (tile_x, tile_y),
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'level': self.level
        }
        
        return tile_image, tile_info
    
    def get_all_tiles(self) -> List[Tuple[Image.Image, Dict]]:
        """Extract all tiles from the slide.
        
        Returns:
            List of (tile_image, tile_info) tuples
        """
        tiles = []
        for tile_y in range(self.n_tiles_y):
            for tile_x in range(self.n_tiles_x):
                try:
                    tile_image, tile_info = self.extract_tile(tile_x, tile_y)
                    tiles.append((tile_image, tile_info))
                except Exception as e:
                    self.logger.error(f"Error extracting tile ({tile_x}, {tile_y}): {e}")
                    # Add a placeholder for the failed tile
                    tile_info = {
                        'index': (tile_x, tile_y),
                        'x': tile_x * self.stride,
                        'y': tile_y * self.stride,
                        'width': min(self.tile_size, self.level_width - tile_x * self.stride),
                        'height': min(self.tile_size, self.level_height - tile_y * self.stride),
                        'level': self.level,
                        'error': str(e)
                    }
                    blank_image = Image.new('RGB', (tile_info['width'], tile_info['height']), color=(255, 255, 255))
                    tiles.append((blank_image, tile_info))
        
        return tiles
    
    def get_tile_coordinates(self) -> List[Dict]:
        """Get coordinates for all tiles without extracting them.
        
        Returns:
            List of tile_info dictionaries
        """
        tile_coordinates = []
        for tile_y in range(self.n_tiles_y):
            for tile_x in range(self.n_tiles_x):
                x = tile_x * self.stride
                y = tile_y * self.stride
                width = min(self.tile_size, self.level_width - x)
                height = min(self.tile_size, self.level_height - y)
                
                tile_info = {
                    'index': (tile_x, tile_y),
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height,
                    'level': self.level
                }
                tile_coordinates.append(tile_info)
        
        return tile_coordinates


class TileStitcher:
    """Utility for stitching tiles back into a whole slide image."""
    
    def __init__(self,
                output_width: int,
                output_height: int,
                background_color: Tuple[int, int, int] = (255, 255, 255)):
        """Initialize the tile stitcher.
        
        Args:
            output_width: Width of the output image
            output_height: Height of the output image
            background_color: RGB background color for the output image
        """
        self.output_width = max(1, output_width)
        self.output_height = max(1, output_height)
        self.background_color = background_color
        
        # Create empty output image
        self.output_image = Image.new('RGB', (self.output_width, self.output_height), background_color)
        
        self.logger = get_logger(name="tile_stitcher")
        self.logger.info(f"Initialized tile stitcher")
        self.logger.info(f"Output dimensions: {self.output_width}x{self.output_height}")
    
    def add_tile(self, tile_image: Image.Image, x: int, y: int) -> None:
        """Add a tile to the output image.
        
        Args:
            tile_image: Tile image to add
            x: X-coordinate of the top-left corner of the tile
            y: Y-coordinate of the top-left corner of the tile
        """
        # Ensure coordinates are within bounds
        if x < 0 or y < 0 or x >= self.output_width or y >= self.output_height:
            self.logger.warning(f"Tile coordinates ({x}, {y}) out of bounds, skipping")
            return
        
        # Calculate the actual region to paste
        paste_width = min(tile_image.width, self.output_width - x)
        paste_height = min(tile_image.height, self.output_height - y)
        
        if paste_width <= 0 or paste_height <= 0:
            self.logger.warning(f"Tile dimensions invalid for pasting at ({x}, {y}), skipping")
            return
        
        # Crop tile if needed
        if tile_image.width > paste_width or tile_image.height > paste_height:
            tile_image = tile_image.crop((0, 0, paste_width, paste_height))
        
        # Paste the tile
        self.output_image.paste(tile_image, (x, y))
    
    def get_output_image(self) -> Image.Image:
        """Get the stitched output image.
        
        Returns:
            PIL Image of the stitched output
        """
        return self.output_image.copy()
    
    def save_output_image(self, output_path: str) -> None:
        """Save the stitched output image.
        
        Args:
            output_path: Path to save the output image
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        try:
            self.output_image.save(output_path)
        except Exception as e:
            raise IOError(f"Failed to save stitched image to {output_path}: {e}")