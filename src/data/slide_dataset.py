import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional, Union, Callable

from ..utils.slide_io import SlideReader, TileExtractor


class SlideTileDataset(Dataset):
    """Dataset for slide tiles."""
    
    def __init__(
        self,
        slide_path: str,
        tile_size: int = 1024,
        overlap: int = 256,
        level: int = 0,
        transform: Optional[Callable] = None,
        return_coordinates: bool = False,
        filter_background: bool = True,
        background_threshold: int = 220
    ):
        """Initialize the dataset.
        
        Args:
            slide_path: Path to the slide file
            tile_size: Size of tiles
            overlap: Overlap between adjacent tiles
            level: Magnification level
            transform: Optional transform to apply to tiles
            return_coordinates: Whether to return tile coordinates
            filter_background: Whether to filter out background tiles
            background_threshold: Threshold for background detection
        """
        self.slide_path = slide_path
        self.tile_size = tile_size
        self.overlap = overlap
        self.level = level
        self.transform = transform
        self.return_coordinates = return_coordinates
        self.filter_background = filter_background
        self.background_threshold = background_threshold
        
        # Open slide
        self.slide_reader = SlideReader(slide_path)
        
        # Create tile extractor
        self.tile_extractor = TileExtractor(
            self.slide_reader,
            tile_size=tile_size,
            overlap=overlap,
            level=level
        )
        
        # Get tile coordinates
        all_tile_coordinates = self.tile_extractor.get_tile_coordinates()
        
        # Filter background tiles if needed
        if filter_background:
            self.tile_coordinates = []
            for tile_info in all_tile_coordinates:
                # Extract tile
                tile_image, _ = self.tile_extractor.extract_tile(
                    tile_info['index'][0], tile_info['index'][1]
                )
                
                # Calculate mean pixel value for background detection
                tile_np = np.array(tile_image)
                mean_value = np.mean(tile_np)
                
                if mean_value <= background_threshold:
                    self.tile_coordinates.append(tile_info)
        else:
            self.tile_coordinates = all_tile_coordinates
    
    def __len__(self) -> int:
        """Get the number of tiles."""
        return len(self.tile_coordinates)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a tile.
        
        Args:
            idx: Index of the tile
            
        Returns:
            Dictionary with tile data
        """
        # Get tile info
        tile_info = self.tile_coordinates[idx]
        
        # Extract tile
        tile_image, _ = self.tile_extractor.extract_tile(
            tile_info['index'][0], tile_info['index'][1]
        )
        
        # Apply transform if available
        if self.transform:
            # Convert PIL Image to numpy array
            tile_np = np.array(tile_image)
            # Apply transform
            transformed = self.transform(image=tile_np)
            # Get transformed image
            image = transformed['image']
        else:
            image = np.array(tile_image)
        
        # Create result
        result = {'image': image}
        
        # Add coordinates if needed
        if self.return_coordinates:
            result.update({
                'x': torch.tensor(tile_info['x']),
                'y': torch.tensor(tile_info['y']),
                'width': torch.tensor(tile_info['width']),
                'height': torch.tensor(tile_info['height']),
                'level': torch.tensor(tile_info['level']),
                'index': torch.tensor(tile_info['index'])
            })
        
        return result
    
    def __del__(self):
        """Close resources when the dataset is deleted."""
        if hasattr(self, 'slide_reader'):
            self.slide_reader.close()