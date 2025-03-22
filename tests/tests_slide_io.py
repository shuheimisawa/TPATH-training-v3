import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.slide_io import SlideReader, TileExtractor, TileStitcher


class MockOpenSlide:
    """Mock class for OpenSlide."""
    
    def __init__(self, width, height, level_count, level_dimensions, level_downsamples):
        self.dimensions = (width, height)
        self.level_count = level_count
        self.level_dimensions = level_dimensions
        self.level_downsamples = level_downsamples
    
    def get_thumbnail(self, size):
        # Return a dummy thumbnail
        return Image.new('RGB', size, color=(255, 255, 255))
    
    def read_region(self, location, level, size):
        # Return a dummy region
        return Image.new('RGBA', size, color=(255, 255, 255, 255))
    
    def close(self):
        pass


class TestSlideIO(unittest.TestCase):
    """Test cases for slide IO utilities."""
    
    @patch('openslide.OpenSlide')
    @patch('openslide.is_openslide')
    def setUp(self, mock_is_openslide, mock_openslide_class):
        """Set up test case."""
        # Mock openslide.is_openslide to return True
        mock_is_openslide.return_value = True
        
        # Set up mock slide parameters
        self.width = 10000
        self.height = 8000
        self.level_count = 3
        self.level_dimensions = [(10000, 8000), (5000, 4000), (2500, 2000)]
        self.level_downsamples = [1.0, 2.0, 4.0]
        
        # Create mock OpenSlide instance
        self.mock_slide = MockOpenSlide(
            self.width,
            self.height,
            self.level_count,
            self.level_dimensions,
            self.level_downsamples
        )
        
        # Set up mock OpenSlide class
        mock_openslide_class.return_value = self.mock_slide
        
        # Initialize SlideReader with mock
        self.slide_path = "dummy_slide.svs"
        self.slide_reader = SlideReader(self.slide_path)
    
    def test_slide_reader_init(self):
        """Test SlideReader initialization."""
        self.assertEqual(self.slide_reader.width, self.width)
        self.assertEqual(self.slide_reader.height, self.height)
        self.assertEqual(self.slide_reader.level_count, self.level_count)
        self.assertEqual(self.slide_reader.level_dimensions, self.level_dimensions)
        self.assertEqual(self.slide_reader.level_downsamples, self.level_downsamples)
    
    def test_get_slide_thumbnail(self):
        """Test getting slide thumbnail."""
        thumb_size = (1024, 768)
        thumbnail = self.slide_reader.get_slide_thumbnail(thumb_size)
        
        self.assertIsInstance(thumbnail, Image.Image)
        self.assertEqual(thumbnail.size, thumb_size)
    
    def test_read_region(self):
        """Test reading a region from the slide."""
        location = (500, 500)
        level = 0
        size = (256, 256)
        
        region = self.slide_reader.read_region(location, level, size)
        
        self.assertIsInstance(region, Image.Image)
        self.assertEqual(region.size, size)
        self.assertEqual(region.mode, 'RGB')  # Should be converted to RGB
    
    def test_get_tile(self):
        """Test getting a tile from the slide."""
        x, y = 1000, 1000
        width, height = 512, 512
        level = 0
        
        tile = self.slide_reader.get_tile(x, y, width, height, level)
        
        self.assertIsInstance(tile, Image.Image)
        self.assertEqual(tile.size, (width, height))
        self.assertEqual(tile.mode, 'RGB')
    
    def test_context_manager(self):
        """Test using SlideReader as a context manager."""
        with self.slide_reader as sr:
            self.assertEqual(sr.width, self.width)
            self.assertEqual(sr.height, self.height)
    
    def test_tile_extractor_init(self):
        """Test TileExtractor initialization."""
        tile_size = 1024
        overlap = 256
        level = 0
        
        tile_extractor = TileExtractor(
            self.slide_reader,
            tile_size=tile_size,
            overlap=overlap,
            level=level
        )
        
        self.assertEqual(tile_extractor.tile_size, tile_size)
        self.assertEqual(tile_extractor.overlap, overlap)
        self.assertEqual(tile_extractor.level, level)
        self.assertEqual(tile_extractor.stride, tile_size - overlap)
        
        expected_n_tiles_x = (self.width + tile_extractor.stride - 1) // tile_extractor.stride
        expected_n_tiles_y = (self.height + tile_extractor.stride - 1) // tile_extractor.stride
        
        self.assertEqual(tile_extractor.n_tiles_x, expected_n_tiles_x)
        self.assertEqual(tile_extractor.n_tiles_y, expected_n_tiles_y)
    
    def test_extract_tile(self):
        """Test extracting a specific tile."""
        tile_extractor = TileExtractor(self.slide_reader, tile_size=1024, overlap=0, level=0)
        
        tile_image, tile_info = tile_extractor.extract_tile(0, 0)
        
        self.assertIsInstance(tile_image, Image.Image)
        self.assertEqual(tile_info['index'], (0, 0))
        self.assertEqual(tile_info['x'], 0)
        self.assertEqual(tile_info['y'], 0)
        self.assertEqual(tile_info['width'], 1024)
        self.assertEqual(tile_info['height'], 1024)
        self.assertEqual(tile_info['level'], 0)
    
    def test_extract_tile_with_overlap(self):
        """Test extracting a tile with overlap."""
        tile_extractor = TileExtractor(self.slide_reader, tile_size=1024, overlap=256, level=0)
        
        # Extract first tile
        _, first_tile_info = tile_extractor.extract_tile(0, 0)
        
        # Extract second tile
        _, second_tile_info = tile_extractor.extract_tile(1, 0)
        
        # Check that the second tile starts at stride distance from the first
        self.assertEqual(second_tile_info['x'], first_tile_info['x'] + tile_extractor.stride)
        self.assertEqual(second_tile_info['y'], first_tile_info['y'])
    
    def test_extract_tile_out_of_bounds(self):
        """Test extracting a tile with invalid indices."""
        tile_extractor = TileExtractor(self.slide_reader, tile_size=1024, overlap=0, level=0)
        
        with self.assertRaises(ValueError):
            tile_extractor.extract_tile(-1, 0)
        
        with self.assertRaises(ValueError):
            tile_extractor.extract_tile(0, -1)
        
        with self.assertRaises(ValueError):
            tile_extractor.extract_tile(tile_extractor.n_tiles_x, 0)
        
        with self.assertRaises(ValueError):
            tile_extractor.extract_tile(0, tile_extractor.n_tiles_y)
    
    def test_get_all_tiles(self):
        """Test getting all tiles."""
        # Use a small tile size to limit the number of tiles for testing
        tile_extractor = TileExtractor(self.slide_reader, tile_size=1024, overlap=0, level=2)
        
        tiles = tile_extractor.get_all_tiles()
        
        self.assertEqual(len(tiles), tile_extractor.n_tiles_x * tile_extractor.n_tiles_y)
        
        for tile_image, tile_info in tiles:
            self.assertIsInstance(tile_image, Image.Image)
            self.assertIsInstance(tile_info, dict)
            self.assertIn('index', tile_info)
            self.assertIn('x', tile_info)
            self.assertIn('y', tile_info)
            self.assertIn('width', tile_info)
            self.assertIn('height', tile_info)
            self.assertIn('level', tile_info)
    
    def test_get_tile_coordinates(self):
        """Test getting tile coordinates without extracting them."""
        tile_extractor = TileExtractor(self.slide_reader, tile_size=1024, overlap=0, level=2)
        
        tile_coordinates = tile_extractor.get_tile_coordinates()
        
        self.assertEqual(len(tile_coordinates), tile_extractor.n_tiles_x * tile_extractor.n_tiles_y)
        
        for tile_info in tile_coordinates:
            self.assertIsInstance(tile_info, dict)
            self.assertIn('index', tile_info)
            self.assertIn('x', tile_info)
            self.assertIn('y', tile_info)
            self.assertIn('width', tile_info)
            self.assertIn('height', tile_info)
            self.assertIn('level', tile_info)
    
    def test_tile_stitcher_init(self):
        """Test TileStitcher initialization."""
        output_width = 1000
        output_height = 800
        background_color = (255, 255, 255)
        
        stitcher = TileStitcher(output_width, output_height, background_color)
        
        self.assertEqual(stitcher.output_width, output_width)
        self.assertEqual(stitcher.output_height, output_height)
        self.assertEqual(stitcher.background_color, background_color)
        
        # Check that output image was created correctly
        self.assertIsInstance(stitcher.output_image, Image.Image)
        self.assertEqual(stitcher.output_image.size, (output_width, output_height))
        
        # Sample a few pixels to verify background color
        img_array = np.array(stitcher.output_image)
        self.assertTrue(np.all(img_array[0, 0] == background_color))
        self.assertTrue(np.all(img_array[output_height//2, output_width//2] == background_color))
    
    def test_add_tile(self):
        """Test adding a tile to the output image."""
        output_width = 1000
        output_height = 800
        stitcher = TileStitcher(output_width, output_height)
        
        # Create a sample tile with a colored rectangle
        tile_size = (200, 200)
        tile_color = (255, 0, 0)  # Red
        tile_image = Image.new('RGB', tile_size, color=tile_color)
        
        # Add tile at a specific position
        x, y = 100, 100
        stitcher.add_tile(tile_image, x, y)
        
        # Check that the tile was added correctly
        output_array = np.array(stitcher.output_image)
        
        # Check background color in untouched area
        self.assertTrue(np.all(output_array[0, 0] == (255, 255, 255)))
        
        # Check tile color in the tile area
        self.assertTrue(np.all(output_array[y+tile_size[1]//2, x+tile_size[0]//2] == tile_color))
    
    def test_get_output_image(self):
        """Test getting the stitched output image."""
        output_width = 1000
        output_height = 800
        stitcher = TileStitcher(output_width, output_height)
        
        output_image = stitcher.get_output_image()
        
        self.assertIsInstance(output_image, Image.Image)
        self.assertEqual(output_image.size, (output_width, output_height))
    
    def test_save_output_image(self, tmp_path=''):
        """Test saving the output image."""
        # Use a temporary file path
        if not tmp_path:
            tmp_path = 'temp_output.png'
        
        output_width = 100
        output_height = 80
        stitcher = TileStitcher(output_width, output_height)
        
        # Add a colored tile
        tile_image = Image.new('RGB', (50, 50), color=(255, 0, 0))
        stitcher.add_tile(tile_image, 25, 15)
        
        # Save output image
        stitcher.save_output_image(tmp_path)
        
        # Check that the file was created
        self.assertTrue(os.path.exists(tmp_path))
        
        # Check that the saved image can be loaded
        saved_image = Image.open(tmp_path)
        self.assertEqual(saved_image.size, (output_width, output_height))
        
        # Clean up
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == '__main__':
    unittest.main()