import os
import sys
import openslide
import numpy as np
from pathlib import Path

def test_svs_files():
    print("OpenSlide version:", openslide.__version__)
    
    # Path to SVS files
    slide_dir = "data_test/raw/slides"
    print(f"\nLooking for SVS files in: {slide_dir}")
    
    for file in sorted(os.listdir(slide_dir)):
        if file.endswith('.svs'):
            slide_path = os.path.join(slide_dir, file)
            print(f"\nTesting slide: {file}")
            try:
                # Open the slide
                slide = openslide.OpenSlide(slide_path)
                
                # Print basic information
                print("Successfully opened slide!")
                print("Dimensions:", slide.dimensions)
                print("Level count:", slide.level_count)
                print("Level dimensions:", slide.level_dimensions)
                print("Level downsamples:", slide.level_downsamples)
                print("Properties:", dict(slide.properties))
                
                # Try reading a small region (1000x1000 from the center)
                w, h = slide.dimensions
                center_x = w // 2 - 500
                center_y = h // 2 - 500
                region = slide.read_region((center_x, center_y), 0, (1000, 1000))
                print("Successfully read region from center")
                
                # Convert to numpy array to verify data
                region_np = np.array(region)
                print("Region shape:", region_np.shape)
                
                slide.close()
                print("Slide closed successfully")
                
            except Exception as e:
                print("Error processing slide:", str(e))

if __name__ == "__main__":
    test_svs_files() 