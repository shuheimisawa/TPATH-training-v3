from PIL import Image
import os

# Disable PIL's size limit
Image.MAX_IMAGE_PIXELS = None

def check_tiff_file(file_path):
    try:
        with Image.open(file_path) as img:
            print(f"\nFile: {os.path.basename(file_path)}")
            print(f"Format: {img.format}")
            print(f"Mode: {img.mode}")
            print(f"Size: {img.size}")
            print(f"Info: {img.info}")
            
            # Try to read a small region to verify the file is readable
            try:
                region = img.crop((0, 0, min(1024, img.size[0]), min(1024, img.size[1])))
                print("Successfully read a region of the image")
                region.close()
            except Exception as e:
                print(f"Error reading image region: {e}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

def main():
    slide_dir = "data_test/raw/slides_tiff"
    if os.path.exists(slide_dir):
        for file in os.listdir(slide_dir):
            if file.endswith(('.tiff', '.tif')):
                file_path = os.path.join(slide_dir, file)
                check_tiff_file(file_path)
    else:
        print(f"Directory {slide_dir} not found")

if __name__ == "__main__":
    main() 