import os
import sys
from pathlib import Path

# Try to find OpenSlide DLL directory
try:
    import openslide_bin
    dll_paths = [
        str(Path(openslide_bin.__file__).parent / "dll"),
        str(Path(sys.prefix) / "Library" / "bin"),
        os.environ.get("PATH", "").split(os.pathsep)
    ]
    
    for path in dll_paths:
        if isinstance(path, list):
            for p in path:
                if os.path.exists(p):
                    try:
                        os.add_dll_directory(p)
                    except Exception:
                        pass
        elif os.path.exists(path):
            try:
                os.add_dll_directory(path)
            except Exception:
                pass
except Exception as e:
    print(f"Warning: Could not add DLL directories: {e}")

# Now import OpenSlide
try:
    import openslide
    print("Successfully imported OpenSlide!")
except Exception as e:
    print(f"Error importing OpenSlide: {e}")
    sys.exit(1)

def test_openslide():
    print("\nOpenSlide version:", openslide.__version__)
    
    # Try to list available slide files
    slide_dir = "data_test/raw/slides_tiff"
    if os.path.exists(slide_dir):
        print("\nLooking for slide files in:", slide_dir)
        files_found = False
        for file in os.listdir(slide_dir):
            if file.endswith(('.tiff', '.tif', '.svs', '.ndpi')):
                files_found = True
                slide_path = os.path.join(slide_dir, file)
                print(f"\nTesting slide: {file}")
                try:
                    slide = openslide.OpenSlide(slide_path)
                    print("Successfully opened slide!")
                    print("Dimensions:", slide.dimensions)
                    print("Level count:", slide.level_count)
                    print("Level dimensions:", slide.level_dimensions)
                    slide.close()
                except Exception as e:
                    print("Error opening slide:", str(e))
        if not files_found:
            print("No slide files found in the directory")
    else:
        print(f"Directory {slide_dir} not found")

if __name__ == "__main__":
    test_openslide() 