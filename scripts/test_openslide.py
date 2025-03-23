import openslide
print("OpenSlide import successful")
print("OpenSlide version:", openslide.__version__)
# Optional: list available formats
print("Available formats:", openslide.OpenSlide.detect_format)