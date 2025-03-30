# scripts/compare_normalizers.py

import os
import numpy as np
import cv2
import argparse
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.utils.stain_normalization import VahadaneNormalizer
from src.utils.io import load_image, save_image
from src.utils.logger import get_logger


def compare_normalizers(image_path, reference_path, output_dir):
    """Compare different stain normalization methods."""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images
    image = load_image(image_path)
    reference = load_image(reference_path)
    
    # Create normalizers
    normalizers = {
        'Original': None,
        'Reference': None,
        'Vahadane': VahadaneNormalizer()
    }
    
    # Fit normalizers to reference image
    for name, normalizer in normalizers.items():
        if normalizer is not None:
            normalizer.fit(reference)
    
    # Apply normalization
    results = {
        'Original': image,
        'Reference': reference
    }
    
    for name, normalizer in normalizers.items():
        if normalizer is not None:
            results[name] = normalizer.transform(image)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes = axes.flatten()
    
    for i, (name, result) in enumerate(results.items()):
        if i < len(axes):
            axes[i].imshow(result)
            axes[i].set_title(name)
            axes[i].axis('off')
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison.png'), dpi=300)
    plt.close()
    
    # Save individual images
    for name, result in results.items():
        save_image(result, os.path.join(output_dir, f'{name.lower()}.png'))
    
    return results


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Compare stain normalization methods')
    
    parser.add_argument('--image-path', type=str, required=True,
                        help='Path to input image')
    parser.add_argument('--reference-path', type=str, required=True,
                        help='Path to reference image')
    parser.add_argument('--output-dir', type=str, default='experiments/normalization_comparison',
                        help='Path to output directory')
    
    args = parser.parse_args()
    
    # Compare normalizers
    compare_normalizers(
        image_path=args.image_path,
        reference_path=args.reference_path,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()