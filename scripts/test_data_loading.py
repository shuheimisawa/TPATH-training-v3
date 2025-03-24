# scripts/test_data_loading.py
import os
import sys
import torch
import argparse
import numpy as np
from PIL import Image
import json

# Add the parent directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def parse_args():
    parser = argparse.ArgumentParser(description='Test data loading')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--output-dir', type=str, default='data_test_output',
                        help='Path to output directory')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check directories
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    test_dir = os.path.join(args.data_dir, 'test')
    
    print(f"Checking data directories:")
    for name, dir_path in [('Training', train_dir), ('Validation', val_dir), ('Test', test_dir)]:
        if os.path.exists(dir_path):
            print(f"✓ {name} directory exists: {dir_path}")
            
            # Count image files
            png_files = [f for f in os.listdir(dir_path) if f.endswith('.png')]
            regular_images = [f for f in png_files if not f.endswith('_vis.png')]
            vis_images = [f for f in png_files if f.endswith('_vis.png')]
            
            print(f"  • Found {len(regular_images)} regular images and {len(vis_images)} visualization images")
            
            # Check if annotation file exists
            annotation_file = os.path.join(dir_path, f"{os.path.basename(dir_path)}_annotations.json")
            if os.path.exists(annotation_file):
                print(f"✓ Annotation file exists: {annotation_file}")
                
                # Check annotation format
                with open(annotation_file, 'r') as f:
                    annotations = json.load(f)
                
                print(f"  • Annotation file contains data for {len(annotations)} images")
                
                # Analyze the first entry
                first_key = next(iter(annotations))
                first_entry = annotations[first_key]
                
                print(f"  • First entry key: {first_key}")
                print(f"  • First entry type: {type(first_entry).__name__}")
                
                if isinstance(first_entry, list):
                    print(f"  • List of annotations with {len(first_entry)} items")
                    if first_entry:
                        print(f"  • First annotation keys: {list(first_entry[0].keys())}")
                elif isinstance(first_entry, dict):
                    print(f"  • Dictionary with keys: {list(first_entry.keys())}")
                    if 'annotations' in first_entry:
                        print(f"  • Contains {len(first_entry['annotations'])} annotation items")
                
                # Check a few images to see if they match with annotations
                sample_count = min(3, len(regular_images))
                print(f"\nChecking {sample_count} sample images:")
                
                for i in range(sample_count):
                    img_filename = regular_images[i]
                    img_id = os.path.splitext(img_filename)[0]
                    
                    print(f"  Image {i+1}: {img_filename}")
                    
                    # Check if this image is in annotations
                    if img_id in annotations:
                        print(f"  ✓ Found in annotations")
                        
                        # Load and save a sample
                        img_path = os.path.join(dir_path, img_filename)
                        img = Image.open(img_path)
                        img.save(os.path.join(args.output_dir, f"{name.lower()}_{i+1}.png"))
                        
                        # Show annotation info
                        anno = annotations[img_id]
                        if isinstance(anno, list):
                            print(f"    • Contains {len(anno)} annotations")
                        elif isinstance(anno, dict) and 'annotations' in anno:
                            print(f"    • Contains {len(anno['annotations'])} annotations")
                    else:
                        print(f"  ✗ Not found in annotations")
                
                print("\n")
            else:
                print(f"✗ Annotation file missing: {annotation_file}")
        else:
            print(f"✗ {name} directory does not exist: {dir_path}")
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main()