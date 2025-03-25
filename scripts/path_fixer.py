import os
import json
import argparse

def fix_annotation_paths(annotations_file, output_file=None):
    """
    Fix the annotation keys by converting full paths to just filenames.
    
    Args:
        annotations_file: Path to the original annotations JSON file
        output_file: Path to save the fixed annotations (if None, will use original file with '_fixed' suffix)
    
    Returns:
        Path to the fixed annotations file
    """
    if output_file is None:
        base, ext = os.path.splitext(annotations_file)
        output_file = f"{base}_fixed{ext}"
    
    print(f"Reading annotations from {annotations_file}")
    with open(annotations_file, 'r') as f:
        annotations = json.load(f)
    
    # Create a new dictionary with fixed keys
    fixed_annotations = {}
    for key, value in annotations.items():
        # Extract just the filename from the path
        filename = os.path.basename(key)
        fixed_annotations[filename] = value
    
    print(f"Original annotations had {len(annotations)} entries")
    print(f"Fixed annotations have {len(fixed_annotations)} entries")
    
    # Save the fixed annotations
    with open(output_file, 'w') as f:
        json.dump(fixed_annotations, f, indent=2)
    
    print(f"Fixed annotations saved to {output_file}")
    return output_file

def main():
    parser = argparse.ArgumentParser(description="Fix annotation file paths")
    parser.add_argument("--annotations", required=True, help="Path to annotations JSON file")
    parser.add_argument("--output", help="Path to save fixed annotations (default: input_fixed.json)")
    args = parser.parse_args()
    
    fix_annotation_paths(args.annotations, args.output)

if __name__ == "__main__":
    main()