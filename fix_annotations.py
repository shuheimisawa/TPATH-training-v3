import os
import json

# Define the annotation files
annotation_files = [
    r'C:\Users\misaw\OneDrive\Shuhei\OneDrive\Desktop\python_work\TPATH Training\glomeruli-training\data_test\processed\train\train_annotations.json',
    r'C:\Users\misaw\OneDrive\Shuhei\OneDrive\Desktop\python_work\TPATH Training\glomeruli-training\data_test\processed\val\val_annotations.json',
    r'C:\Users\misaw\OneDrive\Shuhei\OneDrive\Desktop\python_work\TPATH Training\glomeruli-training\data_test\processed\test\test_annotations.json'
]

# Process each file
for annotations_file in annotation_files:
    # Define output file
    output_file = annotations_file.replace('.json', '_fixed.json')
    
    print(f'Processing: {annotations_file}')
    
    try:
        # Load annotations
        with open(annotations_file, 'r') as f:
            annotations = json.load(f)
        
        # Create a new dictionary with fixed keys
        fixed_annotations = {}
        for key, value in annotations.items():
            # Extract just the filename from the path
            filename = os.path.basename(key)
            fixed_annotations[filename] = value
        
        print(f'Original annotations had {len(annotations)} entries')
        print(f'Fixed annotations have {len(fixed_annotations)} entries')
        
        # Save the fixed annotations
        with open(output_file, 'w') as f:
            json.dump(fixed_annotations, f, indent=2)
        
        print(f'Fixed annotations saved to {output_file}')
        
    except Exception as e:
        print(f'Error processing {annotations_file}: {e}')