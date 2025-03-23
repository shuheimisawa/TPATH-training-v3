import json
import os

def check_annotations(file_path):
    print(f"Checking annotations file: {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        print(f"Data type: {type(data)}")
        print(f"Data length: {len(data)}")
        
        if isinstance(data, dict):
            print("Sample keys:", list(data.keys())[:5])
            sample_item = next(iter(data.values()))
            print("Sample item keys:", list(sample_item.keys()) if isinstance(sample_item, dict) else "Not a dict")
        elif isinstance(data, list):
            print("Sample item:", data[0] if data else "Empty list")
            if data:
                print("Sample item keys:", list(data[0].keys()) if isinstance(data[0], dict) else "Not a dict")
        
    except Exception as e:
        print(f"Error reading annotations: {e}")

if __name__ == "__main__":
    check_annotations("data_test/processed/train/train_annotations.json") 