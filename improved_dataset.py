import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class GlomeruliDataset(Dataset):
    """Dataset for glomeruli segmentation with robust path handling."""
    
    def __init__(self, data_dir, mode='train', transform=None):
        """Initialize GlomeruliDataset.
        
        Args:
            data_dir (str): Path to data directory
            mode (str): 'train', 'val', or 'test'
            transform: Optional transform to apply to images
        """
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        
        # Load annotations
        annotations_file = os.path.join(data_dir, f"{mode}_annotations.json")
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Get list of image files
        self.image_files = []
        self.image_ids = []
        
        # First, try loading with full path keys
        for filename in os.listdir(data_dir):
            if filename.endswith('.png') and not filename.endswith('_vis.png'):
                # Create multiple potential keys to check
                image_path = os.path.join(data_dir, filename)
                basename = os.path.basename(image_path)
                rel_path = os.path.relpath(image_path)
                alt_path = rel_path.replace('\\', '/')
                
                # Check if any key format exists in annotations
                if image_path in self.annotations:
                    self.image_files.append(image_path)
                    self.image_ids.append(image_path)
                elif rel_path in self.annotations:
                    self.image_files.append(image_path)
                    self.image_ids.append(rel_path)
                elif alt_path in self.annotations:
                    self.image_files.append(image_path)
                    self.image_ids.append(alt_path)
                elif basename in self.annotations:
                    self.image_files.append(image_path)
                    self.image_ids.append(basename)
                else:
                    # Try fallback: check if the annotation key ends with this filename
                    found = False
                    for anno_key in self.annotations.keys():
                        if anno_key.endswith(basename):
                            self.image_files.append(image_path)
                            self.image_ids.append(anno_key)
                            found = True
                            break
                    
                    if not found:
                        print(f"Warning: No annotation found for {filename}")
        
        print(f"Loaded {len(self.image_files)} images with annotations")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image_id = self.image_ids[idx]
        
        # Open image with PIL
        image = Image.open(image_path).convert('RGB')
        
        # Convert to numpy array
        image_np = np.array(image)
        
        # Get annotations for this image
        annotations = self.annotations[image_id]
        
        # Prepare target
        boxes = []
        labels = []
        masks = []
        
        # Process each annotation
        for anno in annotations:
            bbox = anno.get('bbox', [0, 0, 1, 1])
            category = anno.get('category', 'background')
            segmentation = anno.get('segmentation', [])
            
            # Convert category to label index
            # This mapping should be consistent with your model's class indices
            if category == 'Normal':
                label = 1
            elif category == 'Sclerotic':
                label = 2
            elif category == 'Partially_sclerotic':
                label = 3
            elif category == 'Uncertain':
                label = 4
            else:
                label = 0  # background
            
            # Add to lists
            boxes.append(bbox)
            labels.append(label)
            
            # Create mask from segmentation
            mask = np.zeros((image_np.shape[0], image_np.shape[1]), dtype=np.uint8)
            
            # Fill mask using segmentation polygons
            for polygon in segmentation:
                if len(polygon) >= 6:  # At least 3 points (x,y pairs)
                    # Convert flat list to list of points
                    points = []
                    for i in range(0, len(polygon), 2):
                        if i + 1 < len(polygon):
                            points.append([polygon[i], polygon[i + 1]])
                    
                    # Convert to numpy array
                    points = np.array(points, dtype=np.int32)
                    
                    # Fill polygon
                    mask = cv2.fillPoly(mask, [points], 1)
            
            masks.append(mask)
        
        # Convert to torch tensors
        if boxes:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            masks = torch.tensor(masks, dtype=torch.uint8)
        else:
            # Empty annotations
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            masks = torch.zeros((0, image_np.shape[0], image_np.shape[1]), dtype=torch.uint8)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([idx]),
            'area': (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) if len(boxes) > 0 else torch.zeros(0),
            'iscrowd': torch.zeros((len(boxes),), dtype=torch.int64)
        }
        
        # Apply transforms
        if self.transform is not None:
            image_np, target = self.transform(image_np, target)
        
        return {'image': image_np, 'target': target}