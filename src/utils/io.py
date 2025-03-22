import os
import json
import yaml
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Union, Optional, Tuple


def load_image(image_path: str) -> np.ndarray:
    """Load an image from path.
    
    Args:
        image_path: Path to the image
        
    Returns:
        Image as numpy array
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    try:
        # Try OpenCV first
        image = cv2.imread(image_path)
        if image is None:
            # Fallback to PIL
            image = np.array(Image.open(image_path))
            if len(image.shape) == 3 and image.shape[2] == 3:
                # If image is already RGB, just return it
                return image
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # Convert RGBA to RGB
                return image[:, :, :3]
            elif len(image.shape) == 2:
                # Grayscale to RGB
                return np.stack((image,) * 3, axis=-1)
        else:
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
    except Exception as e:
        raise IOError(f"Failed to load image {image_path}: {e}")


def save_image(image: np.ndarray, save_path: str) -> None:
    """Save an image to path.
    
    Args:
        image: Image as numpy array
        save_path: Path to save the image
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        # Check image dimensions
        if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
            # Grayscale image
            cv2.imwrite(save_path, image)
        else:
            # Convert RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Save image
            cv2.imwrite(save_path, image_bgr)
    except Exception as e:
        raise IOError(f"Failed to save image {save_path}: {e}")


def load_json(file_path: str) -> Dict:
    """Load a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary with loaded data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found at {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {file_path}: {e}")
    except Exception as e:
        raise IOError(f"Failed to load JSON {file_path}: {e}")


def save_json(data: Dict, file_path: str) -> None:
    """Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        raise IOError(f"Failed to save JSON {file_path}: {e}")


def load_yaml(file_path: str) -> Dict:
    """Load a YAML file.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Dictionary with loaded data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YAML file not found at {file_path}")
    
    try:
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return data
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {file_path}: {e}")
    except Exception as e:
        raise IOError(f"Failed to load YAML {file_path}: {e}")


def save_yaml(data: Dict, file_path: str) -> None:
    """Save data to a YAML file.
    
    Args:
        data: Data to save
        file_path: Path to save the file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        with open(file_path, 'w') as f:
            yaml.safe_dump(data, f, sort_keys=False)
    except Exception as e:
        raise IOError(f"Failed to save YAML {file_path}: {e}")