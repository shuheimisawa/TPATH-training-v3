import os
import json
import yaml
import cv2
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Union, Optional


def load_image(image_path: str) -> np.ndarray:
    """Load an image from path.
    
    Args:
        image_path: Path to the image
        
    Returns:
        Image as numpy array
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image


def save_image(image: np.ndarray, save_path: str) -> None:
    """Save an image to path.
    
    Args:
        image: Image as numpy array
        save_path: Path to save the image
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert RGB to BGR for OpenCV
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Save image
    cv2.imwrite(save_path, image_bgr)


def load_json(file_path: str) -> Dict:
    """Load a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary with loaded data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found at {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data


def save_json(data: Dict, file_path: str) -> None:
    """Save data to a JSON file.
    
    Args:
        data: Data to save
        file_path: Path to save the file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_yaml(file_path: str) -> Dict:
    """Load a YAML file.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Dictionary with loaded data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"YAML file not found at {file_path}")
    
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return data


def save_yaml(data: Dict, file_path: str) -> None:
    """Save data to a YAML file.
    
    Args:
        data: Data to save
        file_path: Path to save the file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        yaml.safe_dump(data, f, sort_keys=False)
