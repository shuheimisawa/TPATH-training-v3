# scripts/inference_enhanced.py
import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import cv2
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import time
import traceback

# Add the parent directory to sys.path to resolve imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config.model_config import ModelConfig
from src.models.cascade_mask_rcnn import CascadeMaskRCNN
from src.models.components.enhanced_mask_head import MaskRCNNHeadWithBoundary
from src.utils.io import load_yaml, load_image, save_image
from src.utils.logger import get_logger
from src.utils.directml_adapter import get_dml_device, is_available, empty_cache
from src.utils.stain_normalization import StainNormalizationTransform
from src.utils.slide_io import SlideReader, TileExtractor
from src.evaluation.visualization import visualize_prediction
from src.data.transforms import get_val_transforms