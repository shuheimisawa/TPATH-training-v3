# Update to src/config/model_config.py (we need to create this if it doesn't exist)

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class StainNormalizationConfig:
    """Configuration for stain normalization."""
    
    method: str = "vahadane"  # 'macenko', 'reinhard', or 'vahadane'
    reference_image_path: Optional[str] = None
    params_path: Optional[str] = None
    
    # Vahadane parameters
    n_stains: int = 2
    lambda1: float = 0.1
    lambda2: float = 0.1


@dataclass
class FeatureExtractionConfig:
    """Configuration for feature extraction."""
    
    # Texture features
    use_texture_features: bool = True
    gabor_frequencies: List[float] = field(default_factory=lambda: [0.1, 0.25, 0.4])
    gabor_orientations: List[float] = field(default_factory=lambda: [0, 0.25, 0.5, 0.75])
    lbp_radius: int = 3
    lbp_points: int = 24
    
    # Color features
    use_color_features: bool = True
    color_bins: int = 32
    
    # Morphological features
    use_morphological_features: bool = True


@dataclass
class DetectionConfig:
    """Configuration for glomeruli detection (stage 1)."""
    
    # Cascade Mask R-CNN parameters
    num_stages: int = 3
    backbone: str = "resnet50"
    fpn: bool = True
    anchor_sizes: List[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    aspect_ratios: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    
    # Detection thresholds
    score_threshold: float = 0.7
    nms_threshold: float = 0.5
    
    # Training parameters
    roi_batch_size: int = 128
    positive_fraction: float = 0.25
    bbox_reg_weights: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.2, 0.2])


@dataclass
class ClassificationConfig:
    """Configuration for glomeruli classification (stage 2)."""
    
    # Model parameters
    num_classes: int = 4  # Normal, Sclerotic, Partially_sclerotic, Uncertain
    feature_dim: int = 256
    
    # Input parameters
    patch_size: int = 256
    in_channels: int = 3
    
    # Classification thresholds
    confidence_threshold: float = 0.8


@dataclass
class ModelConfig:
    """Model configuration."""
    
    # Pipeline mode
    two_stage: bool = True  # Whether to use the two-stage pipeline
    
    # Preprocessing
    normalization: StainNormalizationConfig = field(default_factory=StainNormalizationConfig)
    feature_extraction: FeatureExtractionConfig = field(default_factory=FeatureExtractionConfig)
    
    # Stage 1: Glomeruli Detection
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    
    # Stage 2: Glomeruli Classification (only used if two_stage=True)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)