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
class BackboneConfig:
    """Configuration for the backbone network."""
    
    name: str = "resnet50"
    pretrained: bool = True
    out_channels: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    freeze_stages: int = 1  # Number of stages to freeze during training
    norm_eval: bool = True  # Whether to set BN layers to eval mode
    out_indices: Tuple[int, ...] = (0, 1, 2, 3)  # Output stages
    
@dataclass
class FPNConfig:
    """Configuration for Feature Pyramid Network."""
    
    in_channels: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    out_channels: int = 256
    num_outs: int = 5
    add_extra_convs: bool = True
    extra_convs_on_inputs: bool = True
    num_blocks: int = 3
    attention_type: str = "none"

@dataclass
class CascadeConfig:
    """Configuration for cascade stages."""
    
    num_stages: int = 3
    stage_loss_weights: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])
    iou_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.6, 0.7])

@dataclass
class MaskConfig:
    """Configuration for mask head."""
    
    roi_size: Tuple[int, int] = (14, 14)
    num_classes: int = 5  # Including background
    use_attention: bool = False  # Disable attention since it's not in the checkpoint
    attention_type: str = "self"

@dataclass
class ROIConfig:
    """Configuration for Region of Interest (ROI) head."""
    
    box_head_dim: int = 1024
    num_classes: int = 5  # Including background
    box_score_thresh: float = 0.05
    box_nms_thresh: float = 0.5
    box_detections_per_img: int = 100
    box_fg_iou_thresh: float = 0.5
    box_bg_iou_thresh: float = 0.5
    box_batch_size_per_image: int = 512
    box_positive_fraction: float = 0.25
    bbox_reg_weights: List[float] = field(default_factory=lambda: [10.0, 10.0, 5.0, 5.0])
    roi_size: Tuple[int, int] = (7, 7)
    roi_sample_num: int = 2
    classes: int = 5  # Including background
    reg_class_agnostic: bool = False

@dataclass
class DetectionConfig:
    """Configuration for glomeruli detection."""
    
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    fpn: FPNConfig = field(default_factory=FPNConfig)
    num_classes: int = 5  # Including background
    score_threshold: float = 0.5
    nms_threshold: float = 0.5
    roi: ROIConfig = field(default_factory=ROIConfig)
    cascade: CascadeConfig = field(default_factory=CascadeConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    use_bifpn: bool = False
    use_attention: bool = False  # Disable attention since it's not in the checkpoint
    attention_type: str = "self"


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