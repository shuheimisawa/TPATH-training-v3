# src/config/training_config.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class OptimizerConfig:
    type: str = "SGD"
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0001


@dataclass
class LRSchedulerConfig:
    type: str = "multistep"
    milestones: List[int] = field(default_factory=lambda: [8, 11])
    gamma: float = 0.1


@dataclass
class DataConfig:
    train_path: str = "data_test/processed/train"
    val_path: str = "data_test/processed/val"
    test_path: str = "data_test/processed/test"
    img_size: Tuple[int, int] = (512, 512)
    classes: List[str] = field(default_factory=lambda: ["Normal", "Sclerotic", "Partially_sclerotic", "Uncertain"])
    
    # Data augmentation
    use_augmentation: bool = True
    augmentations: Dict = field(default_factory=lambda: {
        "horizontal_flip": {"p": 0.5},
        "vertical_flip": {"p": 0.5},
        "random_rotate_90": {"p": 0.5},
        "transpose": {"p": 0.5},
        "random_crop": {"p": 0.5, "size": [512, 512]},
        "random_brightness_contrast": {"p": 0.5, "brightness_limit": 0.2, "contrast_limit": 0.2},
        "random_gamma": {"p": 0.5, "gamma_limit": (80, 120)},
        "gauss_noise": {"p": 0.5, "var_limit": (10.0, 50.0)},
        "elastic_transform": {"p": 0.5, "alpha": 120, "sigma": 120 * 0.05, "alpha_affine": 120 * 0.03}
    })
    
    # Normalization
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class ClassWeightsConfig:
    # Background class weight always set to 1.0 (index 0)
    # The remaining weights correspond to the classes defined in DataConfig.classes
    background: float = 1.0
    Normal: float = 1.0
    Sclerotic: float = 2.0         # Higher weight for challenging class
    Partially_sclerotic: float = 1.5
    Uncertain: float = 1.0


@dataclass
class TrainingConfig:
    seed: int = 42
    epochs: int = 12
    batch_size: int = 2
    workers: int = 4
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    lr_scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Class weights for loss calculation
    class_weights: ClassWeightsConfig = field(default_factory=ClassWeightsConfig)
    
    # Loss weights
    rpn_cls_loss_weight: float = 1.0
    rpn_bbox_loss_weight: float = 1.0
    rcnn_cls_loss_weight: float = 1.0
    rcnn_bbox_loss_weight: float = 1.0
    mask_loss_weight: float = 1.0
    dice_weight: float = 0.5  # Weight for Dice loss component in mask loss
    
    # Checkpointing and logging
    checkpoint_dir: str = "experiments/checkpoints"
    log_dir: str = "experiments/logs"
    save_freq: int = 1
    eval_freq: int = 1
    log_freq: int = 10
    
    # Distributed training
    distributed: bool = False
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    
    # Debugging
    debug: bool = False
    debug_samples: int = 10