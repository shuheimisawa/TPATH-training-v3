# src/config/model_config.py
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union


@dataclass
class BackboneConfig:
    name: str = "resnet50"
    pretrained: bool = True
    freeze_stages: int = 1
    norm_eval: bool = True
    out_indices: Tuple[int, ...] = (0, 1, 2, 3)


@dataclass
class FPNConfig:
    in_channels: List[int] = (256, 512, 1024, 2048)
    out_channels: int = 256
    num_outs: int = 5
    add_extra_convs: bool = True
    extra_convs_on_inputs: bool = False


@dataclass
class RPNConfig:
    anchor_scales: List[int] = (8, 16, 32)
    anchor_ratios: List[float] = (0.5, 1.0, 2.0)
    anchor_strides: List[int] = (4, 8, 16, 32, 64)
    target_means: List[float] = (0.0, 0.0, 0.0, 0.0)
    target_stds: List[float] = (1.0, 1.0, 1.0, 1.0)
    feat_channels: int = 256
    use_sigmoid_cls: bool = True
    nms_pre: int = 2000
    nms_post: int = 2000
    nms_thr: float = 0.7
    min_bbox_size: int = 0
    num_max_proposals: int = 2000


@dataclass
class ROIConfig:
    roi_layer: dict = None
    roi_size: Tuple[int, int] = (7, 7)
    roi_sample_num: int = 2
    target_means: List[float] = (0.0, 0.0, 0.0, 0.0)
    target_stds: List[float] = (0.1, 0.1, 0.2, 0.2)
    reg_class_agnostic: bool = False
    classes: int = 5  # background + 4 classes (Normal, Sclerotic, Partially_sclerotic, Uncertain)

    def __post_init__(self):
        if self.roi_layer is None:
            self.roi_layer = {
                'type': 'RoIAlign',
                'out_size': self.roi_size,
                'sample_num': self.roi_sample_num
            }


@dataclass
class CascadeRCNNConfig:
    num_stages: int = 3
    stage_loss_weights: List[float] = (1, 0.5, 0.25)
    bbox_reg_weights: List[Tuple[float, float, float, float]] = (
        (10.0, 10.0, 5.0, 5.0),
        (20.0, 20.0, 10.0, 10.0),
        (30.0, 30.0, 15.0, 15.0)
    )
    iou_thresholds: List[float] = (0.5, 0.6, 0.7)


@dataclass
class MaskConfig:
    roi_size: Tuple[int, int] = (14, 14)
    in_channels: int = 256
    conv_kernel_size: int = 1
    classes: int = 5  # background + 4 classes (Normal, Sclerotic, Partially_sclerotic, Uncertain)
    

@dataclass
class ModelConfig:
    backbone: BackboneConfig = BackboneConfig()
    fpn: FPNConfig = FPNConfig()
    rpn: RPNConfig = RPNConfig()
    roi: ROIConfig = ROIConfig()
    cascade: CascadeRCNNConfig = CascadeRCNNConfig()
    mask: MaskConfig = MaskConfig()
    num_classes: int = 5  # background + 4 classes (Normal, Sclerotic, Partially_sclerotic, Uncertain)
    pretrained: Optional[str] = None
    

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
    type: str = "step"
    step_size: int = 8
    gamma: float = 0.1
    milestones: List[int] = field(default_factory=lambda: [8, 11])


@dataclass
class DataConfig:
    train_path: str = "data/train"
    val_path: str = "data/val"
    test_path: str = "data/test"
    img_size: Tuple[int, int] = (1024, 1024)
    classes: List[str] = field(default_factory=lambda: ["Normal", "Sclerotic", "Partially_sclerotic", "Uncertain"])
    
    # Data augmentation
    use_augmentation: bool = True
    augmentations: Dict = field(default_factory=lambda: {
        "horizontal_flip": {"p": 0.5},
        "vertical_flip": {"p": 0.5},
        "random_rotate_90": {"p": 0.5},
        "transpose": {"p": 0.5},
        "random_crop": {"p": 0.3, "height": 800, "width": 800}
    })
    
    # Normalization
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass
class TrainingConfig:
    seed: int = 42
    epochs: int = 12
    batch_size: int = 2
    workers: int = 4
    optimizer: OptimizerConfig = OptimizerConfig()
    lr_scheduler: LRSchedulerConfig = LRSchedulerConfig()
    data: DataConfig = DataConfig()
    
    # Loss weights
    rpn_cls_loss_weight: float = 1.0
    rpn_bbox_loss_weight: float = 1.0
    rcnn_cls_loss_weight: float = 1.0
    rcnn_bbox_loss_weight: float = 1.0
    mask_loss_weight: float = 1.0
    
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