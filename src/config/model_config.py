# src/config/model_config.py
from dataclasses import dataclass, field
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
    in_channels: List[int] = field(default_factory=lambda: [256, 512, 1024, 2048])
    out_channels: int = 256
    num_outs: int = 5
    add_extra_convs: bool = True
    extra_convs_on_inputs: bool = False
    num_blocks: int = 3
    attention_type: str = 'none'


@dataclass
class RPNConfig:
    anchor_scales: List[int] = field(default_factory=lambda: [8, 16, 32])
    anchor_ratios: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    anchor_strides: List[int] = field(default_factory=lambda: [4, 8, 16, 32, 64])
    target_means: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    target_stds: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0])
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
    target_means: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    target_stds: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.2, 0.2])
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
    stage_loss_weights: List[float] = field(default_factory=lambda: [1, 0.5, 0.25])
    bbox_reg_weights: List[Tuple[float, float, float, float]] = field(
        default_factory=lambda: [
            (10.0, 10.0, 5.0, 5.0),
            (20.0, 20.0, 10.0, 10.0),
            (30.0, 30.0, 15.0, 15.0)
        ]
    )
    iou_thresholds: List[float] = field(default_factory=lambda: [0.5, 0.6, 0.7])


@dataclass
class MaskConfig:
    roi_size: Tuple[int, int] = (14, 14)
    in_channels: int = 256
    conv_kernel_size: int = 1
    classes: int = 5  # background + 4 classes (Normal, Sclerotic, Partially_sclerotic, Uncertain)
    

@dataclass
class ModelConfig:
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    fpn: FPNConfig = field(default_factory=FPNConfig)
    rpn: RPNConfig = field(default_factory=RPNConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    cascade: CascadeRCNNConfig = field(default_factory=CascadeRCNNConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    num_classes: int = 5  # background + 4 classes (Normal, Sclerotic, Partially_sclerotic, Uncertain)
    pretrained: Optional[str] = None
    use_bifpn: bool = False
    use_attention: bool = False
    attention_type: str = 'none'