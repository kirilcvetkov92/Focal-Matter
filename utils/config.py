from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

@dataclass
class ModelConfig:
    encoder_weights_path: Optional[str] = "checkpoints/focalnet_tiny_srf.pth"
    checkpoint_path: Optional[str] = "checkpoints/0candidate5_step40000.ckpt"
    checkpoint_key: str = "model"
    strict_checkpoint: bool = False

@dataclass
class DataConfig:
    matting_root: str = "F://Master/Train/"
    background_root: str = "F://Master/train2014/"
    real_world_aug: bool = False
    basic_data_returnhog: Any = 1024
    hard_alpha_paths: Tuple[str, ...] = ()
    hard_sample_prob: float = 0.0
    hard_fg_dir: Optional[str] = None
    batch_size: int = 1
    workers: int = 4
    shuffle: bool = True
    drop_last: bool = True
    pin_memory: bool = True


@dataclass
class TrainConfig:
    start_epoch: int = 1
    epochs: int = 199
    learning_rate: float = 1.25e-5
    betas: Tuple[float, float] = (0.5, 0.9999)
    weight_decay: float = 1e-6
    no_decay_name_parts: Tuple[str, ...] = (
        "norm",
        "bias",
    )
    accum_steps: int = 2
    grad_clip: float = 10.0
    cosine_t_max: int = 200
    progress_every: int = 1
    log_every: int = 100
    save_every_steps: int = 20000
    save_epoch_checkpoint: bool = True
    checkpoint_prefix: str = "candidate5"
    run_eval_after_epoch: bool = True
    eval_full_metrics: bool = False


@dataclass
class LossConfig:
    alpha_weight: float = 0.5
    unknown_weight: float = 0.5
    laplacian_weight: float = 0.5
    laplacian_levels: int = 5
    laplacian_kernel_size: Tuple[int, int] = (5, 5)
    laplacian_sigma: Tuple[float, float] = (1.49, 1.49)
    laplacian_border_type: str = "reflect"
    laplacian_type: str = "L1"
    laplacian_multiplier: float = 1.99
    laplacian_clip: bool = True
    laplacian_clipmin: float = 0
    laplacian_clipmax: float = 1
    laplacian_reduction: str = "mean"


@dataclass
class RuntimeConfig:
    cuda_visible_devices: Optional[str] = "0"
    device: str = "cuda"
    seed: Optional[int] = None
    cudnn_benchmark: bool = True
    kmp_duplicate_lib_ok: bool = False


@dataclass
class ExperimentConfig:
    root: str = "experiments"
    name: str = "candidate5"
    timestamp_format: str = "%Y%m%d-%H%M%S"
    log_level: str = "INFO"
    log_to_console: bool = True
    log_to_file: bool = True
    save_config_snapshot: bool = True


@dataclass
class EncoderConfig:
    in_chans: int = 3
    embed_dim: int = 96
    depths: Tuple[int, int, int, int] = (2, 2, 6, 2)
    focal_levels: Tuple[int, int, int, int] = (2, 2, 2, 2)
    focal_windows: Tuple[int, int, int, int] = (3, 3, 3, 3)



@dataclass
class EvalConfig:
    trimap_dir: str = "G://Master/master_validation/trimaps/"
    image_dir: str = "G://Master/master_validation/merged/"
    alpha_dir: str = "G://Master/master_validation/alpha_copy/"
    output_dir: Optional[str] = None
    pad_to_multiple: int = 32
    write_outputs: bool = True
    full_metrics: bool = True
    limit: Optional[int] = None


@dataclass
class ProjectConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


DEFAULT_CONFIG = ProjectConfig()


def clone_config() -> ProjectConfig:
    return copy.deepcopy(DEFAULT_CONFIG)


def update_config(config: ProjectConfig, overrides: Dict[str, Any]) -> ProjectConfig:
    for key, value in overrides.items():
        current = getattr(config, key)
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if not hasattr(current, nested_key):
                    raise KeyError("Unknown config key: {}.{}".format(key, nested_key))
                setattr(current, nested_key, nested_value)
        else:
            setattr(config, key, value)
    return config
