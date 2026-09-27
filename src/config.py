"""YAML-backed training configuration, with CLI-friendly overrides.

Centralizing hyperparameters here (instead of hardcoded notebook cells)
means a training run is reproducible from a single file + optional
command-line overrides, and can be diffed/reviewed like any other code.
"""

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class TrainConfig:
    data_root: str = "processed"
    test_root: str = "processed/test"
    model_name: str = "resnet50_fpn"
    common_size_width: int = 1024
    common_size_height: int = 1024
    batch_size: int = 16
    val_batch_size: int = 32
    num_epochs: int = 15
    learning_rate: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    train_val_split: float = 0.8
    num_workers: int = 2
    seed: int = 42
    device: Optional[str] = None  # None -> auto-detect cuda/cpu
    output_dir: str = "outputs"
    use_augmentation: bool = True

    @property
    def common_size(self) -> tuple:
        return (self.common_size_width, self.common_size_height)


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> TrainConfig:
    """Load a TrainConfig from a YAML file, then apply CLI-style overrides.

    Args:
        path: Path to a YAML file. If None, defaults are used.
        overrides: Dict of field_name -> value; None values are ignored (this
            is how argparse's ``default=None`` flags become "no override").
    """
    data: Dict[str, Any] = {}
    if path is not None:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

    valid_fields = {f.name for f in fields(TrainConfig)}
    unknown = set(data) - valid_fields
    if unknown:
        raise ValueError(f"Unknown config key(s) in {path}: {sorted(unknown)}")

    config = TrainConfig(**data)

    if overrides:
        for key, value in overrides.items():
            if value is None:
                continue
            if key not in valid_fields:
                raise ValueError(f"Unknown override key: {key}")
            setattr(config, key, value)

    return config


def save_config(config: TrainConfig, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(asdict(config), f, sort_keys=False)
