#!/usr/bin/env python3
"""CLI training entry point.

    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --model-name retinanet --epochs 20 --lr 0.001

Replaces the hardcoded hyperparameters that used to live directly in
waste_project.ipynb with a YAML config + overridable CLI flags, so a run is
reproducible and diffable.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from src.augmentation import get_eval_transform, get_train_transform
from src.config import TrainConfig, load_config, save_config
from src.data import DumpsiteDataset
from src.model import get_model_by_name, train_batch, validate_batch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_dataloaders(config: TrainConfig):
    train_transform = get_train_transform() if config.use_augmentation else get_eval_transform()

    full_dataset = DumpsiteDataset(
        config.data_root, bbox_transform=train_transform, common_size=config.common_size
    )
    train_size = int(config.train_val_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config.seed)
    )
    # Validation should never see training-time augmentation.
    val_dataset.dataset = DumpsiteDataset(
        config.data_root, bbox_transform=get_eval_transform(), common_size=config.common_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=DumpsiteDataset.collate_fn,
        num_workers=config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.val_batch_size,
        shuffle=False,
        collate_fn=DumpsiteDataset.collate_fn,
        num_workers=config.num_workers,
    )
    return train_loader, val_loader


def train(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(config, str(output_dir / "config.yaml"))

    train_loader, val_loader = build_dataloaders(config)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    model = get_model_by_name(config.model_name).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )

    best_val_loss = float("inf")
    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0.0
        for inputs in train_loader:
            loss, _ = train_batch(inputs, model, optimizer, device=device)
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        val_loss = 0.0
        for inputs in val_loader:
            loss, _ = validate_batch(inputs, model, device=device)
            val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)

        print(f"Epoch {epoch + 1}/{config.num_epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        checkpoint_path = output_dir / f"{config.model_name}_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / f"{config.model_name}_best.pth")
            print(f"  -> new best model (val_loss={val_loss:.4f}), saved to {output_dir / f'{config.model_name}_best.pth'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data-root", dest="data_root", type=str, default=None)
    parser.add_argument("--model-name", dest="model_name", type=str, default=None)
    parser.add_argument("--epochs", dest="num_epochs", type=int, default=None)
    parser.add_argument("--batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--lr", dest="learning_rate", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output-dir", dest="output_dir", type=str, default=None)
    parser.add_argument("--no-augmentation", dest="use_augmentation", action="store_const", const=False, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    config = load_config(args.config, overrides=overrides)
    train(config)


if __name__ == "__main__":
    main()
