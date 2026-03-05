import os
import json
import random
from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class DataConfig:
    base_dir: str = "data/plantvillage/color"
    img_size: int = 224
    batch_size: int = 32
    val_split: float = 0.2
    seed: int = 42
    num_workers: int = 2


def get_transforms(img_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
    train_tfms = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tfms, val_tfms


def build_dataloaders(cfg: DataConfig, save_class_map_path: str = "artifacts/reports/class_to_idx.json"):
    if not os.path.exists(cfg.base_dir):
        raise FileNotFoundError(f"Dataset path not found: {cfg.base_dir}")

    seed_everything(cfg.seed)

    train_tfms, val_tfms = get_transforms(cfg.img_size)

    full_train = datasets.ImageFolder(cfg.base_dir, transform=train_tfms)
    full_val = datasets.ImageFolder(cfg.base_dir, transform=val_tfms)

    n_total = len(full_train)
    n_val = int(cfg.val_split * n_total)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(cfg.seed)
    train_ds, _ = random_split(full_train, [n_train, n_val], generator=generator)
    _, val_ds = random_split(full_val, [n_train, n_val], generator=generator)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    os.makedirs(os.path.dirname(save_class_map_path), exist_ok=True)
    with open(save_class_map_path, "w") as f:
        json.dump(full_train.class_to_idx, f, indent=2)

    return train_loader, val_loader, full_train.class_to_idx