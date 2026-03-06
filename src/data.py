import os
import json
import random
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image


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
    val_split: float = 0.1
    test_split: float = 0.1
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

    eval_tfms = transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return train_tfms, eval_tfms


class TransformedSubset(Dataset):
    def __init__(self, dataset, indices, transform=None):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image_path, label = self.dataset.samples[self.indices[idx]]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def build_dataloaders(
    cfg: DataConfig,
    save_class_map_path: str = "artifacts/reports/class_to_idx.json"
):
    if not os.path.exists(cfg.base_dir):
        raise FileNotFoundError(f"Dataset path not found: {cfg.base_dir}")

    if cfg.val_split < 0 or cfg.test_split < 0:
        raise ValueError("val_split and test_split must be non-negative")

    if cfg.val_split + cfg.test_split >= 1.0:
        raise ValueError("val_split + test_split must be less than 1.0")

    seed_everything(cfg.seed)

    train_tfms, eval_tfms = get_transforms(cfg.img_size)

    base_dataset = datasets.ImageFolder(cfg.base_dir)
    n_total = len(base_dataset)

    n_val = int(cfg.val_split * n_total)
    n_test = int(cfg.test_split * n_total)
    n_train = n_total - n_val - n_test

    if n_train <= 0:
        raise ValueError("Train split is empty. Reduce val_split or test_split.")

    generator = torch.Generator().manual_seed(cfg.seed)
    indices = torch.randperm(n_total, generator=generator).tolist()

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_ds = TransformedSubset(base_dataset, train_indices, transform=train_tfms)
    val_ds = TransformedSubset(base_dataset, val_indices, transform=eval_tfms)
    test_ds = TransformedSubset(base_dataset, test_indices, transform=eval_tfms)

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

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    os.makedirs(os.path.dirname(save_class_map_path), exist_ok=True)
    with open(save_class_map_path, "w") as f:
        json.dump(base_dataset.class_to_idx, f, indent=2)

    return train_loader, val_loader, test_loader, base_dataset.class_to_idx