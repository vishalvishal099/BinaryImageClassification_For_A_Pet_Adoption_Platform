"""
Dataset class for Cats vs Dogs classification.
"""

import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .preprocess import (get_class_labels, get_train_transforms,
                         get_val_transforms)


class CatsDogsDataset(Dataset):
    """
    PyTorch Dataset for Cats vs Dogs classification.
    """

    def __init__(
        self, data_dir: str, split: str = "train", transform: Optional[Callable] = None
    ):
        """
        Initialize the dataset.

        Args:
            data_dir: Path to processed data directory
            split: One of 'train', 'val', 'test'
            transform: Optional transform to apply
        """
        self.data_dir = Path(data_dir) / split
        self.split = split
        self.transform = transform or (
            get_train_transforms() if split == "train" else get_val_transforms()
        )

        self.class_labels = get_class_labels()
        self.samples = self._load_samples()

    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load all image paths and labels."""
        samples = []

        for class_name, label in self.class_labels.items():
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.iterdir():
                    if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                        samples.append((str(img_path), label))

        # Shuffle for training
        if self.split == "train":
            random.shuffle(samples)

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Load image
        image = np.array(Image.open(img_path).convert("RGB"))

        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


def create_dataloaders(
    data_dir: str, batch_size: int = 32, num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.

    Args:
        data_dir: Path to processed data directory
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_dataset = CatsDogsDataset(data_dir, split="train")
    val_dataset = CatsDogsDataset(data_dir, split="val")
    test_dataset = CatsDogsDataset(data_dir, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
