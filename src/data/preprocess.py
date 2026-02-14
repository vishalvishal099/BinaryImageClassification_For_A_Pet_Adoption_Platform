"""
Data Preprocessing Module for Cats vs Dogs Classification.

This module handles:
- Image loading and resizing to 224x224 RGB
- Train/validation/test splitting (80/10/10)
- Data augmentation for training
"""

import random
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from tqdm import tqdm

# Constants
IMAGE_SIZE = 224
RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1


def set_seed(seed: int = RANDOM_SEED) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def load_and_resize_image(
    image_path: str, target_size: Tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE)
) -> Optional[np.ndarray]:
    """
    Load an image and resize it to the target size.

    Args:
        image_path: Path to the image file
        target_size: Target (width, height) tuple

    Returns:
        Resized RGB image as numpy array, or None if loading fails
    """
    try:
        # Load image using PIL for better format support
        img = Image.open(image_path)

        # Convert to RGB if necessary (handles grayscale, RGBA, etc.)
        if img.mode != "RGB":
            img = img.convert("RGB")

        # Resize using high-quality resampling
        img = img.resize(target_size, Image.Resampling.LANCZOS)

        # Convert to numpy array
        return np.array(img)

    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def validate_image(image: np.ndarray) -> bool:
    """
    Validate that an image has the correct shape and dtype.

    Args:
        image: Image array to validate

    Returns:
        True if valid, False otherwise
    """
    if image is None:
        return False

    if len(image.shape) != 3:
        return False

    if image.shape[2] != 3:
        return False

    if image.shape[0] != IMAGE_SIZE or image.shape[1] != IMAGE_SIZE:
        return False

    return True


def get_train_transforms() -> A.Compose:
    """
    Get data augmentation transforms for training.

    Returns:
        Albumentations Compose object with training transforms
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.Affine(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
            A.GaussianBlur(blur_limit=(3, 7), p=0.1),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def get_val_transforms() -> A.Compose:
    """
    Get transforms for validation/test data (no augmentation).

    Returns:
        Albumentations Compose object with validation transforms
    """
    return A.Compose(
        [
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )


def split_dataset(
    image_paths: List[str],
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split dataset into train, validation, and test sets.

    Args:
        image_paths: List of image file paths
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_paths, val_paths, test_paths)
    """
    assert (
        abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    ), "Ratios must sum to 1.0"

    # Set seed and shuffle
    random.seed(seed)
    shuffled_paths = image_paths.copy()
    random.shuffle(shuffled_paths)

    # Calculate split indices
    n = len(shuffled_paths)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    # Split
    train_paths = shuffled_paths[:train_end]
    val_paths = shuffled_paths[train_end:val_end]
    test_paths = shuffled_paths[val_end:]

    return train_paths, val_paths, test_paths


def process_and_save_images(
    image_paths: List[str],
    output_dir: str,
    class_name: str,
    target_size: Tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE),
) -> int:
    """
    Process images and save to output directory.

    Args:
        image_paths: List of source image paths
        output_dir: Directory to save processed images
        class_name: Class name (cat/dog) for subdirectory
        target_size: Target image size

    Returns:
        Number of successfully processed images
    """
    output_path = Path(output_dir) / class_name
    output_path.mkdir(parents=True, exist_ok=True)

    processed_count = 0

    for img_path in tqdm(image_paths, desc=f"Processing {class_name}"):
        img = load_and_resize_image(img_path, target_size)

        if img is not None and validate_image(img):
            # Save processed image
            filename = Path(img_path).name
            output_file = output_path / filename

            # Save as JPEG
            Image.fromarray(img).save(output_file, "JPEG", quality=95)
            processed_count += 1

    return processed_count


def preprocess_dataset(
    raw_data_dir: str,
    processed_data_dir: str,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> dict:
    """
    Preprocess the entire dataset.

    Args:
        raw_data_dir: Path to raw dataset directory
        processed_data_dir: Path to save processed data
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio

    Returns:
        Dictionary with statistics about processed data
    """
    set_seed(RANDOM_SEED)

    raw_path = Path(raw_data_dir)
    processed_path = Path(processed_data_dir)

    # Clear existing processed data
    if processed_path.exists():
        shutil.rmtree(processed_path)

    stats = {"train": {}, "val": {}, "test": {}}

    # Process each class
    for class_name in ["cats", "dogs"]:
        # Find all images for this class
        class_dir = raw_path / class_name
        if not class_dir.exists():
            # Try alternate naming
            class_dir = raw_path / class_name[:-1]  # cat/dog instead of cats/dogs

        if not class_dir.exists():
            print(f"Warning: Could not find directory for {class_name}")
            continue

        # Get all image paths
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
        image_paths = [
            str(p) for p in class_dir.iterdir() if p.suffix.lower() in image_extensions
        ]

        print(f"Found {len(image_paths)} images for {class_name}")

        # Split dataset
        train_paths, val_paths, test_paths = split_dataset(
            image_paths, train_ratio, val_ratio, test_ratio
        )

        # Process and save each split
        for split_name, paths in [
            ("train", train_paths),
            ("val", val_paths),
            ("test", test_paths),
        ]:
            output_dir = processed_path / split_name
            count = process_and_save_images(paths, str(output_dir), class_name)
            stats[split_name][class_name] = count
            print(f"  {split_name}: {count} images")

    return stats


def get_class_labels() -> dict:
    """Get mapping of class names to labels."""
    return {"cats": 0, "dogs": 1}


def get_label_names() -> dict:
    """Get mapping of labels to class names."""
    return {0: "cat", 1: "dog"}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Cats vs Dogs dataset")
    parser.add_argument(
        "--raw-dir", type=str, default="data/raw", help="Path to raw dataset"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="data/processed",
        help="Path to save processed data",
    )

    args = parser.parse_args()

    print("Starting data preprocessing...")
    stats = preprocess_dataset(args.raw_dir, args.processed_dir)
    print("\nPreprocessing complete!")
    print(f"Statistics: {stats}")
