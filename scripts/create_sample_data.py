#!/usr/bin/env python3
"""
Create sample synthetic data for quick start demonstration.
Generates random images for cats and dogs categories.
"""

import os
import random
import numpy as np
from PIL import Image
from pathlib import Path


def create_synthetic_image(category: str, size: tuple = (224, 224)) -> Image.Image:
    """
    Create a synthetic image for demonstration.
    Uses different color distributions for cats vs dogs.
    """
    # Create base array
    img_array = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    
    if category == "cats":
        # Cats: More orange/gray tones
        base_color = [random.randint(150, 220), random.randint(100, 180), random.randint(80, 150)]
    else:
        # Dogs: More brown/golden tones
        base_color = [random.randint(120, 200), random.randint(80, 160), random.randint(50, 120)]
    
    # Fill with base color and add noise
    for i in range(3):
        noise = np.random.randint(-30, 30, size=size, dtype=np.int16)
        channel = np.clip(base_color[i] + noise, 0, 255).astype(np.uint8)
        img_array[:, :, i] = channel
    
    # Add some patterns to make images more realistic
    # Add random circles (eyes, spots)
    for _ in range(random.randint(2, 5)):
        cx, cy = random.randint(0, size[0]-1), random.randint(0, size[1]-1)
        radius = random.randint(5, 30)
        color = [random.randint(0, 255) for _ in range(3)]
        
        for x in range(max(0, cx-radius), min(size[0], cx+radius)):
            for y in range(max(0, cy-radius), min(size[1], cy+radius)):
                if (x - cx)**2 + (y - cy)**2 < radius**2:
                    for c in range(3):
                        img_array[x, y, c] = color[c]
    
    return Image.fromarray(img_array)


def create_sample_dataset(
    output_dir: str = "data/raw",
    num_train_per_class: int = 100,
    num_val_per_class: int = 20,
    num_test_per_class: int = 20
):
    """
    Create a sample dataset with train/val/test splits.
    """
    output_path = Path(output_dir)
    
    categories = ["cats", "dogs"]
    splits = {
        "train": num_train_per_class,
        "val": num_val_per_class,
        "test": num_test_per_class
    }
    
    print("Creating sample dataset for demonstration...")
    print(f"Output directory: {output_path.absolute()}")
    
    total_images = 0
    
    for split_name, num_images in splits.items():
        for category in categories:
            split_dir = output_path / split_name / category
            split_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  Creating {num_images} {category} images for {split_name}...")
            
            for i in range(num_images):
                img = create_synthetic_image(category)
                img_path = split_dir / f"{category}_{i:04d}.jpg"
                img.save(img_path, "JPEG", quality=85)
                total_images += 1
    
    print(f"\nDataset created successfully!")
    print(f"Total images: {total_images}")
    print(f"\nStructure:")
    print(f"  {output_path}/")
    for split_name, num_images in splits.items():
        print(f"    {split_name}/")
        for category in categories:
            print(f"      {category}/ ({num_images} images)")
    
    return output_path


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create sample dataset for demonstration")
    parser.add_argument("--output-dir", type=str, default="data/raw",
                        help="Output directory for the dataset")
    parser.add_argument("--train-samples", type=int, default=100,
                        help="Number of training samples per class")
    parser.add_argument("--val-samples", type=int, default=20,
                        help="Number of validation samples per class")
    parser.add_argument("--test-samples", type=int, default=20,
                        help="Number of test samples per class")
    
    args = parser.parse_args()
    
    create_sample_dataset(
        output_dir=args.output_dir,
        num_train_per_class=args.train_samples,
        num_val_per_class=args.val_samples,
        num_test_per_class=args.test_samples
    )


if __name__ == "__main__":
    main()
