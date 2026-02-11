#!/usr/bin/env python3
"""
Dataset Download Script.

Downloads the Cats and Dogs classification dataset from Kaggle using kagglehub.
"""

import os
import sys
import shutil
from pathlib import Path
import argparse
import kagglehub


def download_from_kaggle(output_dir: str, dataset: str = "bhavikjikadara/dog-and-cat-classification-dataset"):
    """
    Download dataset from Kaggle using kagglehub.
    
    Args:
        output_dir: Directory to save the dataset
        dataset: Kaggle dataset identifier
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading dataset: {dataset}")
    print(f"Output directory: {output_path}")
    
    try:
        # Download using kagglehub
        path = kagglehub.dataset_download(dataset)
        print(f"Dataset downloaded to: {path}")
        
        source_path = Path(path)
        
        # Show dataset contents
        print(f"\nDataset contents:")
        for item in source_path.rglob("*"):
            if item.is_file():
                print(f"  {item.relative_to(source_path)}")
        
        # Copy the dataset to our data/raw directory
        if source_path.exists():
            for item in source_path.iterdir():
                dest = output_path / item.name
                if item.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(item, dest)
                    print(f"Copied directory: {item.name}")
                else:
                    shutil.copy2(item, dest)
                    print(f"Copied file: {item.name}")
        
        print("Download complete!")
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False


def organize_dataset(data_dir: str):
    """
    Organize the downloaded dataset into the expected structure.
    
    Expected structure:
    data/raw/
        cats/
            cat.1.jpg
            cat.2.jpg
            ...
        dogs/
            dog.1.jpg
            dog.2.jpg
            ...
    """
    data_path = Path(data_dir)
    
    # Create target directories
    cats_dir = data_path / "cats"
    dogs_dir = data_path / "dogs"
    cats_dir.mkdir(exist_ok=True)
    dogs_dir.mkdir(exist_ok=True)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    for img_path in data_path.rglob('*'):
        if img_path.suffix.lower() in image_extensions:
            filename = img_path.name.lower()
            
            # Determine class
            if 'cat' in filename:
                target_dir = cats_dir
            elif 'dog' in filename:
                target_dir = dogs_dir
            else:
                continue
            
            # Move file
            target_path = target_dir / img_path.name
            if img_path != target_path and not target_path.exists():
                shutil.copy2(img_path, target_path)
    
    # Clean up any nested directories
    for item in data_path.iterdir():
        if item.is_dir() and item.name not in ['cats', 'dogs']:
            shutil.rmtree(item)
    
    # Count images
    cat_count = len(list(cats_dir.glob('*')))
    dog_count = len(list(dogs_dir.glob('*')))
    
    print(f"\nDataset organized:")
    print(f"  Cats: {cat_count} images")
    print(f"  Dogs: {dog_count} images")
    print(f"  Total: {cat_count + dog_count} images")


def main():
    parser = argparse.ArgumentParser(description="Download Cats vs Dogs dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Output directory for dataset"
    )
    parser.add_argument(
        "--organize",
        action="store_true",
        help="Organize dataset into cats/dogs directories"
    )
    
    args = parser.parse_args()
    
    # Download
    success = download_from_kaggle(args.output_dir)
    
    if success and args.organize:
        organize_dataset(args.output_dir)


if __name__ == "__main__":
    main()
