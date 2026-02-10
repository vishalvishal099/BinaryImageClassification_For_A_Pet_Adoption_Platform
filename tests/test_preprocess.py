"""
Unit tests for data preprocessing functions.
"""

import pytest
import numpy as np
from PIL import Image
import tempfile
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.preprocess import (
    load_and_resize_image,
    validate_image,
    split_dataset,
    get_class_labels,
    get_label_names,
    IMAGE_SIZE,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO
)


class TestLoadAndResizeImage:
    """Tests for load_and_resize_image function."""
    
    def test_load_valid_rgb_image(self, tmp_path):
        """Test loading a valid RGB image."""
        # Create a test image
        img = Image.new('RGB', (500, 500), color='red')
        img_path = tmp_path / "test_image.jpg"
        img.save(img_path)
        
        # Load and resize
        result = load_and_resize_image(str(img_path))
        
        # Assertions
        assert result is not None
        assert result.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
        assert result.dtype == np.uint8
    
    def test_load_grayscale_converts_to_rgb(self, tmp_path):
        """Test that grayscale images are converted to RGB."""
        # Create a grayscale image
        img = Image.new('L', (300, 300), color=128)
        img_path = tmp_path / "gray_image.png"
        img.save(img_path)
        
        # Load and resize
        result = load_and_resize_image(str(img_path))
        
        # Should be converted to RGB
        assert result is not None
        assert result.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    
    def test_load_rgba_converts_to_rgb(self, tmp_path):
        """Test that RGBA images are converted to RGB."""
        # Create an RGBA image
        img = Image.new('RGBA', (400, 400), color=(255, 0, 0, 128))
        img_path = tmp_path / "rgba_image.png"
        img.save(img_path)
        
        # Load and resize
        result = load_and_resize_image(str(img_path))
        
        # Should be converted to RGB
        assert result is not None
        assert result.shape == (IMAGE_SIZE, IMAGE_SIZE, 3)
    
    def test_load_nonexistent_file_returns_none(self):
        """Test that loading a non-existent file returns None."""
        result = load_and_resize_image("/nonexistent/path/image.jpg")
        assert result is None
    
    def test_custom_target_size(self, tmp_path):
        """Test loading with custom target size."""
        img = Image.new('RGB', (500, 500), color='blue')
        img_path = tmp_path / "test_image.jpg"
        img.save(img_path)
        
        # Load with custom size
        result = load_and_resize_image(str(img_path), target_size=(128, 128))
        
        assert result is not None
        assert result.shape == (128, 128, 3)
    
    def test_aspect_ratio_not_preserved(self, tmp_path):
        """Test that non-square images are resized to square."""
        # Create a non-square image
        img = Image.new('RGB', (800, 400), color='green')
        img_path = tmp_path / "wide_image.jpg"
        img.save(img_path)
        
        result = load_and_resize_image(str(img_path))
        
        # Result should be square
        assert result.shape[0] == result.shape[1] == IMAGE_SIZE


class TestValidateImage:
    """Tests for validate_image function."""
    
    def test_valid_image(self):
        """Test validation of a correct image."""
        img = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        assert validate_image(img) is True
    
    def test_none_image(self):
        """Test validation of None."""
        assert validate_image(None) is False
    
    def test_wrong_dimensions(self):
        """Test validation of image with wrong dimensions."""
        # 2D image
        img_2d = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        assert validate_image(img_2d) is False
        
        # 4D image
        img_4d = np.zeros((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        assert validate_image(img_4d) is False
    
    def test_wrong_channels(self):
        """Test validation of image with wrong number of channels."""
        # 4 channels (RGBA)
        img_4c = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 4), dtype=np.uint8)
        assert validate_image(img_4c) is False
        
        # 1 channel (grayscale)
        img_1c = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.uint8)
        assert validate_image(img_1c) is False
    
    def test_wrong_size(self):
        """Test validation of image with wrong size."""
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        assert validate_image(img) is False


class TestSplitDataset:
    """Tests for split_dataset function."""
    
    def test_default_split_ratios(self):
        """Test that default split ratios are correct."""
        # Create sample paths
        paths = [f"image_{i}.jpg" for i in range(100)]
        
        train, val, test = split_dataset(paths)
        
        # Check approximate sizes (may vary by 1-2 due to rounding)
        assert len(train) == int(100 * TRAIN_RATIO)
        assert len(val) == int(100 * VAL_RATIO)
        assert len(test) == 100 - len(train) - len(val)
    
    def test_custom_split_ratios(self):
        """Test custom split ratios."""
        paths = [f"image_{i}.jpg" for i in range(100)]
        
        train, val, test = split_dataset(
            paths,
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.1
        )
        
        assert len(train) == 70
        assert len(val) == 20
        assert len(test) == 10
    
    def test_no_overlap_between_splits(self):
        """Test that there is no overlap between splits."""
        paths = [f"image_{i}.jpg" for i in range(100)]
        
        train, val, test = split_dataset(paths)
        
        train_set = set(train)
        val_set = set(val)
        test_set = set(test)
        
        # Check no overlap
        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0
    
    def test_all_samples_included(self):
        """Test that all samples are included in splits."""
        paths = [f"image_{i}.jpg" for i in range(100)]
        
        train, val, test = split_dataset(paths)
        
        all_split = set(train + val + test)
        assert all_split == set(paths)
    
    def test_reproducibility_with_seed(self):
        """Test that splits are reproducible with same seed."""
        paths = [f"image_{i}.jpg" for i in range(100)]
        
        train1, val1, test1 = split_dataset(paths, seed=42)
        train2, val2, test2 = split_dataset(paths, seed=42)
        
        assert train1 == train2
        assert val1 == val2
        assert test1 == test2
    
    def test_different_seeds_give_different_splits(self):
        """Test that different seeds give different splits."""
        paths = [f"image_{i}.jpg" for i in range(100)]
        
        train1, _, _ = split_dataset(paths, seed=42)
        train2, _, _ = split_dataset(paths, seed=123)
        
        assert train1 != train2
    
    def test_invalid_ratios_raises_error(self):
        """Test that invalid ratios raise an error."""
        paths = [f"image_{i}.jpg" for i in range(100)]
        
        with pytest.raises(AssertionError):
            split_dataset(paths, train_ratio=0.5, val_ratio=0.3, test_ratio=0.1)


class TestClassLabels:
    """Tests for class label functions."""
    
    def test_get_class_labels(self):
        """Test get_class_labels returns correct mapping."""
        labels = get_class_labels()
        
        assert "cats" in labels
        assert "dogs" in labels
        assert labels["cats"] == 0
        assert labels["dogs"] == 1
    
    def test_get_label_names(self):
        """Test get_label_names returns correct mapping."""
        names = get_label_names()
        
        assert 0 in names
        assert 1 in names
        assert names[0] == "cat"
        assert names[1] == "dog"
    
    def test_labels_and_names_are_consistent(self):
        """Test that class labels and label names are consistent."""
        class_labels = get_class_labels()
        label_names = get_label_names()
        
        for class_name, label in class_labels.items():
            # Remove trailing 's' from class name for comparison
            expected_name = class_name.rstrip('s')
            assert label_names[label] == expected_name


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
