#!/usr/bin/env python3
"""Test the inference service."""

import requests
import sys
from pathlib import Path

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint."""
    print("Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_predict(image_path: str):
    """Test prediction endpoint."""
    print(f"\nTesting /predict endpoint with {image_path}...")
    try:
        with open(image_path, "rb") as f:
            files = {"file": (Path(image_path).name, f, "image/jpeg")}
            response = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
        
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"  Predicted Class: {result['predicted_class']}")
            print(f"  Confidence: {result['confidence']:.2%}")
            print(f"  Probabilities: cat={result['probabilities']['cat']:.2%}, dog={result['probabilities']['dog']:.2%}")
        else:
            print(f"  Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"  Error: {e}")
        return False


def test_metrics():
    """Test metrics endpoint."""
    print("\nTesting /metrics endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/metrics", timeout=5)
        print(f"  Status: {response.status_code}")
        print(f"  Metrics available: {'predictions_total' in response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"  Error: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Inference Service Test")
    print("=" * 60)
    
    # Test health
    health_ok = test_health()
    
    if not health_ok:
        print("\nService not available. Make sure the inference service is running:")
        print("  MODEL_PATH=models/best_model.pt uvicorn src.inference.app:app --port 8000")
        sys.exit(1)
    
    # Find a test image
    test_images = list(Path("data/processed/test/cats").glob("*.jpg"))[:1]
    test_images += list(Path("data/processed/test/dogs").glob("*.jpg"))[:1]
    
    if test_images:
        for img_path in test_images:
            test_predict(str(img_path))
    else:
        print("\nNo test images found in data/processed/test/")
    
    # Test metrics
    test_metrics()
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
