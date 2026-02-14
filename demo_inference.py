#!/usr/bin/env python3
"""
Demo Script: End-to-End Inference Flow

This script demonstrates how to interact with the Cats vs Dogs classifier API.
Run this script to see the complete inference flow in action.
"""

import requests
import json
from pathlib import Path
import sys

BASE_URL = "http://localhost:8000"

def print_header(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_health():
    """Test the health endpoint."""
    print_header("Step 1: Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_root():
    """Test the root endpoint."""
    print_header("Step 2: API Info")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=10)
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_predict(image_path, expected_class):
    """Test prediction with an image."""
    print(f"\n--- Testing with {expected_class.upper()} image ---")
    print(f"Image: {image_path}")
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": (Path(image_path).name, f, "image/jpeg")}
            response = requests.post(f"{BASE_URL}/predict", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            predicted = result['predicted_class']
            confidence = result['confidence']
            
            # Check if prediction matches expected
            match = "✅" if predicted == expected_class else "❌"
            
            print(f"Predicted: {predicted.upper()} {match}")
            print(f"Confidence: {confidence:.1%}")
            print(f"Probabilities: cat={result['probabilities']['cat']:.1%}, dog={result['probabilities']['dog']:.1%}")
            print(f"Processing Time: {result['processing_time_ms']:.2f}ms")
            return True
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_metrics():
    """Test the metrics endpoint."""
    print_header("Step 4: Prometheus Metrics")
    try:
        response = requests.get(f"{BASE_URL}/metrics", timeout=10)
        
        # Extract key metrics
        lines = response.text.split('\n')
        print("Key Metrics:")
        for line in lines:
            if 'prediction_requests_total' in line and not line.startswith('#'):
                print(f"  {line}")
            if 'prediction_latency_seconds_count' in line:
                print(f"  {line}")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║     🐱🐶 CATS VS DOGS CLASSIFIER - DEMO INFERENCE 🐶🐱     ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Health Check
    if not test_health():
        print("\n❌ Service is not running! Please start the container:")
        print("   podman run -d -p 8000:8000 -v ./models:/app/models cats-dogs-classifier:latest")
        sys.exit(1)
    
    # Step 2: API Info
    test_root()
    
    # Step 3: Predictions
    print_header("Step 3: Image Classification")
    
    # Find test images
    data_dir = Path("data/processed/test")
    
    cat_images = list((data_dir / "cats").glob("*.jpg"))[:2]
    dog_images = list((data_dir / "dogs").glob("*.jpg"))[:2]
    
    if not cat_images and not dog_images:
        print("No test images found. Using placeholder...")
    else:
        # Test with cat images
        for img in cat_images:
            test_predict(str(img), "cat")
        
        # Test with dog images  
        for img in dog_images:
            test_predict(str(img), "dog")
    
    # Step 4: Metrics
    test_metrics()
    
    # Summary
    print_header("🎉 Demo Complete!")
    print("""
    You can interact with the API using:
    
    1. Swagger UI:   http://localhost:8000/docs
    2. Health:       curl http://localhost:8000/health
    3. Predict:      curl -X POST -F "file=@image.jpg" http://localhost:8000/predict
    4. Metrics:      curl http://localhost:8000/metrics
    """)

if __name__ == "__main__":
    main()
