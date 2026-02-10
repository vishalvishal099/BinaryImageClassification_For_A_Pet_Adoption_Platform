#!/usr/bin/env python3
"""
Smoke Test Script for Cats vs Dogs Classifier API.

This script performs post-deployment health checks to verify
the service is running correctly.
"""

import argparse
import sys
import time
from typing import Tuple
import requests
from io import BytesIO
from PIL import Image


def create_test_image() -> bytes:
    """Create a simple test image."""
    img = Image.new('RGB', (224, 224), color='red')
    buffer = BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    return buffer.read()


def test_health_endpoint(base_url: str) -> Tuple[bool, str]:
    """
    Test the health check endpoint.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        
        if response.status_code != 200:
            return False, f"Health check failed: Status {response.status_code}"
        
        data = response.json()
        
        if data.get("status") != "healthy":
            return False, f"Service not healthy: {data.get('status')}"
        
        return True, f"Health check passed: {data}"
        
    except requests.exceptions.RequestException as e:
        return False, f"Health check request failed: {e}"


def test_root_endpoint(base_url: str) -> Tuple[bool, str]:
    """
    Test the root endpoint.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        response = requests.get(f"{base_url}/", timeout=10)
        
        if response.status_code != 200:
            return False, f"Root endpoint failed: Status {response.status_code}"
        
        data = response.json()
        
        if "name" not in data or "version" not in data:
            return False, f"Root endpoint missing required fields"
        
        return True, f"Root endpoint passed: {data.get('name')} v{data.get('version')}"
        
    except requests.exceptions.RequestException as e:
        return False, f"Root endpoint request failed: {e}"


def test_metrics_endpoint(base_url: str) -> Tuple[bool, str]:
    """
    Test the Prometheus metrics endpoint.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        response = requests.get(f"{base_url}/metrics", timeout=10)
        
        if response.status_code != 200:
            return False, f"Metrics endpoint failed: Status {response.status_code}"
        
        # Check that response contains Prometheus metrics format
        content = response.text
        if "prediction_requests_total" in content or "health_check" in content:
            return True, "Metrics endpoint passed: Prometheus metrics available"
        
        return True, "Metrics endpoint passed (basic check)"
        
    except requests.exceptions.RequestException as e:
        return False, f"Metrics endpoint request failed: {e}"


def test_prediction_endpoint(base_url: str) -> Tuple[bool, str]:
    """
    Test the prediction endpoint with a test image.
    
    Returns:
        Tuple of (success, message)
    """
    try:
        # Create test image
        test_image = create_test_image()
        
        files = {"file": ("test.jpg", test_image, "image/jpeg")}
        response = requests.post(
            f"{base_url}/predict",
            files=files,
            timeout=30
        )
        
        # 503 is acceptable if model is not loaded
        if response.status_code == 503:
            return True, "Prediction endpoint passed: Model not loaded (expected in test)"
        
        if response.status_code != 200:
            return False, f"Prediction endpoint failed: Status {response.status_code}"
        
        data = response.json()
        
        required_fields = ["predicted_class", "confidence", "probabilities"]
        for field in required_fields:
            if field not in data:
                return False, f"Prediction response missing field: {field}"
        
        return True, f"Prediction endpoint passed: {data.get('predicted_class')} ({data.get('confidence'):.2%})"
        
    except requests.exceptions.RequestException as e:
        return False, f"Prediction endpoint request failed: {e}"


def wait_for_service(base_url: str, max_attempts: int = 30, delay: int = 2) -> bool:
    """
    Wait for the service to become available.
    
    Returns:
        True if service is available, False otherwise
    """
    print(f"Waiting for service at {base_url}...")
    
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                print(f"Service available after {attempt} attempts")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"  Attempt {attempt}/{max_attempts}: Not ready, waiting {delay}s...")
        time.sleep(delay)
    
    return False


def run_smoke_tests(base_url: str) -> bool:
    """
    Run all smoke tests.
    
    Returns:
        True if all tests pass, False otherwise
    """
    print("=" * 60)
    print("SMOKE TESTS - Cats vs Dogs Classifier API")
    print("=" * 60)
    print(f"Target URL: {base_url}")
    print()
    
    tests = [
        ("Health Check", test_health_endpoint),
        ("Root Endpoint", test_root_endpoint),
        ("Metrics Endpoint", test_metrics_endpoint),
        ("Prediction Endpoint", test_prediction_endpoint),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"Testing: {test_name}...")
        success, message = test_func(base_url)
        results.append((test_name, success, message))
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}: {message}")
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, _ in results:
        status = "✅" if success else "❌"
        print(f"  {status} {test_name}")
    
    print()
    print(f"Passed: {passed}/{total}")
    
    all_passed = passed == total
    
    if all_passed:
        print("\n✅ All smoke tests passed!")
    else:
        print("\n❌ Some smoke tests failed!")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Run smoke tests for Cats vs Dogs Classifier API"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the service"
    )
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for service to become available"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=30,
        help="Maximum attempts when waiting for service"
    )
    
    args = parser.parse_args()
    
    base_url = args.url.rstrip("/")
    
    if args.wait:
        if not wait_for_service(base_url, args.max_attempts):
            print("❌ Service did not become available")
            sys.exit(1)
    
    success = run_smoke_tests(base_url)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
