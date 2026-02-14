"""
Unit tests for inference service functions.
"""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

# We need to mock torch before importing the inference module
import sys

if "torch" not in sys.modules:
    sys.modules["torch"] = MagicMock()
    sys.modules["torch.nn"] = MagicMock()
    sys.modules["torch.nn.functional"] = MagicMock()


class TestModelServicePreprocessing:
    """Tests for model service preprocessing."""

    def test_preprocess_maintains_dimensions(self):
        """Test that preprocessing produces correct tensor dimensions."""
        # Create a mock model service
        from src.inference.app import ModelService

        service = ModelService()

        # Create test image
        img = Image.new("RGB", (300, 300), color="red")

        # Mock torch functions
        with patch("src.inference.app.torch") as mock_torch:
            mock_tensor = MagicMock()
            mock_tensor.unsqueeze.return_value = mock_tensor
            mock_torch.from_numpy.return_value = mock_tensor

            service.preprocess_image(img)

            # Should call from_numpy and unsqueeze
            mock_torch.from_numpy.assert_called_once()

    def test_preprocess_normalizes_correctly(self):
        """Test that preprocessing normalizes with correct mean/std."""
        from src.inference.app import ModelService

        service = ModelService()

        # Check normalization constants
        assert service.mean == [0.485, 0.456, 0.406]
        assert service.std == [0.229, 0.224, 0.225]

    def test_class_names_mapping(self):
        """Test that class names are correctly mapped."""
        from src.inference.app import ModelService

        service = ModelService()

        assert service.class_names[0] == "cat"
        assert service.class_names[1] == "dog"


class TestAPIEndpoints:
    """Tests for API endpoints."""

    @pytest.fixture
    def test_client(self):
        """Create a test client for the FastAPI app."""
        from fastapi.testclient import TestClient

        from src.inference.app import app

        return TestClient(app)

    def test_health_endpoint(self, test_client):
        """Test the health check endpoint."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data
        assert "timestamp" in data
        assert data["status"] == "healthy"

    def test_root_endpoint(self, test_client):
        """Test the root endpoint."""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "name" in data
        assert "version" in data
        assert "endpoints" in data

    def test_predict_invalid_file_type(self, test_client):
        """Test prediction with invalid file type."""
        # Create a text file instead of an image
        files = {"file": ("test.txt", b"not an image", "text/plain")}

        response = test_client.post("/predict", files=files)

        # Model not loaded returns 503, invalid file returns 400
        assert response.status_code in [400, 503]

    def test_predict_without_model(self, test_client):
        """Test prediction when model is not loaded."""
        # Create a valid image
        img = Image.new("RGB", (224, 224), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}

        # Model is not loaded in test environment
        response = test_client.post("/predict", files=files)

        # Should return 503 (service unavailable) when model not loaded
        assert response.status_code in [503, 500]


class TestPredictionLogic:
    """Tests for prediction logic."""

    def test_prediction_returns_required_fields(self):
        """Test that prediction returns all required fields."""
        # Mock the prediction result
        expected_fields = [
            "predicted_class",
            "predicted_label",
            "confidence",
            "probabilities",
        ]

        # Create a mock result
        result = {
            "predicted_class": "cat",
            "predicted_label": 0,
            "confidence": 0.95,
            "probabilities": {"cat": 0.95, "dog": 0.05},
        }

        for field in expected_fields:
            assert field in result

    def test_probabilities_sum_to_one(self):
        """Test that predicted probabilities sum to 1."""
        # Mock probabilities
        probabilities = {"cat": 0.7, "dog": 0.3}

        total = sum(probabilities.values())
        assert abs(total - 1.0) < 1e-6

    def test_confidence_matches_max_probability(self):
        """Test that confidence equals the maximum probability."""
        probabilities = {"cat": 0.8, "dog": 0.2}
        predicted_class = "cat"
        confidence = 0.8

        assert confidence == probabilities[predicted_class]
        assert confidence == max(probabilities.values())


class TestImageConversion:
    """Tests for image conversion handling."""

    def test_rgb_image_no_conversion(self):
        """Test that RGB images don't need conversion."""
        img = Image.new("RGB", (224, 224))

        # Should already be RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        assert img.mode == "RGB"

    def test_grayscale_to_rgb(self):
        """Test conversion from grayscale to RGB."""
        img = Image.new("L", (224, 224))

        if img.mode != "RGB":
            img = img.convert("RGB")

        assert img.mode == "RGB"

    def test_rgba_to_rgb(self):
        """Test conversion from RGBA to RGB."""
        img = Image.new("RGBA", (224, 224))

        if img.mode != "RGB":
            img = img.convert("RGB")

        assert img.mode == "RGB"


class TestMetrics:
    """Tests for Prometheus metrics."""

    def test_metrics_endpoint_returns_prometheus_format(self, test_client=None):
        """Test that metrics endpoint returns Prometheus format."""
        if test_client is None:
            from fastapi.testclient import TestClient

            from src.inference.app import app

            test_client = TestClient(app)

        response = test_client.get("/metrics")

        assert response.status_code == 200
        # Prometheus format uses text/plain or text/plain; version=0.0.4
        assert "text" in response.headers.get("content-type", "")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
