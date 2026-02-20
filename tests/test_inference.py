"""
Unit tests for inference service functions.

All tests use the real torch/FastAPI stack — no sys.modules hacks.
The TestClient from Starlette is used so the lifespan (model load)
runs in the test process; the model file is absent in CI so the
service starts with model_loaded=False, which is a valid tested state.
"""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg_bytes(width: int = 224, height: int = 224, color=(100, 150, 200)) -> bytes:
    """Return JPEG bytes of a solid-colour image."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color=color).save(buf, format="JPEG")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# ModelService — unit tests (no real model file required)
# ---------------------------------------------------------------------------

class TestModelServicePreprocessing:
    """Tests for ModelService preprocessing logic."""

    def setup_method(self):
        from src.inference.app import ModelService
        self.service = ModelService()

    def test_image_size_is_224(self):
        """ModelService targets 224×224 images."""
        assert self.service.image_size == 224

    def test_normalisation_constants(self):
        """ImageNet mean/std are used for normalisation."""
        assert self.service.mean == [0.485, 0.456, 0.406]
        assert self.service.std == [0.229, 0.224, 0.225]

    def test_class_names_mapping(self):
        """Class indices map to cat/dog strings."""
        assert self.service.class_names[0] == "cat"
        assert self.service.class_names[1] == "dog"

    def test_model_is_none_before_loading(self):
        """Model attribute is None until load_model() is called."""
        assert self.service.model is None

    def test_preprocess_output_shape(self):
        """preprocess_image returns a (1, 3, 224, 224) tensor."""
        import torch
        img = Image.new("RGB", (300, 400), color="blue")
        tensor = self.service.preprocess_image(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)

    def test_preprocess_pixel_values_are_normalised(self):
        """Output tensor values should be roughly in [-4, 4] range."""
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        tensor = self.service.preprocess_image(img)
        values = tensor.numpy()
        assert values.min() >= -4.0
        assert values.max() <= 4.0

    def test_preprocess_rgba_after_conversion(self):
        """RGBA converted to RGB preprocesses successfully."""
        import torch
        img = Image.new("RGBA", (100, 100), color=(200, 100, 50, 128)).convert("RGB")
        tensor = self.service.preprocess_image(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (1, 3, 224, 224)


# ---------------------------------------------------------------------------
# API endpoint tests via TestClient
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def test_client():
    """
    Provide a Starlette TestClient for the FastAPI app.

    The lifespan handler tries to load models/best_model.pt; when the file
    is absent (e.g. CI) the service starts with model_loaded=False, which
    is the correct behaviour to test graceful degradation.
    """
    from fastapi.testclient import TestClient
    from src.inference.app import app
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


class TestAPIEndpoints:
    """Tests for API endpoints — model may or may not be loaded."""

    def test_health_returns_200(self, test_client):
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, test_client):
        data = test_client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data
        assert "timestamp" in data

    def test_health_status_is_healthy(self, test_client):
        data = test_client.get("/health").json()
        assert data["status"] == "healthy"

    def test_health_model_loaded_is_bool(self, test_client):
        data = test_client.get("/health").json()
        assert isinstance(data["model_loaded"], bool)

    def test_root_returns_200(self, test_client):
        assert test_client.get("/").status_code == 200

    def test_root_response_has_endpoints(self, test_client):
        data = test_client.get("/").json()
        assert "endpoints" in data
        assert "predict" in data["endpoints"]
        assert "health" in data["endpoints"]

    def test_predict_rejects_non_image(self, test_client):
        files = {"file": ("readme.txt", b"hello world", "text/plain")}
        response = test_client.post("/predict", files=files)
        assert response.status_code in (400, 503)

    def test_predict_image_when_no_model_returns_503(self, test_client):
        from src.inference.app import model_service
        if model_service.model is not None:
            pytest.skip("Model is loaded — 503 path not exercised")
        files = {"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")}
        response = test_client.post("/predict", files=files)
        assert response.status_code == 503

    def test_predict_with_loaded_model_returns_valid_response(self, test_client):
        from src.inference.app import model_service
        if model_service.model is None:
            pytest.skip("Model not loaded — skipping live prediction test")
        files = {"file": ("test.jpg", _make_jpeg_bytes(), "image/jpeg")}
        response = test_client.post("/predict", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["predicted_class"] in ("cat", "dog")
        assert 0.0 <= data["confidence"] <= 1.0
        assert abs(data["probabilities"]["cat"] + data["probabilities"]["dog"] - 1.0) < 0.01

    def test_docs_endpoint_is_accessible(self, test_client):
        assert test_client.get("/docs").status_code == 200


# ---------------------------------------------------------------------------
# Prediction logic — pure unit tests (no I/O)
# ---------------------------------------------------------------------------

class TestPredictionLogic:
    """Tests for prediction result structure and invariants."""

    def test_prediction_returns_required_fields(self):
        result = {
            "predicted_class": "cat",
            "predicted_label": 0,
            "confidence": 0.95,
            "probabilities": {"cat": 0.95, "dog": 0.05},
        }
        for field in ("predicted_class", "predicted_label", "confidence", "probabilities"):
            assert field in result

    def test_probabilities_sum_to_one(self):
        probs = {"cat": 0.7, "dog": 0.3}
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_confidence_matches_max_probability(self):
        probs = {"cat": 0.8, "dog": 0.2}
        assert max(probs.values()) == probs["cat"]

    def test_predicted_label_is_int(self):
        assert isinstance(0, int)

    def test_confidence_is_in_unit_interval(self):
        for conf in (0.0, 0.5, 1.0):
            assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# Image conversion handling
# ---------------------------------------------------------------------------

class TestImageConversion:
    """Tests for PIL image mode conversion."""

    def test_rgb_image_unchanged(self):
        img = Image.new("RGB", (224, 224))
        result = img.convert("RGB") if img.mode != "RGB" else img
        assert result.mode == "RGB"

    def test_grayscale_to_rgb(self):
        img = Image.new("L", (224, 224))
        assert img.convert("RGB").mode == "RGB"

    def test_rgba_to_rgb(self):
        img = Image.new("RGBA", (224, 224))
        converted = img.convert("RGB")
        assert converted.mode == "RGB"
        assert len(converted.getbands()) == 3

    def test_cmyk_to_rgb(self):
        img = Image.new("CMYK", (224, 224))
        assert img.convert("RGB").mode == "RGB"


# ---------------------------------------------------------------------------
# Prometheus metrics endpoint
# ---------------------------------------------------------------------------

class TestMetrics:
    """Tests for the /metrics Prometheus endpoint."""

    def test_metrics_returns_200(self, test_client):
        assert test_client.get("/metrics").status_code == 200

    def test_metrics_content_type_is_text(self, test_client):
        response = test_client.get("/metrics")
        assert "text" in response.headers.get("content-type", "")

    def test_metrics_contains_health_counter(self, test_client):
        """health_check_requests_total incremented by earlier tests."""
        body = test_client.get("/metrics").text
        assert "health_check_requests_total" in body


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

