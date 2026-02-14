"""
FastAPI Inference Service for Cats vs Dogs Classification.

This service provides REST API endpoints for:
- Health check
- Image classification prediction
- Prometheus metrics
"""

import io
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import structlog
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image
from prometheus_client import (CONTENT_TYPE_LATEST, Counter, Histogram,
                               generate_latest)
from pydantic import BaseModel
from starlette.responses import Response

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer(),
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    "prediction_requests_total",
    "Total number of prediction requests",
    ["class_predicted"],
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Time spent processing prediction requests",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
HEALTH_CHECK_COUNTER = Counter(
    "health_check_requests_total", "Total number of health check requests"
)
ERROR_COUNTER = Counter(
    "prediction_errors_total", "Total number of prediction errors", ["error_type"]
)


class ModelService:
    """Service for loading and running inference with the model."""

    def __init__(self, model_path: str = "models/best_model.pt"):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.class_names = {0: "cat", 1: "dog"}
        self.image_size = 224
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def load_model(self) -> bool:
        """Load the trained model."""
        try:
            # Import model architecture
            import sys

            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from src.models.cnn import get_model

            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Get model configuration
            config = checkpoint.get("config", {})
            model_name = config.get("model_name", "simple_cnn")

            # Initialize model
            self.model = get_model(
                model_name=model_name,
                num_classes=2,
                dropout_rate=config.get("dropout_rate", 0.5),
            )

            # Load weights
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            logger.info(
                "model_loaded",
                model_path=self.model_path,
                device=str(self.device),
                model_name=model_name,
            )
            return True

        except Exception as e:
            logger.error("model_load_failed", error=str(e))
            return False

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for inference."""
        # Resize
        image = image.resize(
            (self.image_size, self.image_size), Image.Resampling.LANCZOS
        )

        # Convert to numpy array
        img_array = np.array(image).astype(np.float32) / 255.0

        # Normalize
        for i in range(3):
            img_array[:, :, i] = (img_array[:, :, i] - self.mean[i]) / self.std[i]

        # Convert to tensor (HWC -> CHW)
        img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))

        # Add batch dimension
        return img_tensor.unsqueeze(0).to(self.device)

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Run prediction on an image.

        Args:
            image: PIL Image to classify

        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess
        input_tensor = self.preprocess_image(image)

        # Inference
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = F.softmax(outputs, dim=1)

        # Get prediction
        probs = probabilities.cpu().numpy()[0]
        predicted_class = int(np.argmax(probs))
        confidence = float(probs[predicted_class])

        return {
            "predicted_class": self.class_names[predicted_class],
            "predicted_label": predicted_class,
            "confidence": confidence,
            "probabilities": {
                self.class_names[0]: float(probs[0]),
                self.class_names[1]: float(probs[1]),
            },
        }


# Global model service
model_service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("application_starting")

    # Load model
    model_path = os.environ.get("MODEL_PATH", "models/best_model.pt")
    model_service.model_path = model_path

    if Path(model_path).exists():
        success = model_service.load_model()
        if not success:
            logger.warning("model_not_loaded", reason="Load failed")
    else:
        logger.warning("model_not_loaded", reason="Model file not found")

    logger.info("application_started")

    yield

    # Shutdown
    logger.info("application_shutting_down")


# Create FastAPI app
app = FastAPI(
    title="Cats vs Dogs Classifier API",
    description="REST API for binary image classification (Cats vs Dogs)",
    version="1.0.0",
    lifespan=lifespan,
)


# Request/Response models
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_label: int
    confidence: float
    probabilities: Dict[str, float]
    processing_time_ms: float


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None


# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()

    # Process request
    response = await call_next(request)

    # Calculate processing time
    process_time = time.time() - start_time

    # Log request (excluding sensitive data)
    logger.info(
        "request_processed",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        processing_time_ms=round(process_time * 1000, 2),
        client_host=request.client.host if request.client else None,
    )

    return response


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Returns the service status, model loading state, and device information.
    """
    HEALTH_CHECK_COUNTER.inc()

    return HealthResponse(
        status="healthy",
        model_loaded=model_service.model is not None,
        device=str(model_service.device),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image"},
        500: {"model": ErrorResponse, "description": "Internal error"},
    },
    tags=["Prediction"],
)
async def predict(file: UploadFile = File(...)):
    """
    Predict whether an image contains a cat or a dog.

    - **file**: Image file (JPEG, PNG, etc.)

    Returns the predicted class, confidence score, and class probabilities.
    """
    start_time = time.time()

    # Check if model is loaded
    if model_service.model is None:
        ERROR_COUNTER.labels(error_type="model_not_loaded").inc()
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please try again later."
        )

    # Validate file type
    content_type = file.content_type
    if not content_type or not content_type.startswith("image/"):
        ERROR_COUNTER.labels(error_type="invalid_content_type").inc()
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {content_type}. Expected image file.",
        )

    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Run prediction
        with PREDICTION_LATENCY.time():
            result = model_service.predict(image)

        # Update metrics
        PREDICTION_COUNTER.labels(class_predicted=result["predicted_class"]).inc()

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000

        logger.info(
            "prediction_completed",
            predicted_class=result["predicted_class"],
            confidence=result["confidence"],
            processing_time_ms=round(processing_time, 2),
        )

        return PredictionResponse(
            predicted_class=result["predicted_class"],
            predicted_label=result["predicted_label"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
            processing_time_ms=round(processing_time, 2),
        )

    except Exception as e:
        ERROR_COUNTER.labels(error_type="prediction_failed").inc()
        logger.error("prediction_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint.

    Returns metrics in Prometheus text format.
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", tags=["Info"])
async def root():
    """API root - returns basic information."""
    return {
        "name": "Cats vs Dogs Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "metrics": "/metrics",
            "artifacts": "/artifacts/compare",
            "docs": "/docs",
        },
    }


class ArtifactInfo(BaseModel):
    """Artifact information model."""

    image: str
    image_id: str
    status: str
    created: Optional[str] = None
    size: Optional[str] = None
    digest: Optional[str] = None


class ArtifactComparisonResponse(BaseModel):
    """Artifact comparison response model."""

    local: ArtifactInfo
    registry: ArtifactInfo
    comparison: Dict[str, Any]
    timestamp: str


@app.get("/artifacts/compare", response_model=ArtifactComparisonResponse, tags=["Artifacts"])
async def compare_artifacts():
    """
    Compare local running artifact with GitHub Container Registry artifact.
    
    Returns detailed information about both artifacts and whether they match.
    """
    # Get local container info
    local_info = {
        "image": "unknown",
        "image_id": "unknown",
        "status": "unknown",
        "created": None,
        "size": None,
        "digest": None,
    }
    
    # Try to get info from environment variables (passed at runtime or build time)
    container_image = os.environ.get("CONTAINER_IMAGE", "localhost/cats-dogs-classifier:latest")
    build_timestamp = os.environ.get("BUILD_TIMESTAMP", datetime.utcnow().isoformat())
    git_sha = os.environ.get("GIT_SHA", "unknown")
    local_image_id = os.environ.get("LOCAL_IMAGE_ID", "unknown")
    image_tag = os.environ.get("IMAGE_TAG", "latest")
    
    # Populate local info from environment
    local_info["image"] = container_image
    local_info["image_id"] = local_image_id[:12] if local_image_id and local_image_id != "unknown" else "unknown"
    local_info["created"] = build_timestamp
    local_info["status"] = "running (in-container)"
    
    # Get registry info
    registry_image = os.environ.get(
        "REGISTRY_IMAGE",
        "ghcr.io/vishalvishal099/binaryimageclassification_for_a_pet_adoption_platform:latest"
    )
    registry_image_id = os.environ.get("REGISTRY_IMAGE_ID", "N/A")
    
    registry_info = {
        "image": registry_image,
        "image_id": registry_image_id[:12] if registry_image_id and registry_image_id != "N/A" else "N/A",
        "status": "available",
        "created": None,
        "size": None,
        "digest": None,
    }
    
    # Compare artifacts
    same_image = local_info["image_id"] == registry_info["image_id"] and local_info["image_id"] != "unknown"
    same_digest = (
        local_info["digest"] == registry_info["digest"] 
        and local_info["digest"] is not None
    )
    
    comparison = {
        "match": same_image or same_digest,
        "local_id": local_info["image_id"],
        "registry_id": registry_info["image_id"],
        "git_sha": git_sha,
        "recommendation": (
            "✅ Artifacts are in sync!"
            if (same_image or same_digest)
            else "⚠️ Artifacts differ - consider updating local or pushing new build"
        ),
    }
    
    logger.info(
        "artifact_comparison",
        local_id=local_info["image_id"],
        registry_id=registry_info["image_id"],
        match=comparison["match"],
    )
    
    return ArtifactComparisonResponse(
        local=ArtifactInfo(**local_info),
        registry=ArtifactInfo(**registry_info),
        comparison=comparison,
        timestamp=datetime.utcnow().isoformat(),
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False, log_level="info")
