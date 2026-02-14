#!/bin/bash
# =============================================================================
# Run Container with Artifact Metadata
# =============================================================================
# This script starts the cats-dogs-classifier container with all build metadata
# properly configured for artifact comparison.

set -e

# Configuration
CONTAINER_NAME="${CONTAINER_NAME:-cats-dogs-api}"
IMAGE_NAME="${IMAGE_NAME:-cats-dogs-classifier:latest}"
PORT="${PORT:-8000}"
REGISTRY_IMAGE="${REGISTRY_IMAGE:-ghcr.io/vishalvishal099/binaryimageclassification_for_a_pet_adoption_platform:latest}"

# Detect container runtime
if command -v podman &> /dev/null; then
    RUNTIME="podman"
elif command -v docker &> /dev/null; then
    RUNTIME="docker"
else
    echo "❌ Error: Neither podman nor docker found!"
    exit 1
fi

echo "🐳 Using container runtime: $RUNTIME"

# Stop existing container if running
echo "🛑 Stopping existing container..."
$RUNTIME stop $CONTAINER_NAME 2>/dev/null || true
$RUNTIME rm $CONTAINER_NAME 2>/dev/null || true

# Get local image ID
LOCAL_IMAGE_ID=$($RUNTIME images $IMAGE_NAME --format "{{.ID}}" | head -1)
echo "📦 Local Image ID: $LOCAL_IMAGE_ID"

# Try to get registry image ID (if pulled)
REGISTRY_IMAGE_ID=$($RUNTIME images ${REGISTRY_IMAGE} --format "{{.ID}}" | head -1 2>/dev/null || echo "N/A")
if [ -n "$REGISTRY_IMAGE_ID" ] && [ "$REGISTRY_IMAGE_ID" != "N/A" ]; then
    echo "🌐 Registry Image ID: $REGISTRY_IMAGE_ID"
else
    echo "🌐 Registry Image ID: Not pulled locally"
    REGISTRY_IMAGE_ID="N/A"
fi

# Get git info
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
echo "📝 Git SHA: $GIT_SHA"

# Run container with metadata
echo ""
echo "🚀 Starting container $CONTAINER_NAME on port $PORT..."
$RUNTIME run -d \
    --name $CONTAINER_NAME \
    -p $PORT:8000 \
    -e LOCAL_IMAGE_ID="$LOCAL_IMAGE_ID" \
    -e REGISTRY_IMAGE_ID="$REGISTRY_IMAGE_ID" \
    -e REGISTRY_IMAGE="$REGISTRY_IMAGE" \
    -e CONTAINER_IMAGE="localhost/$IMAGE_NAME" \
    -e GIT_SHA="$GIT_SHA" \
    $IMAGE_NAME

# Wait for container to be ready
echo "⏳ Waiting for container to be ready..."
sleep 3

# Health check
echo ""
echo "🏥 Health check..."
curl -s http://localhost:$PORT/health | python3 -m json.tool 2>/dev/null || echo "Health check failed"

echo ""
echo "✅ Container started successfully!"
echo ""
echo "📍 Endpoints:"
echo "   - Health:    http://localhost:$PORT/health"
echo "   - Predict:   http://localhost:$PORT/predict"
echo "   - Metrics:   http://localhost:$PORT/metrics"
echo "   - Artifacts: http://localhost:$PORT/artifacts/compare"
echo "   - Docs:      http://localhost:$PORT/docs"
