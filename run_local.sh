#!/bin/bash
# =============================================================================
# Run MLOps Pipeline Locally (No Docker Required)
# =============================================================================

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}MLOps Pipeline - Local Execution${NC}"
echo -e "${BLUE}======================================${NC}"

# Activate virtual environment
echo -e "${GREEN}✓ Activating virtual environment${NC}"
source venv/bin/activate

# Set environment variables
export MODEL_PATH="models/best_model.pt"
export MLFLOW_TRACKING_URI="http://localhost:5000"

# Function to check if a port is in use
port_in_use() {
    lsof -i :$1 > /dev/null 2>&1
}

# Start MLflow UI
echo -e "${GREEN}✓ Starting MLflow UI on port 5000${NC}"
if port_in_use 5000; then
    echo "  MLflow already running on port 5000"
else
    nohup mlflow ui --port 5000 > mlflow.log 2>&1 &
    echo "  MLflow UI: http://localhost:5000"
fi

# Start FastAPI inference service
echo -e "${GREEN}✓ Starting Inference Service on port 8000${NC}"
if port_in_use 8000; then
    echo "  Inference service already running on port 8000"
else
    nohup uvicorn src.inference.app:app --host 0.0.0.0 --port 8000 > inference.log 2>&1 &
    echo "  Inference API: http://localhost:8000"
    echo "  API Docs: http://localhost:8000/docs"
fi

# Wait for services to start
sleep 3

# Test the services
echo -e "${GREEN}✓ Testing services${NC}"
curl -s http://localhost:8000/health | python3 -m json.tool || echo "  Inference service starting..."
echo ""

echo -e "${BLUE}======================================${NC}"
echo -e "${GREEN}Services Running:${NC}"
echo -e "  - MLflow UI:        http://localhost:5000"
echo -e "  - Inference API:    http://localhost:8000"
echo -e "  - API Docs:         http://localhost:8000/docs"
echo -e "  - Health Check:     http://localhost:8000/health"
echo -e "  - Metrics:          http://localhost:8000/metrics"
echo -e "${BLUE}======================================${NC}"
echo ""
echo -e "Logs:"
echo -e "  - MLflow:    tail -f mlflow.log"
echo -e "  - Inference: tail -f inference.log"
echo ""
echo -e "To stop services:"
echo -e "  pkill -f 'mlflow ui'"
echo -e "  pkill -f 'uvicorn src.inference.app'"
