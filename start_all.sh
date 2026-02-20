#!/bin/bash
set -e

PROJECT_DIR="/Users/v0s01jh/Documents/BinaryImageClassification_For_A_Pet_Adoption_Platform"
cd "$PROJECT_DIR"

echo "============================================"
echo " Stopping any existing services..."
echo "============================================"
pkill -f mlflow 2>/dev/null || true
pkill -f uvicorn 2>/dev/null || true
pkill -f "port-forward" 2>/dev/null || true
podman stop prometheus grafana 2>/dev/null || true
podman rm prometheus grafana 2>/dev/null || true
sleep 2

echo ""
echo "============================================"
echo " Starting MLflow on port 5001..."
echo "============================================"
source "$PROJECT_DIR/venv/bin/activate"
nohup mlflow server --host 0.0.0.0 --port 5001 > "$PROJECT_DIR/mlflow.log" 2>&1 &
MLFLOW_PID=$!
echo "MLflow PID: $MLFLOW_PID"
sleep 3

echo ""
echo "============================================"
echo " Starting FastAPI on port 8000..."
echo "============================================"
nohup uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload > "$PROJECT_DIR/inference.log" 2>&1 &
API_PID=$!
echo "FastAPI PID: $API_PID"
sleep 3

echo ""
echo "============================================"
echo " Starting Prometheus on port 9090..."
echo "============================================"
podman run -d \
  --name prometheus \
  -p 9090:9090 \
  -v "$PROJECT_DIR/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
  prom/prometheus:latest \
  --config.file=/etc/prometheus/prometheus.yml \
  --web.enable-lifecycle
echo "Prometheus started"
sleep 3

echo ""
echo "============================================"
echo " Starting Grafana on port 3000..."
echo "============================================"
podman run -d \
  --name grafana \
  -p 3000:3000 \
  -e GF_SECURITY_ADMIN_USER=admin \
  -e GF_SECURITY_ADMIN_PASSWORD=admin \
  -e GF_USERS_ALLOW_SIGN_UP=false \
  grafana/grafana:latest
echo "Grafana started"
sleep 3

echo ""
echo "============================================"
echo " Pushing sample metrics to Prometheus..."
echo "============================================"
python "$PROJECT_DIR/scripts/push_metrics.py" &
echo "Metrics pusher started"

echo ""
echo "============================================"
echo " Importing Grafana Dashboard..."
echo "============================================"
sleep 5
python "$PROJECT_DIR/scripts/setup_grafana.py"

echo ""
echo "============================================"
echo "  ALL SERVICES RUNNING!"
echo "============================================"
echo ""
echo "  MLflow:     http://localhost:5001"
echo "  FastAPI:    http://localhost:8000"
echo "  API Docs:   http://localhost:8000/docs"
echo "  Prometheus: http://localhost:9090"
echo "  Grafana:    http://localhost:3000  (admin/admin)"
echo ""
echo "  ArgoCD: Run 'minikube start --driver=podman' first,"
echo "          then 'kubectl port-forward svc/argocd-server -n argocd 9443:443'"
echo "============================================"
