#!/bin/bash
set -e

PROJECT_DIR="/Users/v0s01jh/Documents/BinaryImageClassification_For_A_Pet_Adoption_Platform"
cd "$PROJECT_DIR"

# ── helper: wait until a local port is accepting TCP connections ──────────────
wait_for_port() {
  local port=$1 label=$2 retries=30
  echo -n "  Waiting for $label (port $port)..."
  for i in $(seq 1 $retries); do
    if nc -z localhost "$port" 2>/dev/null; then
      echo " ready ✅"
      return 0
    fi
    sleep 1
    echo -n "."
  done
  echo " timed out ⚠️  (continuing anyway)"
}

echo "============================================"
echo " Stopping any existing services..."
echo "============================================"
pkill -f mlflow      2>/dev/null || true
pkill -f uvicorn     2>/dev/null || true
pkill -f "port-forward" 2>/dev/null || true
pkill -f push_metrics   2>/dev/null || true
podman stop prometheus grafana 2>/dev/null || true
podman rm   prometheus grafana 2>/dev/null || true
sleep 2

echo ""
echo "============================================"
echo " Starting MLflow on port 5001..."
echo "============================================"
source "$PROJECT_DIR/venv/bin/activate"
nohup mlflow server \
  --host 0.0.0.0 \
  --port 5001 \
  --backend-store-uri "sqlite:///$PROJECT_DIR/mlflow.db" \
  > "$PROJECT_DIR/mlflow.log" 2>&1 &
MLFLOW_PID=$!
echo "MLflow PID: $MLFLOW_PID"
wait_for_port 5001 "MLflow"

echo ""
echo "============================================"
echo " Starting FastAPI on port 8000..."
echo "============================================"
nohup uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload > "$PROJECT_DIR/inference.log" 2>&1 &
API_PID=$!
echo "FastAPI PID: $API_PID"
wait_for_port 8000 "FastAPI"

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
wait_for_port 9090 "Prometheus"

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
wait_for_port 3000 "Grafana"

echo ""
echo "============================================"
echo " Starting Metrics Server on port 8081..."
echo "============================================"
nohup python "$PROJECT_DIR/scripts/push_metrics.py" \
  > "$PROJECT_DIR/metrics.log" 2>&1 &
METRICS_PID=$!
echo "Metrics server PID: $METRICS_PID"
wait_for_port 8081 "Metrics server"

echo ""
echo "============================================"
echo " Importing Grafana Dashboard..."
echo "============================================"
sleep 5
python "$PROJECT_DIR/scripts/setup_grafana.py" || echo "  ⚠️  Grafana dashboard import failed (non-fatal)"

echo ""
echo "============================================"
echo " Starting ArgoCD (Minikube + port-forward)..."
echo "============================================"

# Start Minikube if not already running
if ! minikube status 2>/dev/null | grep -q "Running"; then
  echo "Starting Minikube..."
  minikube start --driver=podman
else
  echo "Minikube already running"
fi

# Apply ArgoCD install if not already present
if ! kubectl get namespace argocd &>/dev/null; then
  echo "Installing ArgoCD..."
  kubectl create namespace argocd
  kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
  echo "Waiting for ArgoCD pods to be ready (up to 3 min)..."
  kubectl wait --for=condition=available deployment/argocd-server \
    -n argocd --timeout=180s
else
  echo "ArgoCD namespace already exists"
fi

# Apply ArgoCD application manifest
kubectl apply -f "$PROJECT_DIR/k8s/argocd-application.yaml" 2>/dev/null || true

# Port-forward ArgoCD UI in background (https://localhost:9443)
pkill -f "port-forward.*argocd-server" 2>/dev/null || true
sleep 1
nohup kubectl port-forward svc/argocd-server -n argocd 9443:443 \
  > "$PROJECT_DIR/argocd-portforward.log" 2>&1 &
ARGOCD_PFW_PID=$!
echo "ArgoCD port-forward PID: $ARGOCD_PFW_PID"
wait_for_port 9443 "ArgoCD UI"

# Print ArgoCD admin password
ARGOCD_PASS=$(kubectl -n argocd get secret argocd-initial-admin-secret \
  -o jsonpath="{.data.password}" 2>/dev/null | base64 -d 2>/dev/null || echo "(already changed or not found)")
echo "ArgoCD admin password: $ARGOCD_PASS"

echo ""
echo "============================================"
echo "  ALL SERVICES RUNNING!"
echo "============================================"
echo ""
echo "  MLflow:     http://localhost:5001"
echo "  FastAPI:    http://localhost:8000"
echo "  API Docs:   http://localhost:8000/docs"
echo "  Prometheus: http://localhost:9090/graph"
echo "  Grafana:    http://localhost:3000/d/pet-adoption-ml-v2  (admin/admin)"
echo "  Metrics:    http://localhost:8081/metrics"
echo "  ArgoCD:     https://localhost:9443  (admin / see above)"
echo "    App:      cats-dogs-classifier"
echo "    GitOps:   auto-syncs k8s/local/ on every git push"
echo ""
echo "  Log files:"
echo "    MLflow:   $PROJECT_DIR/mlflow.log"
echo "    FastAPI:  $PROJECT_DIR/inference.log"
echo "    Metrics:  $PROJECT_DIR/metrics.log"
echo "    ArgoCD:   $PROJECT_DIR/argocd-portforward.log"
echo "    Podman:   podman logs prometheus | podman logs grafana"
echo "============================================"
