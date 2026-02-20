# 🎬 Video Recording Guide: MLOps Project Demo

## Overview
This guide provides step-by-step instructions for recording a comprehensive demo video covering all MLOps milestones (M1-M5) for the Binary Image Classification project.

**Estimated Video Length:** 15-20 minutes  
**Target Audience:** Project evaluators / reviewers

---

## 📋 Pre-Recording Checklist

Before starting the recording, ensure:

- [ ] Terminal is open and in project directory
- [ ] VS Code is open with the project
- [ ] Docker/Podman is running
- [ ] Browser tabs ready for:
  - MLflow UI (http://localhost:5000)
  - FastAPI Swagger (http://localhost:8000/docs)
  - Prometheus (http://localhost:9090) - if running
  - Grafana (http://localhost:3000) - if running
  - GitHub repo Actions tab
- [ ] Sample cat/dog images ready for prediction demo
- [ ] Screen recording software configured (1080p recommended)

---

## 🎥 Recording Script

### **Introduction (30 seconds)**

```
"Hello! This is a demo of my MLOps Binary Image Classification project 
for a Pet Adoption Platform. I'll walk through all five milestones: 
Model Development, Containerization, CI Pipeline, CD Pipeline, and Monitoring."
```

---

## **M1: Model Development & Experiment Tracking (4-5 minutes)**

### 1.1 Show Project Structure
```bash
# Show overall project structure
tree -L 2 -I 'venv|__pycache__|data' .

# Key directories to highlight:
# - src/          → Source code (models, training, inference)
# - configs/      → Training configuration
# - notebooks/    → EDA notebook
# - tests/        → Unit tests
```

### 1.2 Data Version Control (DVC)
```bash
# Show DVC configuration
cat dvc.yaml

# Show DVC pipeline stages
dvc dag

# Show data tracking
cat data/raw.dvc

# Show DVC remote configuration
cat .dvc/config
```

**Talking Points:**
- "DVC tracks our dataset with pointer files committed to git"
- "The dvc.yaml defines our ML pipeline: preprocess → train → evaluate"
- "Data is versioned and can be pulled with `dvc pull`"

### 1.3 MLflow Experiment Tracking
```bash
# Start MLflow UI
mlflow ui --port 5000 &

# Open browser to http://localhost:5000
```

**In MLflow UI, show:**
- Experiment list
- Best run details (parameters, metrics)
- Artifacts: `best_model.pt`, `loss_curves.png`, `confusion_matrix.npy`
- Metric comparison charts

```bash
# Show metrics files
cat models/metrics.json
cat reports/evaluation_metrics.json
```

### 1.4 Show Training Code
```bash
# Briefly show training script structure
head -50 src/training/train.py

# Show model architecture
head -80 src/models/cnn.py
```

**Talking Points:**
- "Training logs parameters, metrics per epoch, and artifacts to MLflow"
- "Loss curves and confusion matrix are auto-generated"

---

## **M2: Model Packaging & Containerization (3-4 minutes)**

### 2.1 Show Dockerfile
```bash
# Display Dockerfile
cat Dockerfile
```

**Talking Points:**
- "Multi-stage build: builder stage installs dependencies, production stage is minimal"
- "Non-root user for security"
- "Health check endpoint configured"

### 2.2 Show Requirements
```bash
# Show pinned dependencies
cat requirements.txt

# Show dev dependencies
cat requirements-dev.txt
```

**Talking Points:**
- "All versions are pinned for reproducibility"
- "Dev dependencies separated for lighter production image"

### 2.3 Build and Run Container
```bash
# Build the image
docker build -t cats-dogs-classifier:demo .

# Run the container
docker run -d -p 8000:8000 --name demo-container cats-dogs-classifier:demo

# Check it's running
docker ps

# Test health endpoint
curl http://localhost:8000/health | jq
```

### 2.4 Show FastAPI Service
**Open browser to http://localhost:8000/docs**

**Demonstrate:**
- `/health` - Health check endpoint
- `/predict` - Upload a sample image and show prediction
- `/metrics` - Prometheus metrics endpoint

```bash
# Test prediction from command line
curl -X POST http://localhost:8000/predict \
  -F "file=@data/raw/cats/1.jpg" | jq
```

### 2.5 Cleanup
```bash
docker stop demo-container && docker rm demo-container
```

---

## **M3: CI Pipeline (3-4 minutes)**

### 3.1 Show CI Workflow
```bash
# Display CI configuration
cat .github/workflows/ci.yml
```

**Talking Points:**
- "Triggered on push to main/develop and PRs"
- "5 jobs: lint → test → build → integration-test → security-scan"

### 3.2 Show GitHub Actions (Browser)
**Navigate to:** `https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform/actions`

**Show:**
- Recent CI runs
- Click into a successful run
- Show each job's logs briefly:
  - **Lint**: black, isort, flake8 checks
  - **Test**: pytest with coverage
  - **Build**: Docker image built and pushed to GHCR
  - **Integration Test**: Container started, endpoints tested
  - **Security Scan**: Trivy vulnerability scan

### 3.3 Show Test Suite
```bash
# Run tests locally
pytest tests/ -v --tb=short

# Show test files
ls tests/

# Show coverage
pytest tests/ --cov=src --cov-report=term-missing
```

### 3.4 Show Container Registry
**Navigate to:** `https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform/pkgs/container/binaryimageclassification_for_a_pet_adoption_platform`

**Show:**
- Published image tags
- Image metadata

---

## **M4: CD Pipeline & Deployment (3-4 minutes)**

### 4.1 Show CD Workflow
```bash
# Display CD configuration
cat .github/workflows/cd.yml
```

**Talking Points:**
- "Triggered automatically when CI succeeds (workflow_run)"
- "Jobs: deploy → smoke-tests → notify"
- "Uses K8s manifests from k8s/local/"

### 4.2 Show Kubernetes Manifests
```bash
# Show K8s manifests
ls k8s/local/

# Show deployment
cat k8s/local/deployment.yaml

# Show service
cat k8s/local/service.yaml
```

**Talking Points:**
- "Deployment with resource limits and rolling update strategy"
- "Service exposes the API on port 8000"
- "ConfigMap for environment configuration"

### 4.3 Show ArgoCD Configuration (GitOps)
```bash
# Show ArgoCD application manifest
cat k8s/argocd-application.yaml
```

**Talking Points:**
- "ArgoCD watches the git repo and auto-syncs to cluster"
- "GitOps approach: git is the source of truth"

### 4.4 Show Smoke Tests
```bash
# Show smoke test script
cat scripts/smoke_test.py
```

**Talking Points:**
- "Post-deployment smoke tests verify health, predict, and metrics endpoints"
- "Fails the CD pipeline if service is unhealthy"

### 4.5 Local K8s Demo (Optional - if K8s available)
```bash
# If minikube/kind is running:
kubectl apply -f k8s/local/
kubectl get pods -n cats-dogs-classifier
kubectl get svc -n cats-dogs-classifier
```

---

## **M5: Monitoring, Logging & Performance Tracking (3-4 minutes)**

### 5.1 Show Monitoring Configuration
```bash
# Show Prometheus config
cat monitoring/prometheus.yml

# Show Grafana datasource
cat monitoring/grafana/provisioning/datasources/datasource.yml
```

### 5.2 Show Metrics in Application
```bash
# Start the service locally
./run_local.sh

# Or if already running:
curl http://localhost:8000/metrics | head -50
```

**Talking Points:**
- "Prometheus metrics exposed at /metrics"
- "Tracks: request count, latency histogram, prediction distribution"

### 5.3 Show Performance Tracker
```bash
# Show performance tracking code
cat src/utils/performance_tracker.py | head -60

# Show how it's used in the app
grep -A5 "performance_tracker" src/inference/app.py | head -20
```

**Talking Points:**
- "PerformanceTracker records inference times, model accuracy"
- "Integrated with structlog for structured JSON logging"

### 5.4 Show Structured Logging
```bash
# Show structlog configuration in app
grep -A10 "structlog" src/inference/app.py

# Make a prediction and show logs
curl -X POST http://localhost:8000/predict -F "file=@data/raw/cats/1.jpg"
tail -5 inference.log
```

### 5.5 Show Simulation Script
```bash
# Show traffic simulation for monitoring
cat scripts/simulate_performance.py | head -50
```

**Talking Points:**
- "Can simulate traffic to generate metrics for dashboards"
- "Useful for load testing and dashboard development"

### 5.6 Grafana Dashboard (Optional - if running)
**Open browser to http://localhost:3000**

**Show:**
- MLOps dashboard with request rate, latency, prediction distribution
- System metrics panels

---

## **Conclusion (30-60 seconds)**

### Show Final Summary
```bash
# Final project stats
echo "=== Project Summary ==="
echo "Total Python files: $(find src tests -name '*.py' | wc -l)"
echo "Test count: $(pytest tests/ --collect-only -q 2>/dev/null | tail -1)"
echo "Docker image size: $(docker images cats-dogs-classifier --format '{{.Size}}')"

# Show git log
git log --oneline -10
```

**Talking Points:**
```
"To summarize what we've covered:

M1 - Model Development: DVC for data versioning, MLflow for experiment tracking,
     loss curves and metrics automatically logged.

M2 - Containerization: Multi-stage Dockerfile, pinned dependencies, 
     FastAPI inference service with health checks.

M3 - CI Pipeline: Automated linting, testing, Docker build, 
     integration tests, and security scanning via GitHub Actions.

M4 - CD Pipeline: Automated deployment to Kubernetes triggered by CI success,
     post-deployment smoke tests, GitOps with ArgoCD support.

M5 - Monitoring: Prometheus metrics, structured logging with structlog,
     performance tracking, Grafana dashboard ready.

Thank you for watching!"
```

---

## 📝 Recording Tips

1. **Pace**: Speak slowly and clearly, pause between sections
2. **Cursor**: Make cursor movements deliberate, highlight key areas
3. **Errors**: If something fails, explain and troubleshoot - shows real-world skills
4. **Terminal**: Use a large font (14pt+) for readability
5. **Browser**: Zoom to 125% for visibility
6. **Clean Desktop**: Close unnecessary windows before recording

---

## 🔧 Quick Commands Reference

```bash
# Start all services
./run_local.sh

# Stop all services
pkill -f 'mlflow ui' && pkill -f 'uvicorn'

# Run tests
pytest tests/ -v

# Build Docker image
docker build -t cats-dogs-classifier .

# Check CI status
gh run list --limit 5

# View CI logs
gh run view <run-id> --log
```

---

## 📁 Files to Highlight

| Milestone | Key Files |
|-----------|-----------|
| M1 | `dvc.yaml`, `mlflow.db`, `src/training/train.py`, `models/metrics.json` |
| M2 | `Dockerfile`, `requirements.txt`, `src/inference/app.py` |
| M3 | `.github/workflows/ci.yml`, `tests/`, `pytest.ini` |
| M4 | `.github/workflows/cd.yml`, `k8s/local/`, `scripts/smoke_test.py` |
| M5 | `monitoring/`, `src/utils/performance_tracker.py`, `/metrics` endpoint |
