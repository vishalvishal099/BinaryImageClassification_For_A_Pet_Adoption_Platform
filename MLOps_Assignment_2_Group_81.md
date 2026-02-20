# 📋 MLOps Assignment 2 — Group 81

**Project:** Binary Image Classification for a Pet Adoption Platform  
**Repository:** [BinaryImageClassification_For_A_Pet_Adoption_Platform](https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform)  
**Date:** February 21, 2026

---

## 📌 Table of Contents

- [M1 — Source Code](#m1--source-code)
- [M2 — CI/CD Pipelines](#m2--cicd-pipelines)
- [M3 — Container Registry](#m3--container-registry)
- [M4 — Experiment Tracking & Data Versioning (Dagshub)](#m4--experiment-tracking--data-versioning-dagshub)
- [M5 — Local Services](#m5--local-services)
- [Documentation Links](#documentation-links)

---

## M1 — Source Code

| Component | GitHub Link |
|-----------|-------------|
| **Repository (main)** | https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform |
| **Model Training** | https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform/blob/main/src/training/train.py |
| **FastAPI Inference App** | https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform/blob/main/src/api/main.py |
| **Preprocessing** | https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform/blob/main/src/data/preprocess.py |
| **DVC Pipeline Config** | https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform/blob/main/dvc.yaml |
| **DVC Remote Config** | https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform/blob/main/.dvc/config |
| **Kubernetes Manifests** | https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform/tree/main/k8s/local |
| **ArgoCD Application** | https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform/blob/main/k8s/argocd-application.yaml |
| **Prometheus Config** | https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform/blob/main/monitoring/prometheus.yml |
| **Metrics Server** | https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform/blob/main/scripts/push_metrics.py |
| **start_all.sh** | https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform/blob/main/start_all.sh |
| **Unit Tests** | https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform/tree/main/tests |

---

## M2 — CI/CD Pipelines

### GitHub Actions — CI Pipeline

| Item | Link |
|------|------|
| **CI Workflow File** | https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform/blob/main/.github/workflows/ci.yml |
| **CI Pipeline Runs** | https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform/actions/workflows/ci.yml |

**CI Jobs:**
1. **Lint & Code Quality** — black, isort, flake8, mypy
2. **Unit Tests** — pytest (35 tests, coverage report → Codecov)
3. **Build Docker Image** — multi-stage build, push to GHCR
4. **Integration Tests** — spin up container, test all endpoints
5. **Security Scan** — Trivy vulnerability scan → GitHub Security tab

---

### GitHub Actions — CD Pipeline (GitOps)

| Item | Link |
|------|------|
| **CD Workflow File** | https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform/blob/main/.github/workflows/cd.yml |
| **CD Pipeline Runs** | https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform/actions/workflows/cd.yml |

**CD GitOps Flow:**
```
git push → CI Pipeline passes
    └──▶ CD: update k8s/local/deployment.yaml with new image SHA
         └──▶ git commit [skip ci] → push to main
              └──▶ ArgoCD detects change → auto-syncs to Minikube cluster
```

**CD Jobs:**
1. **update-manifest** — updates image tag in `k8s/local/deployment.yaml`, commits & pushes `[skip ci]`
2. **smoke-tests** — health check, predict endpoint, metrics endpoint
3. **notify** — reports pipeline success/failure

---

## M3 — Container Registry

| Item | Link |
|------|------|
| **GHCR Package** | https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform/pkgs/container/binaryimageclassification_for_a_pet_adoption_platform |
| **Docker Pull Command** | `docker pull ghcr.io/vishalvishal099/binaryimageclassification_for_a_pet_adoption_platform:latest` |
| **Dockerfile** | https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform/blob/main/Dockerfile |

---

## M4 — Experiment Tracking & Data Versioning (Dagshub)

| Item | Link |
|------|------|
| **Dagshub Repository** | https://dagshub.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform |
| **MLflow Experiment Tracking (Dagshub)** | https://dagshub.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform.mlflow |
| **DVC Data Remote (Dagshub)** | https://dagshub.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform.dvc |

**MLflow Details:**
- Model registered as: `CatsDogsClassifier` → **Production** stage
- Artifacts logged: `best_model.pt`, `loss_curves.png`, `confusion_matrix.npy`
- Local tracking server: `http://localhost:5001`

**DVC Details:**
- Pull data: `dvc pull`
- Run pipeline: `dvc repro`
- Pipeline stages: `preprocess → train → evaluate`

---

## M5 — Local Services

All services are started with `./start_all.sh`

| Service | Local URL | Description |
|---------|-----------|-------------|
| **MLflow UI** | http://localhost:5001 | Experiment tracking & model registry |
| **FastAPI (Inference)** | http://localhost:8000 | REST API — `/predict`, `/health`, `/metrics` |
| **FastAPI Docs (Swagger)** | http://localhost:8000/docs | Interactive API documentation |
| **FastAPI Health** | http://localhost:8000/health | Health check endpoint |
| **FastAPI Metrics** | http://localhost:8000/metrics | Prometheus scrape endpoint |
| **Prometheus** | http://localhost:9090/graph | Metrics collection & querying |
| **Grafana Dashboard** | http://localhost:3000/d/pet-adoption-ml-v2 | ML observability dashboard |
| **Grafana Home** | http://localhost:3000 | (admin / admin) |
| **Metrics Server** | http://localhost:8081/metrics | 31 metric families, 60+ time series |
| **ArgoCD UI** | https://localhost:9443 | GitOps CD controller (admin / see `start_all.sh` output) |

---

## Documentation Links

| Document | GitHub Link | Description |
|----------|-------------|-------------|
| **README** | https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform/blob/main/README.md | Project overview, quick start, GitOps flow |
| **Architecture Diagram** | https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform/blob/main/docs/ARCHITECTURE_DIAGRAM.md | Full CI/CD & MLOps architecture with ASCII diagrams |
| **Setup Guide** | https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform/blob/main/docs/SETUP_GUIDE.md | Step-by-step local setup, K8s, ArgoCD, monitoring |
| **Documentation** | https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform/blob/main/docs/DOCUMENTATION.md | Detailed technical documentation |
| **Project Status** | https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform/blob/main/PROJECT_STATUS.md | Current status of all components |
| **start_all.sh** | https://github.com/vishalvishal099/BinaryImageClassification_For_A_Pet_Adoption_Platform/blob/main/start_all.sh | Single script to start all services |

---

## 🗂️ Repository Structure Summary

```
BinaryImageClassification_For_A_Pet_Adoption_Platform/
├── .github/workflows/
│   ├── ci.yml                   ← CI Pipeline (lint → test → build → push → scan)
│   └── cd.yml                   ← CD Pipeline (GitOps manifest update → ArgoCD sync)
├── src/
│   ├── training/train.py         ← Model training (SimpleCNN, PyTorch)
│   ├── api/main.py               ← FastAPI inference service
│   └── data/preprocess.py        ← Data preprocessing
├── k8s/local/
│   ├── deployment.yaml           ← K8s deployment (image tag auto-updated by CD)
│   ├── service.yaml
│   ├── configmap.yaml
│   └── namespace.yaml
├── k8s/argocd-application.yaml   ← ArgoCD GitOps app (watches k8s/local/)
├── monitoring/prometheus.yml     ← Prometheus scrape config
├── scripts/push_metrics.py       ← Metrics server (31 families, port 8081)
├── dvc.yaml                      ← DVC pipeline stages
├── .dvc/config                   ← DVC remote → Dagshub
├── start_all.sh                  ← Start all services (MLflow, FastAPI, Prometheus, Grafana, ArgoCD)
├── Dockerfile                    ← Multi-stage Docker build
├── docs/
│   ├── ARCHITECTURE_DIAGRAM.md
│   ├── SETUP_GUIDE.md
│   └── DOCUMENTATION.md
└── PROJECT_STATUS.md
```

---

*Generated: February 21, 2026 | Group 81*
