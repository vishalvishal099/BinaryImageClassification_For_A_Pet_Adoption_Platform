# Binary Image Classification MLOps Pipeline
## Cats vs Dogs Classification for Pet Adoption Platform

This project implements an end-to-end MLOps pipeline for binary image classification (Cats vs Dogs) using open-source tools. The pipeline covers model development, experiment tracking, containerization, CI/CD, and monitoring.

## 🏗️ Project Structure

```
├── .github/
│   └── workflows/           # GitHub Actions CI/CD pipelines
├── data/
│   ├── raw/                 # Original dataset
│   ├── processed/           # Preprocessed 224x224 images
│   └── .gitkeep
├── src/
│   ├── data/                # Data processing scripts
│   ├── models/              # Model architectures
│   ├── training/            # Training scripts
│   ├── inference/           # Inference service
│   └── utils/               # Utility functions
├── tests/                   # Unit and integration tests
├── notebooks/               # Jupyter notebooks for exploration
├── configs/                 # Configuration files
├── models/                  # Saved model artifacts
├── mlruns/                  # MLflow tracking
├── k8s/                     # Kubernetes manifests
├── monitoring/              # Prometheus/Grafana configs
├── Dockerfile               # Container definition
├── docker-compose.yml       # Local deployment
├── requirements.txt         # Python dependencies
├── dvc.yaml                 # DVC pipeline
└── .dvc/                    # DVC configuration
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Git & DVC
- Kubernetes (minikube/kind) - optional

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd BinaryImageClassification_For_A_Pet_Adoption_Platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize DVC
dvc init
dvc pull  # Pull data from remote storage
```

### Data Preparation

```bash
# Download dataset from Kaggle
kaggle datasets download -d bhavikjikadara/dog-and-cat-classification-dataset
unzip dog-and-cat-classification-dataset.zip -d data/raw/

# Run preprocessing pipeline
python src/data/preprocess.py

# Version data with DVC
dvc add data/processed
git add data/processed.dvc
git commit -m "Add processed dataset"
```

### Model Training

```bash
# Train the model with MLflow tracking
python src/training/train.py --config configs/train_config.yaml

# View experiments in MLflow UI
mlflow ui --port 5000
```

### Running the Inference Service

```bash
# Using Docker
docker build -t cats-dogs-classifier:latest .
docker run -p 8000:8000 cats-dogs-classifier:latest

# Using Docker Compose
docker-compose up -d

# Test the API
curl http://localhost:8000/health
curl -X POST -F "file=@test_image.jpg" http://localhost:8000/predict
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📋 Milestones

### M1: Model Development & Experiment Tracking
- [x] Git for source code versioning
- [x] DVC for dataset versioning
- [x] Baseline CNN model implementation
- [x] MLflow experiment tracking

### M2: Model Packaging & Containerization
- [x] FastAPI inference service
- [x] Health check and prediction endpoints
- [x] requirements.txt with pinned versions
- [x] Dockerfile for containerization

### M3: CI Pipeline for Build, Test & Image Creation
- [x] Unit tests with pytest
- [x] GitHub Actions CI pipeline
- [x] Docker image publishing to registry

### M4: CD Pipeline & Deployment
- [x] Kubernetes manifests (Deployment + Service)
- [x] Docker Compose configuration
- [x] ArgoCD/GitHub Actions CD flow
- [x] Post-deploy smoke tests

### M5: Monitoring, Logs & Final Submission
- [x] Request/response logging
- [x] Prometheus metrics (request count, latency)
- [x] Model performance tracking

## 🔧 Configuration

### Training Configuration (`configs/train_config.yaml`)
- Batch size, learning rate, epochs
- Model architecture selection
- Data augmentation settings

### Deployment Configuration
- Kubernetes: `k8s/deployment.yaml`, `k8s/service.yaml`
- Docker Compose: `docker-compose.yml`

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check endpoint |
| `/predict` | POST | Predict cat/dog from image |
| `/metrics` | GET | Prometheus metrics |

## 🛠️ CI/CD Pipeline

The pipeline automatically:
1. Runs unit tests on every push/PR
2. Builds Docker image
3. Pushes to container registry
4. Deploys to Kubernetes cluster (on main branch)
5. Runs smoke tests

## 📈 Monitoring

- **Prometheus**: Collects metrics (request count, latency)
- **Logging**: Structured JSON logging for all requests
- **Model Performance**: Tracks predictions and accuracy

## 📝 License

MIT License
