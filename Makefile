# =============================================================================
# Makefile for Cats vs Dogs MLOps Pipeline
# =============================================================================

.PHONY: help setup install lint test train build run deploy clean

# Default target
help:
	@echo "Cats vs Dogs MLOps Pipeline - Available Commands"
	@echo "================================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make setup        - Full project setup"
	@echo "  make install      - Install dependencies"
	@echo "  make install-dev  - Install dev dependencies"
	@echo ""
	@echo "Development:"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo "  make test         - Run unit tests"
	@echo "  make test-cov     - Run tests with coverage"
	@echo ""
	@echo "Data & Training:"
	@echo "  make download     - Download dataset from Kaggle"
	@echo "  make preprocess   - Preprocess dataset"
	@echo "  make train        - Train the model"
	@echo "  make evaluate     - Evaluate the model"
	@echo "  make mlflow-ui    - Start MLflow UI"
	@echo ""
	@echo "Docker:"
	@echo "  make build        - Build Docker image"
	@echo "  make run          - Run Docker container"
	@echo "  make push         - Push Docker image"
	@echo "  make compose-up   - Start with Docker Compose"
	@echo "  make compose-down - Stop Docker Compose"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy-k8s   - Deploy to Kubernetes"
	@echo "  make smoke-test   - Run smoke tests"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make clean-all    - Clean everything including data"

# Variables
PYTHON := python3
PIP := pip
DOCKER := docker
DOCKER_COMPOSE := docker-compose
IMAGE_NAME := cats-dogs-classifier
IMAGE_TAG := latest
REGISTRY := ghcr.io/your-username

# Setup
setup:
	@echo "Setting up project..."
	chmod +x scripts/setup.sh
	./scripts/setup.sh

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install black isort flake8 mypy pre-commit

# Code Quality
lint:
	@echo "Running linters..."
	flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503
	mypy src/ --ignore-missing-imports

format:
	@echo "Formatting code..."
	black src/ tests/
	isort src/ tests/

# Testing
test:
	@echo "Running tests..."
	pytest tests/ -v

test-cov:
	@echo "Running tests with coverage..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=xml
	@echo "Coverage report: htmlcov/index.html"

# Data & Training
download:
	@echo "Downloading dataset..."
	$(PYTHON) scripts/download_dataset.py --output-dir data/raw --organize

preprocess:
	@echo "Preprocessing data..."
	$(PYTHON) src/data/preprocess.py --raw-dir data/raw --processed-dir data/processed

train:
	@echo "Training model..."
	$(PYTHON) src/training/train.py --config configs/train_config.yaml

evaluate:
	@echo "Evaluating model..."
	$(PYTHON) scripts/evaluate.py --model-path models/best_model.pt

mlflow-ui:
	@echo "Starting MLflow UI at http://localhost:5000"
	mlflow ui --port 5000

# Docker
build:
	@echo "Building Docker image..."
	$(DOCKER) build -t $(IMAGE_NAME):$(IMAGE_TAG) .

run:
	@echo "Running Docker container..."
	$(DOCKER) run -p 8000:8000 -v $(PWD)/models:/app/models $(IMAGE_NAME):$(IMAGE_TAG)

push:
	@echo "Pushing Docker image..."
	$(DOCKER) tag $(IMAGE_NAME):$(IMAGE_TAG) $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)
	$(DOCKER) push $(REGISTRY)/$(IMAGE_NAME):$(IMAGE_TAG)

compose-up:
	@echo "Starting Docker Compose..."
	$(DOCKER_COMPOSE) up -d

compose-down:
	@echo "Stopping Docker Compose..."
	$(DOCKER_COMPOSE) down

compose-logs:
	$(DOCKER_COMPOSE) logs -f

# Kubernetes
deploy-k8s:
	@echo "Deploying to Kubernetes..."
	kubectl apply -f k8s/namespace.yaml
	kubectl apply -f k8s/configmap.yaml
	kubectl apply -f k8s/deployment.yaml
	kubectl apply -f k8s/service.yaml
	kubectl rollout status deployment/cats-dogs-classifier -n cats-dogs-classifier

smoke-test:
	@echo "Running smoke tests..."
	$(PYTHON) scripts/smoke_test.py --url http://localhost:8000 --wait

# DVC
dvc-pull:
	@echo "Pulling data from DVC remote..."
	dvc pull

dvc-push:
	@echo "Pushing data to DVC remote..."
	dvc push

# Cleanup
clean:
	@echo "Cleaning build artifacts..."
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	rm -rf .pytest_cache htmlcov .coverage coverage.xml
	rm -rf *.egg-info dist build
	rm -rf .mypy_cache

clean-all: clean
	@echo "Cleaning all generated files..."
	rm -rf data/processed
	rm -rf models/*.pt models/*.pth
	rm -rf mlruns
	rm -rf logs/*
	rm -rf reports/*

# Pipeline commands
pipeline-preprocess:
	dvc repro preprocess

pipeline-train:
	dvc repro train

pipeline-all:
	dvc repro
