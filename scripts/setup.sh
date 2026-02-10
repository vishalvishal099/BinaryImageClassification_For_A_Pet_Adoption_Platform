#!/bin/bash
# =============================================================================
# Setup Script for Cats vs Dogs MLOps Pipeline
# =============================================================================

set -e

echo "=============================================="
echo "Cats vs Dogs MLOps Pipeline - Setup Script"
echo "=============================================="
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

if [[ $(echo "$python_version < 3.9" | bc -l) -eq 1 ]]; then
    echo "Error: Python 3.9+ is required"
    exit 1
fi

# Create virtual environment
echo ""
echo "Step 1: Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created"
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo ""
echo "Step 2: Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo ""
echo "Step 3: Installing dependencies..."
pip install -r requirements.txt

# Initialize Git if not already
echo ""
echo "Step 4: Initializing Git repository..."
if [ ! -d ".git" ]; then
    git init
    echo "Git repository initialized"
else
    echo "Git repository already exists"
fi

# Initialize DVC
echo ""
echo "Step 5: Initializing DVC..."
if [ ! -d ".dvc" ]; then
    dvc init
    echo "DVC initialized"
else
    echo "DVC already initialized"
fi

# Create directories
echo ""
echo "Step 6: Creating directories..."
mkdir -p data/raw data/processed models logs reports

# Add data to DVC tracking
echo ""
echo "Step 7: Setting up DVC tracking..."
if [ -d "data/raw" ] && [ "$(ls -A data/raw 2>/dev/null)" ]; then
    dvc add data/raw
    echo "Raw data added to DVC"
fi

# Set up pre-commit hooks (optional)
echo ""
echo "Step 8: Setting up pre-commit hooks..."
if command -v pre-commit &> /dev/null; then
    pre-commit install
    echo "Pre-commit hooks installed"
else
    echo "pre-commit not found, skipping"
fi

echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Download dataset: python scripts/download_dataset.py --organize"
echo "3. Preprocess data: python src/data/preprocess.py"
echo "4. Train model: python src/training/train.py"
echo "5. Start MLflow UI: mlflow ui --port 5000"
echo ""
echo "For Docker deployment:"
echo "  docker-compose up -d"
echo ""
