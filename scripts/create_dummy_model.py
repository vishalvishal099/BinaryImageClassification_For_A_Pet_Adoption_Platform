#!/usr/bin/env python3
"""
Create a dummy model for testing purposes.

This script creates a minimal trained model that can be used
for testing the inference service without full training.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.cnn import SimpleCNN


def create_dummy_model(output_path: str = "models/best_model.pt"):
    """Create and save a dummy model."""
    # Create model
    model = SimpleCNN(num_classes=2, dropout_rate=0.5)
    
    # Initialize with random weights (already done in __init__)
    
    # Create checkpoint
    checkpoint = {
        'epoch': 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': {},
        'val_accuracy': 0.5,
        'config': {
            'model_name': 'simple_cnn',
            'num_classes': 2,
            'dropout_rate': 0.5,
            'batch_size': 32,
            'learning_rate': 0.001
        }
    }
    
    # Save
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, output_path)
    print(f"Dummy model saved to {output_path}")
    
    # Verify
    loaded = torch.load(output_path)
    print(f"Model config: {loaded['config']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create a dummy model")
    parser.add_argument(
        "--output",
        type=str,
        default="models/best_model.pt",
        help="Output path for the model"
    )
    
    args = parser.parse_args()
    create_dummy_model(args.output)
