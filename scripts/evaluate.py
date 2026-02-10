#!/usr/bin/env python3
"""
Model Evaluation Script.

Evaluates a trained model on the test set and generates detailed metrics.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import CatsDogsDataset
from src.models.cnn import get_model
from src.data.preprocess import get_label_names


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Load a trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    
    config = checkpoint.get('config', {})
    model_name = config.get('model_name', 'simple_cnn')
    
    model = get_model(
        model_name=model_name,
        num_classes=2,
        dropout_rate=config.get('dropout_rate', 0.5)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def evaluate_model(
    model: torch.nn.Module,
    data_dir: str,
    device: torch.device,
    batch_size: int = 32
) -> Dict:
    """
    Evaluate model on test set.
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Load test dataset
    test_dataset = CatsDogsDataset(data_dir, split="test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probabilities[:, 1].cpu().numpy())  # Probability of class 1 (dog)
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='binary')
    recall = recall_score(all_labels, all_preds, average='binary')
    f1 = f1_score(all_labels, all_preds, average='binary')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    # ROC AUC
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc_auc = None
    
    # Class-specific metrics
    label_names = get_label_names()
    report = classification_report(
        all_labels, all_preds,
        target_names=[label_names[0], label_names[1]],
        output_dict=True
    )
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc) if roc_auc else None,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': report,
        'num_samples': len(all_labels)
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Path to processed data"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/evaluation_metrics.json",
        help="Path to save metrics"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model = load_model(args.model_path, device)
    
    # Evaluate
    print("Evaluating model...")
    metrics = evaluate_model(model, args.data_dir, device)
    
    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    if metrics['roc_auc']:
        print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:")
    print(np.array(metrics['confusion_matrix']))
    
    # Save metrics
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nMetrics saved to {output_path}")


if __name__ == "__main__":
    main()
