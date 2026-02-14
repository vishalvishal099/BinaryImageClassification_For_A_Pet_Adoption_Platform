"""
Training Script for Cats vs Dogs Classification with MLflow Tracking.

This script handles:
- Model training with configurable parameters
- Experiment tracking with MLflow
- Logging metrics, parameters, and artifacts
- Model checkpointing
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import create_dataloaders
from src.data.preprocess import get_label_names
from src.models.cnn import count_parameters, get_model


class Trainer:
    """
    Trainer class for Cats vs Dogs classification.
    """

    def __init__(self, config: Dict, device: Optional[str] = None):
        """
        Initialize trainer.

        Args:
            config: Training configuration dictionary
            device: Device to train on ('cuda', 'cpu', or None for auto)
        """
        self.config = config

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Initialize components
        self._setup_model()
        self._setup_dataloaders()
        self._setup_training()

    def _setup_model(self):
        """Initialize model."""
        self.model = get_model(
            model_name=self.config.get("model_name", "simple_cnn"),
            num_classes=self.config.get("num_classes", 2),
            dropout_rate=self.config.get("dropout_rate", 0.5),
        ).to(self.device)

        total, trainable = count_parameters(self.model)
        print(f"Model: {self.config.get('model_name', 'simple_cnn')}")
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")

    def _setup_dataloaders(self):
        """Initialize data loaders."""
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            data_dir=self.config.get("data_dir", "data/processed"),
            batch_size=self.config.get("batch_size", 32),
            num_workers=self.config.get("num_workers", 4),
        )

        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")

    def _setup_training(self):
        """Initialize optimizer, scheduler, and loss function."""
        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        optimizer_name = self.config.get("optimizer", "adam").lower()
        lr = self.config.get("learning_rate", 0.001)
        weight_decay = self.config.get("weight_decay", 0.0001)

        if optimizer_name == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

    def train_epoch(self) -> Tuple[float, float]:
        """
        Train for one epoch.

        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{total_loss / (batch_idx + 1):.4f}",
                    "acc": f"{100. * correct / total:.2f}%",
                }
            )

        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    def validate(
        self, loader: Optional[DataLoader] = None
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Validate the model.

        Args:
            loader: DataLoader to use (defaults to validation loader)

        Returns:
            Tuple of (loss, accuracy, all_preds, all_labels)
        """
        if loader is None:
            loader = self.val_loader

        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(loader, desc="Validating"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        accuracy = accuracy_score(all_labels, all_preds)

        return avg_loss, accuracy, all_preds, all_labels

    def train(
        self,
        num_epochs: int,
        save_dir: str = "models",
        experiment_name: str = "cats_dogs_classification",
    ) -> Dict:
        """
        Full training loop with MLflow tracking.

        Args:
            num_epochs: Number of epochs to train
            save_dir: Directory to save model checkpoints
            experiment_name: MLflow experiment name

        Returns:
            Dictionary with training history
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Set up MLflow
        mlflow.set_experiment(experiment_name)

        history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        best_val_acc = 0.0
        best_model_path = None

        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(
                {
                    "model_name": self.config.get("model_name", "simple_cnn"),
                    "batch_size": self.config.get("batch_size", 32),
                    "learning_rate": self.config.get("learning_rate", 0.001),
                    "num_epochs": num_epochs,
                    "optimizer": self.config.get("optimizer", "adam"),
                    "dropout_rate": self.config.get("dropout_rate", 0.5),
                    "device": str(self.device),
                }
            )

            for epoch in range(1, num_epochs + 1):
                print(f"\n{'=' * 60}")
                print(f"Epoch {epoch}/{num_epochs}")
                print("=" * 60)

                # Train
                train_loss, train_acc = self.train_epoch()

                # Validate
                val_loss, val_acc, val_preds, val_labels = self.validate()

                # Update scheduler
                self.scheduler.step(val_loss)

                # Record history
                history["train_loss"].append(train_loss)
                history["train_acc"].append(train_acc)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)

                # Log metrics to MLflow
                mlflow.log_metrics(
                    {
                        "train_loss": train_loss,
                        "train_accuracy": train_acc,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    },
                    step=epoch,
                )

                print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_path = save_path / "best_model.pt"
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "val_accuracy": val_acc,
                            "config": self.config,
                        },
                        best_model_path,
                    )
                    print(f"Saved best model with val_acc: {val_acc:.4f}")

            # Final evaluation on test set
            print("\n" + "=" * 60)
            print("Final Evaluation on Test Set")
            print("=" * 60)

            test_loss, test_acc, test_preds, test_labels = self.validate(
                self.test_loader
            )

            # Calculate detailed metrics
            precision = precision_score(test_labels, test_preds, average="binary")
            recall = recall_score(test_labels, test_preds, average="binary")
            f1 = f1_score(test_labels, test_preds, average="binary")
            conf_matrix = confusion_matrix(test_labels, test_preds)

            print(f"Test Loss: {test_loss:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"\nConfusion Matrix:\n{conf_matrix}")
            print("\nClassification Report:")
            print(
                classification_report(
                    test_labels,
                    test_preds,
                    target_names=list(get_label_names().values()),
                )
            )

            # Log final metrics
            mlflow.log_metrics(
                {
                    "test_loss": test_loss,
                    "test_accuracy": test_acc,
                    "test_precision": precision,
                    "test_recall": recall,
                    "test_f1": f1,
                }
            )

            # Log confusion matrix as artifact
            np.save(save_path / "confusion_matrix.npy", conf_matrix)
            mlflow.log_artifact(str(save_path / "confusion_matrix.npy"))

            # Log the best model
            if best_model_path:
                mlflow.log_artifact(str(best_model_path))

                # Also log as MLflow model
                mlflow.pytorch.log_model(
                    self.model, "model", registered_model_name="cats_dogs_classifier"
                )

            history["test_loss"] = test_loss
            history["test_acc"] = test_acc
            history["best_val_acc"] = best_val_acc

        return history


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train Cats vs Dogs Classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/processed", help="Path to processed data"
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of epochs (overrides config)"
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device to use (cuda/cpu)"
    )

    args = parser.parse_args()

    # Load config
    if Path(args.config).exists():
        config = load_config(args.config)
    else:
        print(f"Config file not found: {args.config}")
        print("Using default configuration...")
        config = {
            "model_name": "simple_cnn",
            "batch_size": 32,
            "learning_rate": 0.001,
            "num_epochs": 20,
            "optimizer": "adam",
            "dropout_rate": 0.5,
            "num_workers": 4,
        }

    # Override with command line arguments
    config["data_dir"] = args.data_dir
    if args.epochs:
        config["num_epochs"] = args.epochs

    # Initialize trainer
    trainer = Trainer(config, device=args.device)

    # Train
    history = trainer.train(
        num_epochs=config.get("num_epochs", 20),
        save_dir="models",
        experiment_name="cats_dogs_classification",
    )

    print("\nTraining completed!")
    print(f"Best validation accuracy: {history['best_val_acc']:.4f}")


if __name__ == "__main__":
    main()
