"""
Model Performance Tracking Module.

This module provides functionality to track model performance
post-deployment by collecting predictions and comparing with
ground truth labels.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import defaultdict
import threading

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class PerformanceTracker:
    """
    Tracks model performance metrics over time.
    
    Collects predictions and ground truth labels, then computes
    performance metrics periodically.
    """
    
    def __init__(
        self,
        storage_path: str = "logs/performance",
        batch_size: int = 100,
        flush_interval: int = 300  # 5 minutes
    ):
        """
        Initialize the performance tracker.
        
        Args:
            storage_path: Directory to store performance logs
            batch_size: Number of predictions to collect before computing metrics
            flush_interval: Seconds between automatic flushes
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        
        # In-memory storage
        self.predictions: List[Dict] = []
        self.lock = threading.Lock()
        
        # Metrics history
        self.metrics_history: List[Dict] = []
        
        # Counters
        self.total_predictions = 0
        self.class_counts = defaultdict(int)
        self.latency_samples: List[float] = []
        
        # Start background flush thread
        self._start_flush_thread()
    
    def _start_flush_thread(self):
        """Start background thread for periodic flushing."""
        def flush_loop():
            while True:
                time.sleep(self.flush_interval)
                self.flush()
        
        thread = threading.Thread(target=flush_loop, daemon=True)
        thread.start()
    
    def log_prediction(
        self,
        prediction: str,
        confidence: float,
        latency_ms: float,
        ground_truth: Optional[str] = None,
        metadata: Optional[Dict] = None
    ):
        """
        Log a single prediction.
        
        Args:
            prediction: Predicted class label
            confidence: Prediction confidence
            latency_ms: Prediction latency in milliseconds
            ground_truth: Optional ground truth label
            metadata: Optional additional metadata
        """
        with self.lock:
            record = {
                "timestamp": datetime.utcnow().isoformat(),
                "prediction": prediction,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "ground_truth": ground_truth,
                "metadata": metadata or {}
            }
            
            self.predictions.append(record)
            self.total_predictions += 1
            self.class_counts[prediction] += 1
            self.latency_samples.append(latency_ms)
            
            # Auto-flush if batch is full
            if len(self.predictions) >= self.batch_size:
                self._compute_and_store_metrics()
    
    def add_ground_truth(self, timestamp: str, ground_truth: str):
        """
        Add ground truth label to a previous prediction.
        
        Args:
            timestamp: Timestamp of the prediction
            ground_truth: Ground truth label
        """
        with self.lock:
            for pred in self.predictions:
                if pred["timestamp"] == timestamp:
                    pred["ground_truth"] = ground_truth
                    break
    
    def _compute_and_store_metrics(self):
        """Compute metrics from collected predictions."""
        if not self.predictions:
            return
        
        # Filter predictions with ground truth
        labeled = [p for p in self.predictions if p.get("ground_truth")]
        
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_predictions": len(self.predictions),
            "labeled_predictions": len(labeled),
            "class_distribution": dict(self.class_counts),
            "latency": {
                "mean_ms": np.mean(self.latency_samples) if self.latency_samples else 0,
                "median_ms": np.median(self.latency_samples) if self.latency_samples else 0,
                "p95_ms": np.percentile(self.latency_samples, 95) if self.latency_samples else 0,
                "p99_ms": np.percentile(self.latency_samples, 99) if self.latency_samples else 0
            }
        }
        
        # Compute accuracy metrics if we have labeled data
        if labeled:
            predictions = [p["prediction"] for p in labeled]
            ground_truths = [p["ground_truth"] for p in labeled]
            
            metrics["accuracy"] = accuracy_score(ground_truths, predictions)
            metrics["precision"] = precision_score(
                ground_truths, predictions, 
                average='weighted', 
                zero_division=0
            )
            metrics["recall"] = recall_score(
                ground_truths, predictions,
                average='weighted',
                zero_division=0
            )
            metrics["f1_score"] = f1_score(
                ground_truths, predictions,
                average='weighted',
                zero_division=0
            )
        
        self.metrics_history.append(metrics)
        
        # Save to file
        self._save_metrics(metrics)
        
        # Clear predictions (keep recent ones for potential ground truth updates)
        self.predictions = self.predictions[-10:]
        self.latency_samples = self.latency_samples[-100:]
    
    def _save_metrics(self, metrics: Dict):
        """Save metrics to a file."""
        date_str = datetime.utcnow().strftime("%Y-%m-%d")
        metrics_file = self.storage_path / f"metrics_{date_str}.jsonl"
        
        with open(metrics_file, 'a') as f:
            f.write(json.dumps(metrics) + '\n')
    
    def flush(self):
        """Force compute and store metrics."""
        with self.lock:
            self._compute_and_store_metrics()
    
    def get_current_stats(self) -> Dict:
        """Get current statistics."""
        with self.lock:
            labeled = [p for p in self.predictions if p.get("ground_truth")]
            
            stats = {
                "total_predictions": self.total_predictions,
                "pending_predictions": len(self.predictions),
                "labeled_predictions": len(labeled),
                "class_distribution": dict(self.class_counts),
                "latency": {
                    "mean_ms": np.mean(self.latency_samples) if self.latency_samples else 0,
                    "samples": len(self.latency_samples)
                }
            }
            
            if labeled:
                predictions = [p["prediction"] for p in labeled]
                ground_truths = [p["ground_truth"] for p in labeled]
                stats["accuracy"] = accuracy_score(ground_truths, predictions)
            
            return stats
    
    def get_metrics_history(self, limit: int = 100) -> List[Dict]:
        """Get recent metrics history."""
        return self.metrics_history[-limit:]


class SimulatedFeedback:
    """
    Simulates ground truth feedback for testing purposes.
    
    In a real deployment, this would be replaced by actual
    user feedback or labeling pipeline.
    """
    
    def __init__(self, accuracy: float = 0.85, delay_seconds: float = 5.0):
        """
        Initialize simulated feedback.
        
        Args:
            accuracy: Simulated model accuracy (for generating ground truth)
            delay_seconds: Delay before providing feedback
        """
        self.accuracy = accuracy
        self.delay_seconds = delay_seconds
    
    def generate_feedback(
        self,
        prediction: str,
        tracker: PerformanceTracker,
        timestamp: str
    ):
        """
        Generate simulated feedback after a delay.
        
        Args:
            prediction: The model's prediction
            tracker: Performance tracker to update
            timestamp: Timestamp of the original prediction
        """
        import random
        
        def delayed_feedback():
            time.sleep(self.delay_seconds)
            
            # Simulate correct/incorrect based on accuracy
            if random.random() < self.accuracy:
                ground_truth = prediction  # Correct prediction
            else:
                # Incorrect prediction - flip the class
                ground_truth = "dog" if prediction == "cat" else "cat"
            
            tracker.add_ground_truth(timestamp, ground_truth)
        
        thread = threading.Thread(target=delayed_feedback, daemon=True)
        thread.start()


# Global tracker instance
_tracker: Optional[PerformanceTracker] = None


def get_tracker() -> PerformanceTracker:
    """Get or create the global performance tracker."""
    global _tracker
    if _tracker is None:
        _tracker = PerformanceTracker()
    return _tracker
