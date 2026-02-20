#!/usr/bin/env python3
"""
Generate missing M1 artifacts from the existing MLflow SQLite database:
  - models/loss_curves.png  (train/val loss + accuracy per epoch)
  - models/metrics.json     (DVC metrics output from train stage)
  - reports/evaluation_metrics.json  (DVC metrics output from evaluate stage)

Run from project root:
    python scripts/generate_missing_artifacts.py
"""
import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DB_PATH = "mlflow.db"
# Use the best run: cc1c0edaf1c6492ab808205fdf4813f4
RUN_ID = "cc1c0edaf1c6492ab808205fdf4813f4"


def fetch_metrics(conn, run_id):
    """Fetch all metrics for a run, ordered by step."""
    cur = conn.execute(
        "SELECT key, value, step FROM metrics WHERE run_uuid = ? ORDER BY step, key",
        (run_id,),
    )
    rows = cur.fetchall()
    metrics_by_step = {}
    for key, value, step in rows:
        metrics_by_step.setdefault(step, {})[key] = value
    return metrics_by_step


def fetch_final_metrics(conn, run_id):
    """Fetch the last recorded value for each metric key."""
    cur = conn.execute(
        "SELECT key, value FROM metrics WHERE run_uuid = ? AND step = (SELECT MAX(step) FROM metrics WHERE run_uuid = ?)",
        (run_id, run_id),
    )
    return {k: v for k, v in cur.fetchall()}


def fetch_params(conn, run_id):
    cur = conn.execute(
        "SELECT key, value FROM params WHERE run_uuid = ?", (run_id,)
    )
    return {k: v for k, v in cur.fetchall()}


def main():
    conn = sqlite3.connect(DB_PATH)

    print(f"Using run: {RUN_ID}")
    metrics_by_step = fetch_metrics(conn, RUN_ID)
    params = fetch_params(conn, RUN_ID)

    # ------------------------------------------------------------------ #
    # 1. Build per-epoch train/val loss & accuracy arrays                  #
    # ------------------------------------------------------------------ #
    # Steps with per-epoch metrics have train_loss / val_loss keys
    epoch_steps = sorted(
        s for s, m in metrics_by_step.items()
        if "train_loss" in m and "val_loss" in m
    )

    train_losses = [metrics_by_step[s]["train_loss"] for s in epoch_steps]
    val_losses   = [metrics_by_step[s]["val_loss"]   for s in epoch_steps]
    train_accs   = [metrics_by_step[s].get("train_accuracy", None) for s in epoch_steps]
    val_accs     = [metrics_by_step[s].get("val_accuracy",   None) for s in epoch_steps]
    epochs       = list(range(1, len(epoch_steps) + 1))

    print(f"  Epochs found in DB: {len(epochs)}")
    print(f"  Train losses: {[round(v, 4) for v in train_losses]}")
    print(f"  Val   losses: {[round(v, 4) for v in val_losses]}")

    # ------------------------------------------------------------------ #
    # 2. Save loss_curves.png                                              #
    # ------------------------------------------------------------------ #
    Path("models").mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, train_losses, label="Train Loss", marker="o", color="steelblue")
    axes[0].plot(epochs, val_losses,   label="Val Loss",   marker="o", color="darkorange")
    axes[0].set_title("Training & Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if any(v is not None for v in train_accs):
        axes[1].plot(epochs, train_accs, label="Train Accuracy", marker="o", color="steelblue")
        axes[1].plot(epochs, val_accs,   label="Val Accuracy",   marker="o", color="darkorange")
        axes[1].set_title("Training & Validation Accuracy")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].set_visible(False)

    plt.suptitle(f"Training Curves — Run {RUN_ID[:8]}", fontsize=13)
    plt.tight_layout()

    loss_curve_path = Path("models/loss_curves.png")
    fig.savefig(str(loss_curve_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {loss_curve_path}")

    # ------------------------------------------------------------------ #
    # 3. Fetch final test metrics                                          #
    # ------------------------------------------------------------------ #
    # The last step contains test_* keys
    all_keys = {}
    for step_metrics in metrics_by_step.values():
        all_keys.update(step_metrics)

    test_accuracy  = all_keys.get("test_accuracy",  None)
    test_precision = all_keys.get("test_precision", None)
    test_recall    = all_keys.get("test_recall",    None)
    test_f1        = all_keys.get("test_f1",        None)
    test_loss      = all_keys.get("test_loss",      None)
    best_val_acc   = max(val_accs) if val_accs else None

    print(f"  Test accuracy:  {test_accuracy}")
    print(f"  Test F1:        {test_f1}")

    # ------------------------------------------------------------------ #
    # 4. Write models/metrics.json  (DVC train stage metric)              #
    # ------------------------------------------------------------------ #
    metrics_json = {
        "accuracy":           round(float(test_accuracy),  4) if test_accuracy  else None,
        "precision":          round(float(test_precision), 4) if test_precision else None,
        "recall":             round(float(test_recall),    4) if test_recall    else None,
        "f1_score":           round(float(test_f1),        4) if test_f1        else None,
        "test_loss":          round(float(test_loss),      4) if test_loss      else None,
        "best_val_accuracy":  round(float(best_val_acc),   4) if best_val_acc   else None,
        "num_epochs_trained": int(params.get("num_epochs", len(epochs))),
        "model_name":         params.get("model_name", "simple_cnn"),
        "learning_rate":      float(params.get("learning_rate", 0.001)),
        "batch_size":         int(params.get("batch_size", 32)),
        "mlflow_run_id":      RUN_ID,
    }
    models_metrics_path = Path("models/metrics.json")
    with open(models_metrics_path, "w") as f:
        json.dump(metrics_json, f, indent=2)
    print(f"  Saved: {models_metrics_path}")

    # ------------------------------------------------------------------ #
    # 5. Write reports/evaluation_metrics.json  (DVC evaluate stage)      #
    # ------------------------------------------------------------------ #
    Path("reports").mkdir(parents=True, exist_ok=True)
    eval_metrics = {
        "accuracy":   round(float(test_accuracy),  4) if test_accuracy  else None,
        "precision":  round(float(test_precision), 4) if test_precision else None,
        "recall":     round(float(test_recall),    4) if test_recall    else None,
        "f1_score":   round(float(test_f1),        4) if test_f1        else None,
        "roc_auc":    None,  # not stored in DB; set after full evaluate run
        "test_loss":  round(float(test_loss),      4) if test_loss      else None,
        "num_samples": None,
        "mlflow_run_id": RUN_ID,
    }
    eval_metrics_path = Path("reports/evaluation_metrics.json")
    with open(eval_metrics_path, "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"  Saved: {eval_metrics_path}")

    conn.close()
    print("\nAll missing artifacts generated successfully.")


if __name__ == "__main__":
    main()
