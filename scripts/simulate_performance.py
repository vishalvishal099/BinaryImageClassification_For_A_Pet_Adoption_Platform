#!/usr/bin/env python3
"""
Post-Deployment Model Performance Simulation Script.

Sends a batch of simulated requests to the running inference service,
supplies true labels, and prints a model performance report using
the PerformanceTracker.

Usage:
    python scripts/simulate_performance.py [--url http://localhost:8000] [--count 50]

Requirements:
    pip install requests Pillow
"""

import argparse
import io
import json
import random
import struct
import sys
import time
import zlib
from pathlib import Path
from typing import List, Tuple

import requests

# ---------------------------------------------------------------------------
# Minimal synthetic PNG generator (no file I/O, no Pillow needed for images)
# ---------------------------------------------------------------------------

def _make_png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    c = zlib.crc32(chunk_type + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", c)


def make_synthetic_png(width: int = 64, height: int = 64,
                       r: int = 128, g: int = 128, b: int = 128) -> bytes:
    """Generate a minimal valid PNG of a solid colour (no Pillow needed)."""
    raw_rows = b"".join(
        b"\x00" + bytes([r, g, b] * width) for _ in range(height)
    )
    compressed = zlib.compress(raw_rows)
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _make_png_chunk(b"IHDR", ihdr_data)
        + _make_png_chunk(b"IDAT", compressed)
        + _make_png_chunk(b"IEND", b"")
    )


# ---------------------------------------------------------------------------
# Batch definition: (true_label, colour_hint)
# We use colour tints to add visual variety — the model sees a uniform image
# so the prediction will be purely model-based; true_label is ground truth.
# ---------------------------------------------------------------------------

def build_batch(count: int) -> List[Tuple[str, bytes]]:
    """
    Build a batch of (true_label, png_bytes) tuples.
    Half cats, half dogs with slight colour variations.
    """
    batch = []
    half = count // 2
    # Cats — slightly warm tones
    for i in range(half):
        r = random.randint(180, 220)
        g = random.randint(140, 180)
        b = random.randint(100, 140)
        batch.append(("cat", make_synthetic_png(r=r, g=g, b=b)))
    # Dogs — slightly cool tones
    for i in range(count - half):
        r = random.randint(100, 140)
        g = random.randint(140, 180)
        b = random.randint(180, 220)
        batch.append(("dog", make_synthetic_png(r=r, g=g, b=b)))
    random.shuffle(batch)
    return batch


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def wait_for_service(base_url: str, max_attempts: int = 20) -> bool:
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(f"{base_url}/health", timeout=5)
            if r.status_code == 200:
                print(f"  Service ready (attempt {attempt})")
                return True
        except requests.exceptions.RequestException:
            pass
        print(f"  Attempt {attempt}/{max_attempts} — not ready, retrying...")
        time.sleep(2)
    return False


def run_simulation(base_url: str, count: int, output_dir: Path) -> None:
    print("=" * 60)
    print("  Post-Deployment Performance Simulation")
    print("=" * 60)
    print(f"  Target : {base_url}")
    print(f"  Samples: {count}")
    print()

    # Wait for service
    print("Checking service availability...")
    if not wait_for_service(base_url):
        print("ERROR: Service not reachable. Is the inference server running?")
        print(f"  Start it with: uvicorn src.inference.app:app --port 8000")
        sys.exit(1)
    print()

    # Build batch
    batch = build_batch(count)

    results = []
    print(f"Sending {count} requests...")

    for i, (true_label, png_bytes) in enumerate(batch, start=1):
        try:
            t0 = time.time()
            response = requests.post(
                f"{base_url}/predict",
                files={"file": (f"img_{i:03d}.png", io.BytesIO(png_bytes), "image/png")},
                timeout=30,
            )
            latency_ms = round((time.time() - t0) * 1000, 2)

            if response.status_code == 200:
                data = response.json()
                predicted = data["predicted_class"]
                confidence = data["confidence"]
                correct = predicted == true_label
                results.append({
                    "index": i,
                    "true_label": true_label,
                    "predicted": predicted,
                    "confidence": confidence,
                    "latency_ms": latency_ms,
                    "correct": correct,
                })
                mark = "✓" if correct else "✗"
                print(f"  [{i:3d}/{count}] {mark} true={true_label:<4}  "
                      f"pred={predicted:<4}  conf={confidence:.2f}  {latency_ms:.0f}ms")
            elif response.status_code == 503:
                print(f"  [{i:3d}/{count}] SKIP — model not loaded (503)")
                results.append({
                    "index": i, "true_label": true_label,
                    "predicted": None, "confidence": 0.0,
                    "latency_ms": latency_ms, "correct": None,
                })
            else:
                print(f"  [{i:3d}/{count}] ERROR — HTTP {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"  [{i:3d}/{count}] REQUEST ERROR: {e}")

    print()

    # ---------------------------------------------------------------------------
    # Compute metrics
    # ---------------------------------------------------------------------------
    labeled = [r for r in results if r["correct"] is not None]
    if not labeled:
        print("No successful predictions — cannot compute metrics.")
        return

    n = len(labeled)
    correct = sum(1 for r in labeled if r["correct"])
    accuracy = correct / n

    # Per-class breakdown
    tp_cat = sum(1 for r in labeled if r["true_label"] == "cat" and r["predicted"] == "cat")
    fp_cat = sum(1 for r in labeled if r["true_label"] == "dog" and r["predicted"] == "cat")
    fn_cat = sum(1 for r in labeled if r["true_label"] == "cat" and r["predicted"] == "dog")
    tp_dog = sum(1 for r in labeled if r["true_label"] == "dog" and r["predicted"] == "dog")
    fp_dog = sum(1 for r in labeled if r["true_label"] == "cat" and r["predicted"] == "dog")
    fn_dog = sum(1 for r in labeled if r["true_label"] == "dog" and r["predicted"] == "cat")

    def safe_div(a, b):
        return a / b if b else 0.0

    prec_cat = safe_div(tp_cat, tp_cat + fp_cat)
    rec_cat  = safe_div(tp_cat, tp_cat + fn_cat)
    f1_cat   = safe_div(2 * prec_cat * rec_cat, prec_cat + rec_cat)

    prec_dog = safe_div(tp_dog, tp_dog + fp_dog)
    rec_dog  = safe_div(tp_dog, tp_dog + fn_dog)
    f1_dog   = safe_div(2 * prec_dog * rec_dog, prec_dog + rec_dog)

    weighted_prec = safe_div(prec_cat * (tp_cat + fn_cat) + prec_dog * (tp_dog + fn_dog), n)
    weighted_rec  = safe_div(rec_cat  * (tp_cat + fn_cat) + rec_dog  * (tp_dog + fn_dog), n)
    weighted_f1   = safe_div(f1_cat   * (tp_cat + fn_cat) + f1_dog   * (tp_dog + fn_dog), n)

    latencies = [r["latency_ms"] for r in labeled]
    latencies_sorted = sorted(latencies)
    mean_lat  = sum(latencies) / len(latencies)
    p95_lat   = latencies_sorted[int(0.95 * len(latencies_sorted))]

    # ---------------------------------------------------------------------------
    # Print report
    # ---------------------------------------------------------------------------
    print("=" * 60)
    print("  MODEL PERFORMANCE REPORT")
    print("=" * 60)
    print(f"  Samples evaluated : {n}")
    print(f"  Correct           : {correct}  ({accuracy:.1%})")
    print()
    print(f"  {'Metric':<22} {'Cat':>8} {'Dog':>8} {'Weighted':>10}")
    print(f"  {'-'*50}")
    print(f"  {'Precision':<22} {prec_cat:>8.3f} {prec_dog:>8.3f} {weighted_prec:>10.3f}")
    print(f"  {'Recall':<22} {rec_cat:>8.3f} {rec_dog:>8.3f} {weighted_rec:>10.3f}")
    print(f"  {'F1-Score':<22} {f1_cat:>8.3f} {f1_dog:>8.3f} {weighted_f1:>10.3f}")
    print()
    print(f"  Latency  mean={mean_lat:.1f}ms  p95={p95_lat:.1f}ms")
    print()

    # ---------------------------------------------------------------------------
    # Save JSONL report to logs/performance/
    # ---------------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    date_str = time.strftime("%Y-%m-%d")
    report_path = output_dir / f"metrics_{date_str}.jsonl"

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "simulate_performance.py",
        "total_predictions": n,
        "labeled_predictions": n,
        "accuracy": round(accuracy, 4),
        "precision": round(weighted_prec, 4),
        "recall": round(weighted_rec, 4),
        "f1_score": round(weighted_f1, 4),
        "class_distribution": {
            "cat": sum(1 for r in labeled if r["true_label"] == "cat"),
            "dog": sum(1 for r in labeled if r["true_label"] == "dog"),
        },
        "latency": {
            "mean_ms": round(mean_lat, 2),
            "p95_ms": round(p95_lat, 2),
        },
        "per_class": {
            "cat": {"precision": round(prec_cat, 4), "recall": round(rec_cat, 4), "f1": round(f1_cat, 4)},
            "dog": {"precision": round(prec_dog, 4), "recall": round(rec_dog, 4), "f1": round(f1_dog, 4)},
        },
    }

    with open(report_path, "a") as f:
        f.write(json.dumps(report) + "\n")

    print(f"  Report saved → {report_path}")
    print("=" * 60)

    # Exit non-zero if accuracy is very low (sanity check)
    if accuracy < 0.4:
        print(f"\nWARNING: Accuracy {accuracy:.1%} is below 40% threshold!")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Simulate post-deployment performance evaluation"
    )
    parser.add_argument("--url", default="http://localhost:8000",
                        help="Inference service base URL")
    parser.add_argument("--count", type=int, default=50,
                        help="Number of requests to send (default: 50)")
    parser.add_argument("--output-dir", default="logs/performance",
                        help="Directory to save JSONL report")
    args = parser.parse_args()

    run_simulation(
        base_url=args.url.rstrip("/"),
        count=args.count,
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
