#!/usr/bin/env python3
"""
Push sample metrics to Prometheus Pushgateway (or expose via HTTP)
Simulates real ML model metrics for the Pet Adoption Platform.
"""

import time
import random
import math
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading

# ── Metric state ─────────────────────────────────────────────────────────────
request_count = 0
prediction_counts = {"cat": 0, "dog": 0}
latency_samples = []
error_count = 0
start_time = time.time()

def generate_metrics_text():
    """Generate Prometheus-format metrics text with live values."""
    global request_count, prediction_counts, latency_samples, error_count

    # Simulate new requests each scrape
    new_requests = random.randint(1, 5)
    request_count += new_requests

    for _ in range(new_requests):
        label = "cat" if random.random() > 0.45 else "dog"
        prediction_counts[label] += 1
        latency_samples.append(random.uniform(0.05, 0.45))
        if random.random() < 0.02:
            error_count += 1

    # Keep only last 100 latency samples
    if len(latency_samples) > 100:
        latency_samples = latency_samples[-100:]

    avg_latency = sum(latency_samples) / len(latency_samples) if latency_samples else 0
    p95_latency = sorted(latency_samples)[int(len(latency_samples) * 0.95)] if len(latency_samples) >= 20 else avg_latency
    uptime = time.time() - start_time
    now = datetime.now()

    accuracy = round(0.92 + 0.04 * math.sin(time.time() / 300), 4)
    model_confidence = round(0.87 + 0.08 * random.random(), 4)

    lines = [
        "# HELP api_requests_total Total number of API prediction requests",
        "# TYPE api_requests_total counter",
        f'api_requests_total{{method="POST",endpoint="/predict"}} {request_count}',
        "",
        "# HELP api_predictions_total Total predictions by class label",
        "# TYPE api_predictions_total counter",
        f'api_predictions_total{{label="cat"}} {prediction_counts["cat"]}',
        f'api_predictions_total{{label="dog"}} {prediction_counts["dog"]}',
        "",
        "# HELP api_request_latency_seconds Average request latency in seconds",
        "# TYPE api_request_latency_seconds gauge",
        f"api_request_latency_seconds {avg_latency:.4f}",
        "",
        "# HELP api_request_latency_p95_seconds P95 request latency in seconds",
        "# TYPE api_request_latency_p95_seconds gauge",
        f"api_request_latency_p95_seconds {p95_latency:.4f}",
        "",
        "# HELP api_errors_total Total number of API errors",
        "# TYPE api_errors_total counter",
        f"api_errors_total {error_count}",
        "",
        "# HELP model_accuracy Current model accuracy score",
        "# TYPE model_accuracy gauge",
        f"model_accuracy {accuracy}",
        "",
        "# HELP model_confidence Average model prediction confidence",
        "# TYPE model_confidence gauge",
        f"model_confidence {model_confidence}",
        "",
        "# HELP service_uptime_seconds Total service uptime in seconds",
        "# TYPE service_uptime_seconds gauge",
        f"service_uptime_seconds {uptime:.2f}",
        "",
        "# HELP current_hour Current hour of day (0-23) for time-of-day analysis",
        "# TYPE current_hour gauge",
        f"current_hour {now.hour}",
        "",
        "# HELP current_day_of_week Day of week (0=Mon, 6=Sun)",
        "# TYPE current_day_of_week gauge",
        f"current_day_of_week {now.weekday()}",
        "",
        "# HELP requests_per_minute Approximate requests per minute",
        "# TYPE requests_per_minute gauge",
        f"requests_per_minute {round(request_count / max(uptime / 60, 1), 2)}",
        "",
        "# HELP data_version_timestamp Unix timestamp of last data version push",
        "# TYPE data_version_timestamp gauge",
        f"data_version_timestamp {int(time.time())}",
        "",
    ]
    return "\n".join(lines) + "\n"


class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/metrics", "/"):
            body = generate_metrics_text().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress access logs


def run_server(port=8081):
    server = HTTPServer(("0.0.0.0", port), MetricsHandler)
    print(f"[push_metrics] Prometheus metrics server running on port {port}")
    print(f"[push_metrics] Scrape URL: http://localhost:{port}/metrics")
    server.serve_forever()


if __name__ == "__main__":
    run_server(8081)
