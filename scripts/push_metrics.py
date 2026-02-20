#!/usr/bin/env python3
"""
Rich Prometheus metrics server for Pet Adoption ML Platform.
Serves realistic, high-density ML observability metrics on port 8081.
Categories: API traffic, latency percentiles, predictions, model performance,
            errors, system resources, batch inference, data pipeline,
            business metrics, time context.
"""
import math, random, time
from http.server import BaseHTTPRequestHandler, HTTPServer

START_TIME = time.time()
random.seed(int(START_TIME) % 9999)

_counters = {
    "req_predict_200": 0, "req_predict_400": 0, "req_predict_500": 0,
    "req_health_200": 0, "req_metrics_200": 0,
    "req_batch_200": 0, "req_batch_500": 0,
    "pred_cat_high": 0, "pred_cat_med": 0, "pred_cat_low": 0,
    "pred_dog_high": 0, "pred_dog_med": 0, "pred_dog_low": 0,
    "err_timeout": 0, "err_invalid": 0, "err_model": 0, "err_oom": 0,
    "adoption_cat": 0, "adoption_dog": 0,
    "preprocess_ok": 0, "preprocess_fail": 0,
    "dvc_pull": 0,
}
_last_tick = time.time()


def _tick():
    global _last_tick
    now = time.time()
    dt = now - _last_tick
    _last_tick = now
    hour = time.localtime().tm_hour
    traffic = (0.3 + 0.7 * math.sin(math.pi * max(0, hour - 6) / 14)) if 6 <= hour <= 20 else 0.15
    r = dt * traffic

    def add(k, rate, noise=0.15):
        _counters[k] = max(0.0, _counters[k] + max(0, random.gauss(rate * r, noise * rate * r + 1e-9)))

    add("req_predict_200", 14); add("req_predict_400", 0.35); add("req_predict_500", 0.06)
    add("req_health_200", 5);   add("req_metrics_200", 2.5)
    add("req_batch_200", 1.8);  add("req_batch_500", 0.025)
    add("pred_cat_high", 5.2);  add("pred_cat_med", 1.4); add("pred_cat_low", 0.35)
    add("pred_dog_high", 4.8);  add("pred_dog_med", 1.2); add("pred_dog_low", 0.25)
    add("err_timeout", 0.05);   add("err_invalid", 0.25); add("err_model", 0.012); add("err_oom", 0.006)
    add("adoption_cat", 0.9);   add("adoption_dog", 0.7)
    add("preprocess_ok", 15);   add("preprocess_fail", 0.12)
    add("dvc_pull", 0.0015)


def jit(v, pct=0.04):
    return max(0.0, v + random.gauss(0, pct * abs(v) + 1e-9))


def c(k):
    return str(max(0, int(_counters[k])))


def build_metrics():
    _tick()
    uptime = time.time() - START_TIME
    hour = time.localtime().tm_hour
    dow  = time.localtime().tm_wday
    t    = time.time()

    cpu      = jit(30 + 28 * math.sin(t / 290), 0.05)
    mem_rss  = jit(490 + 130 * math.sin(t / 580), 0.03)
    mem_vms  = jit(mem_rss * 2.1, 0.02)
    gpu_util = jit(40 + 20 * math.sin(t / 175), 0.06)
    gpu_mem  = jit(1750 + 250 * math.sin(t / 230), 0.02)

    acc_v1 = jit(0.8830, 0.004); acc_v2 = jit(0.9015, 0.003)
    prec_cat = jit(0.891, 0.005); prec_dog = jit(0.876, 0.006)
    rec_cat  = jit(0.878, 0.005); rec_dog  = jit(0.888, 0.006)
    f1_cat   = jit(0.884, 0.004); f1_dog   = jit(0.882, 0.004)
    conf_cat = jit(0.874, 0.007); conf_dog = jit(0.861, 0.008)

    lat_p50  = jit(0.192, 0.04); lat_p90  = jit(0.378, 0.05)
    lat_p95  = jit(0.432, 0.06); lat_p99  = jit(0.618, 0.07)
    lat_max  = jit(0.910, 0.08); lat_avg  = jit(0.241, 0.04)
    blat_avg = jit(1.130, 0.06); blat_p95 = jit(2.110, 0.08)

    rpm_pred  = jit(18.5, 0.06); rpm_batch = jit(2.1, 0.08)
    err_rate  = jit(0.0235, 0.10)
    batch_q   = max(0, int(jit(4.5, 0.4)))
    batch_sz  = jit(16.2, 0.09); tput = jit(14.8, 0.07)

    pre_resize    = jit(0.012, 0.06); pre_norm = jit(0.008, 0.06)
    pre_augment   = jit(0.014, 0.07); pre_total = pre_resize + pre_norm + pre_augment
    valid_time    = jit(0.0082, 0.05)
    img_small     = max(0, int(jit(340, 0.05)))
    img_medium    = max(0, int(jit(520, 0.04)))
    img_large     = max(0, int(jit(140, 0.06)))
    sessions      = max(0, int(jit(27, 0.10)))
    sess_dur      = jit(142, 0.08)

    out = []
    def g(name, hlp, typ, *samples):
        out.append(f"# HELP {name} {hlp}")
        out.append(f"# TYPE {name} {typ}")
        out.extend(samples)
        out.append("")

    # ── API Requests ──────────────────────────────────────────────────────────
    g("api_requests_total", "HTTP requests by endpoint, method, status", "counter",
      f'api_requests_total{{endpoint="/predict",method="POST",status="200"}} {c("req_predict_200")}',
      f'api_requests_total{{endpoint="/predict",method="POST",status="400"}} {c("req_predict_400")}',
      f'api_requests_total{{endpoint="/predict",method="POST",status="500"}} {c("req_predict_500")}',
      f'api_requests_total{{endpoint="/health",method="GET",status="200"}}   {c("req_health_200")}',
      f'api_requests_total{{endpoint="/metrics",method="GET",status="200"}}  {c("req_metrics_200")}',
      f'api_requests_total{{endpoint="/batch",method="POST",status="200"}}   {c("req_batch_200")}',
      f'api_requests_total{{endpoint="/batch",method="POST",status="500"}}   {c("req_batch_500")}',
    )
    g("requests_per_minute", "Requests per minute by endpoint", "gauge",
      f'requests_per_minute{{endpoint="/predict"}} {rpm_pred:.2f}',
      f'requests_per_minute{{endpoint="/batch"}}   {rpm_batch:.2f}',
    )

    # ── Latency ───────────────────────────────────────────────────────────────
    g("api_request_latency_seconds", "Latency percentiles by endpoint", "gauge",
      f'api_request_latency_seconds{{endpoint="/predict",quantile="avg"}}  {lat_avg:.4f}',
      f'api_request_latency_seconds{{endpoint="/predict",quantile="0.5"}}  {lat_p50:.4f}',
      f'api_request_latency_seconds{{endpoint="/predict",quantile="0.9"}}  {lat_p90:.4f}',
      f'api_request_latency_seconds{{endpoint="/predict",quantile="0.95"}} {lat_p95:.4f}',
      f'api_request_latency_seconds{{endpoint="/predict",quantile="0.99"}} {lat_p99:.4f}',
      f'api_request_latency_seconds{{endpoint="/predict",quantile="max"}}  {lat_max:.4f}',
      f'api_request_latency_seconds{{endpoint="/batch",quantile="avg"}}    {blat_avg:.4f}',
      f'api_request_latency_seconds{{endpoint="/batch",quantile="0.95"}}   {blat_p95:.4f}',
    )

    # ── Predictions ───────────────────────────────────────────────────────────
    g("api_predictions_total", "Predictions by label, confidence bucket, model version", "counter",
      f'api_predictions_total{{label="cat",confidence="high",model="v1"}}   {c("pred_cat_high")}',
      f'api_predictions_total{{label="cat",confidence="medium",model="v1"}} {c("pred_cat_med")}',
      f'api_predictions_total{{label="cat",confidence="low",model="v1"}}    {c("pred_cat_low")}',
      f'api_predictions_total{{label="dog",confidence="high",model="v1"}}   {c("pred_dog_high")}',
      f'api_predictions_total{{label="dog",confidence="medium",model="v1"}} {c("pred_dog_med")}',
      f'api_predictions_total{{label="dog",confidence="low",model="v1"}}    {c("pred_dog_low")}',
    )
    g("model_prediction_confidence", "Average confidence score by class", "gauge",
      f'model_prediction_confidence{{label="cat"}} {conf_cat:.4f}',
      f'model_prediction_confidence{{label="dog"}} {conf_dog:.4f}',
    )

    # ── Model Performance ─────────────────────────────────────────────────────
    g("model_accuracy", "Model accuracy by version", "gauge",
      f'model_accuracy{{version="v1",dataset="val"}} {acc_v1:.4f}',
      f'model_accuracy{{version="v2",dataset="val"}} {acc_v2:.4f}',
    )
    g("model_precision", "Precision by class", "gauge",
      f'model_precision{{label="cat",version="v1"}} {prec_cat:.4f}',
      f'model_precision{{label="dog",version="v1"}} {prec_dog:.4f}',
    )
    g("model_recall", "Recall by class", "gauge",
      f'model_recall{{label="cat",version="v1"}} {rec_cat:.4f}',
      f'model_recall{{label="dog",version="v1"}} {rec_dog:.4f}',
    )
    g("model_f1_score", "F1 score by class", "gauge",
      f'model_f1_score{{label="cat",version="v1"}} {f1_cat:.4f}',
      f'model_f1_score{{label="dog",version="v1"}} {f1_dog:.4f}',
    )

    # ── Errors ────────────────────────────────────────────────────────────────
    g("api_errors_total", "Errors by type", "counter",
      f'api_errors_total{{type="timeout",endpoint="/predict"}}       {c("err_timeout")}',
      f'api_errors_total{{type="invalid_format",endpoint="/predict"}} {c("err_invalid")}',
      f'api_errors_total{{type="model_failure",endpoint="/predict"}} {c("err_model")}',
      f'api_errors_total{{type="out_of_memory",endpoint="/predict"}} {c("err_oom")}',
    )
    g("api_error_rate", "Error rate (errors/requests)", "gauge",
      f"api_error_rate {err_rate:.5f}",
    )

    # ── System Resources ──────────────────────────────────────────────────────
    g("process_cpu_usage_percent", "CPU utilization percent", "gauge",
      f'process_cpu_usage_percent{{core="all"}} {cpu:.2f}',
    )
    g("process_memory_usage_mb", "Memory usage in MB", "gauge",
      f'process_memory_usage_mb{{type="rss"}} {mem_rss:.1f}',
      f'process_memory_usage_mb{{type="vms"}} {mem_vms:.1f}',
    )
    g("gpu_utilization_percent", "GPU utilization percent", "gauge",
      f'gpu_utilization_percent{{device="0"}} {gpu_util:.2f}',
    )
    g("gpu_memory_used_mb", "GPU memory used in MB", "gauge",
      f'gpu_memory_used_mb{{device="0"}}   {gpu_mem:.1f}',
      f'gpu_memory_total_mb{{device="0"}} 8192',
    )

    # ── Batch Inference ───────────────────────────────────────────────────────
    g("batch_queue_depth", "Images queued for batch processing", "gauge",
      f"batch_queue_depth {batch_q}",
    )
    g("batch_size_avg", "Average batch size", "gauge",
      f"batch_size_avg {batch_sz:.1f}",
    )
    g("batch_throughput_images_per_second", "Batch throughput images/sec", "gauge",
      f"batch_throughput_images_per_second {tput:.2f}",
    )

    # ── Data Pipeline ─────────────────────────────────────────────────────────
    g("preprocessing_duration_seconds", "Preprocessing stage durations", "gauge",
      f'preprocessing_duration_seconds{{stage="resize"}}    {pre_resize:.4f}',
      f'preprocessing_duration_seconds{{stage="normalize"}} {pre_norm:.4f}',
      f'preprocessing_duration_seconds{{stage="augment"}}   {pre_augment:.4f}',
      f'preprocessing_duration_seconds{{stage="total"}}     {pre_total:.4f}',
    )
    g("preprocessing_images_total", "Images preprocessed by status", "counter",
      f'preprocessing_images_total{{status="success"}} {c("preprocess_ok")}',
      f'preprocessing_images_total{{status="failed"}}  {c("preprocess_fail")}',
    )
    g("validation_duration_seconds", "Input validation duration", "gauge",
      f"validation_duration_seconds {valid_time:.4f}",
    )
    g("input_image_size_total", "Image size distribution by bucket", "counter",
      f'input_image_size_total{{bucket="small_lt224"}}    {img_small}',
      f'input_image_size_total{{bucket="medium_224_512"}} {img_medium}',
      f'input_image_size_total{{bucket="large_gt512"}}    {img_large}',
    )
    g("dvc_pulls_total", "DVC data pull operations", "counter",
      f"dvc_pulls_total {c('dvc_pull')}",
    )

    # ── Business Metrics ──────────────────────────────────────────────────────
    g("pet_adoptions_total", "Pet adoptions by species", "counter",
      f'pet_adoptions_total{{species="cat"}} {c("adoption_cat")}',
      f'pet_adoptions_total{{species="dog"}} {c("adoption_dog")}',
    )
    g("active_user_sessions", "Active user sessions", "gauge",
      f"active_user_sessions {sessions}",
    )
    g("user_session_duration_seconds", "Average session duration", "gauge",
      f"user_session_duration_seconds {sess_dur:.1f}",
    )

    # ── Time / Uptime ─────────────────────────────────────────────────────────
    g("current_hour", "Current hour (0-23)", "gauge",           f"current_hour {hour}")
    g("current_day_of_week", "Day of week (0=Mon)", "gauge",    f"current_day_of_week {dow}")
    g("service_uptime_seconds", "Service uptime in seconds", "gauge",
      f"service_uptime_seconds {uptime:.2f}",
    )
    g("data_version_timestamp", "Unix timestamp of last DVC sync", "gauge",
      f"data_version_timestamp {int(START_TIME)}",
    )
    g("model_last_trained_timestamp", "Unix timestamp of last training run", "gauge",
      f'model_last_trained_timestamp{{version="v1"}} {int(START_TIME - 86400)}',
      f'model_last_trained_timestamp{{version="v2"}} {int(START_TIME - 3600)}',
    )

    return "\n".join(out) + "\n"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/metrics", "/"):
            body = build_metrics().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()

    def log_message(self, *_):
        pass


if __name__ == "__main__":
    server = HTTPServer(("0.0.0.0", 8081), Handler)
    print("Rich metrics server → http://localhost:8081/metrics")
    print("40+ metrics: API · Latency · Predictions · Model perf · Errors")
    print("             System · Batch · Data pipeline · Business · Time")
    server.serve_forever()
