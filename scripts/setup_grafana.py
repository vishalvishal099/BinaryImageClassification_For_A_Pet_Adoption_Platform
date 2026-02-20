#!/usr/bin/env python3
"""
Setup Grafana: Add Prometheus datasource and create a rich
Pet Adoption Platform - ML Monitoring dashboard.
"""
import http.client
import json
import base64
import time
import sys

GRAFANA_HOST = "localhost"
GRAFANA_PORT = 3000
GRAFANA_USER = "admin"
GRAFANA_PASS = "admin"
PROMETHEUS_URL = "http://host.containers.internal:9090"


def request(method, path, body=None):
    creds = base64.b64encode(f"{GRAFANA_USER}:{GRAFANA_PASS}".encode()).decode()
    headers = {"Content-Type": "application/json", "Authorization": f"Basic {creds}"}
    conn = http.client.HTTPConnection(GRAFANA_HOST, GRAFANA_PORT, timeout=10)
    data = json.dumps(body).encode() if body else None
    conn.request(method, path, body=data, headers=headers)
    r = conn.getresponse()
    return r.status, json.loads(r.read().decode())


def wait_for_grafana(retries=20, delay=3):
    print("Waiting for Grafana to be ready...")
    for i in range(retries):
        try:
            conn = http.client.HTTPConnection(GRAFANA_HOST, GRAFANA_PORT, timeout=3)
            conn.request("GET", "/api/health")
            r = conn.getresponse()
            if r.status == 200:
                print("  Grafana is ready!")
                return True
        except Exception:
            pass
        print(f"  Not ready yet ({i+1}/{retries})...")
        time.sleep(delay)
    return False


def add_datasource():
    print("Adding Prometheus datasource...")
    status, resp = request("POST", "/api/datasources", {
        "name": "Prometheus",
        "type": "prometheus",
        "url": PROMETHEUS_URL,
        "access": "proxy",
        "isDefault": True,
        "jsonData": {"timeInterval": "10s"}
    })
    print(f"  Datasource status: {status}")
    _, sources = request("GET", "/api/datasources")
    if isinstance(sources, list):
        for s in sources:
            if s.get("name") == "Prometheus":
                return s.get("uid", "prometheus")
    return "prometheus"


def create_dashboard(ds_uid):
    print("Creating Pet Adoption Platform MLOps dashboard...")

    def ds():
        return {"type": "prometheus", "uid": ds_uid}

    panels = [
        {"id": 1, "type": "stat", "gridPos": {"x": 0, "y": 0, "w": 4, "h": 4},
         "title": "Total API Requests", "datasource": ds(),
         "options": {"colorMode": "background", "graphMode": "area", "reduceOptions": {"calcs": ["lastNotNull"]}},
         "fieldConfig": {"defaults": {"color": {"mode": "fixed", "fixedColor": "blue"}}},
         "targets": [{"expr": "api_requests_total", "legendFormat": "Requests"}]},

        {"id": 2, "type": "stat", "gridPos": {"x": 4, "y": 0, "w": 4, "h": 4},
         "title": "Cat Predictions", "datasource": ds(),
         "options": {"colorMode": "background", "reduceOptions": {"calcs": ["lastNotNull"]}},
         "fieldConfig": {"defaults": {"color": {"mode": "fixed", "fixedColor": "orange"}}},
         "targets": [{"expr": 'api_predictions_total{label="cat"}', "legendFormat": "Cats"}]},

        {"id": 3, "type": "stat", "gridPos": {"x": 8, "y": 0, "w": 4, "h": 4},
         "title": "Dog Predictions", "datasource": ds(),
         "options": {"colorMode": "background", "reduceOptions": {"calcs": ["lastNotNull"]}},
         "fieldConfig": {"defaults": {"color": {"mode": "fixed", "fixedColor": "green"}}},
         "targets": [{"expr": 'api_predictions_total{label="dog"}', "legendFormat": "Dogs"}]},

        {"id": 4, "type": "stat", "gridPos": {"x": 12, "y": 0, "w": 4, "h": 4},
         "title": "Model Accuracy", "datasource": ds(),
         "options": {"colorMode": "background", "reduceOptions": {"calcs": ["lastNotNull"]}},
         "fieldConfig": {"defaults": {"unit": "percentunit",
            "color": {"mode": "thresholds"},
            "thresholds": {"steps": [{"color": "red", "value": None},
                                     {"color": "yellow", "value": 0.80},
                                     {"color": "green", "value": 0.90}]}}},
         "targets": [{"expr": "model_accuracy", "legendFormat": "Accuracy"}]},

        {"id": 5, "type": "stat", "gridPos": {"x": 16, "y": 0, "w": 4, "h": 4},
         "title": "Model Confidence", "datasource": ds(),
         "options": {"colorMode": "background", "reduceOptions": {"calcs": ["lastNotNull"]}},
         "fieldConfig": {"defaults": {"unit": "percentunit",
            "color": {"mode": "thresholds"},
            "thresholds": {"steps": [{"color": "red", "value": None},
                                     {"color": "yellow", "value": 0.75},
                                     {"color": "green", "value": 0.85}]}}},
         "targets": [{"expr": "model_confidence", "legendFormat": "Confidence"}]},

        {"id": 6, "type": "stat", "gridPos": {"x": 20, "y": 0, "w": 4, "h": 4},
         "title": "Service Uptime", "datasource": ds(),
         "options": {"colorMode": "background", "reduceOptions": {"calcs": ["lastNotNull"]}},
         "fieldConfig": {"defaults": {"unit": "s", "color": {"mode": "fixed", "fixedColor": "purple"}}},
         "targets": [{"expr": "service_uptime_seconds", "legendFormat": "Uptime"}]},

        {"id": 7, "type": "timeseries", "gridPos": {"x": 0, "y": 4, "w": 12, "h": 8},
         "title": "Requests Per Minute", "datasource": ds(),
         "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"},
                                      "custom": {"lineWidth": 2, "fillOpacity": 12}}},
         "options": {"tooltip": {"mode": "multi"}},
         "targets": [{"expr": "requests_per_minute", "legendFormat": "Req/min"}]},

        {"id": 8, "type": "timeseries", "gridPos": {"x": 12, "y": 4, "w": 12, "h": 8},
         "title": "API Latency - Avg and P95", "datasource": ds(),
         "fieldConfig": {"defaults": {"unit": "s", "custom": {"lineWidth": 2, "fillOpacity": 10}}},
         "options": {"tooltip": {"mode": "multi"}},
         "targets": [
             {"expr": "api_request_latency_seconds", "legendFormat": "Avg Latency"},
             {"expr": "api_request_latency_p95_seconds", "legendFormat": "P95 Latency"},
         ]},

        {"id": 9, "type": "piechart", "gridPos": {"x": 0, "y": 12, "w": 8, "h": 8},
         "title": "Prediction Distribution", "datasource": ds(),
         "options": {"pieType": "pie", "displayLabels": ["name", "percent"]},
         "fieldConfig": {"defaults": {"color": {"mode": "palette-classic"}}},
         "targets": [
             {"expr": 'api_predictions_total{label="cat"}', "legendFormat": "Cat"},
             {"expr": 'api_predictions_total{label="dog"}', "legendFormat": "Dog"},
         ]},

        {"id": 10, "type": "timeseries", "gridPos": {"x": 8, "y": 12, "w": 8, "h": 8},
         "title": "Model Accuracy Over Time", "datasource": ds(),
         "fieldConfig": {"defaults": {"unit": "percentunit", "min": 0.70, "max": 1.0,
                                      "color": {"mode": "fixed", "fixedColor": "green"},
                                      "custom": {"lineWidth": 2, "fillOpacity": 10}}},
         "targets": [{"expr": "model_accuracy", "legendFormat": "Accuracy"}]},

        {"id": 11, "type": "timeseries", "gridPos": {"x": 16, "y": 12, "w": 8, "h": 8},
         "title": "Model Confidence Over Time", "datasource": ds(),
         "fieldConfig": {"defaults": {"unit": "percentunit", "min": 0.60, "max": 1.0,
                                      "color": {"mode": "fixed", "fixedColor": "blue"},
                                      "custom": {"lineWidth": 2, "fillOpacity": 10}}},
         "targets": [{"expr": "model_confidence", "legendFormat": "Confidence"}]},

        {"id": 12, "type": "timeseries", "gridPos": {"x": 0, "y": 20, "w": 12, "h": 6},
         "title": "API Errors Per Minute", "datasource": ds(),
         "fieldConfig": {"defaults": {"color": {"mode": "fixed", "fixedColor": "red"},
                                      "custom": {"lineWidth": 2, "fillOpacity": 20}}},
         "targets": [{"expr": "increase(api_errors_total[1m])", "legendFormat": "Errors/min"}]},

        {"id": 13, "type": "stat", "gridPos": {"x": 12, "y": 20, "w": 4, "h": 3},
         "title": "Current Hour", "datasource": ds(),
         "options": {"colorMode": "background", "reduceOptions": {"calcs": ["lastNotNull"]}},
         "fieldConfig": {"defaults": {"color": {"mode": "fixed", "fixedColor": "purple"},
            "mappings": [
                {"type": "range", "options": {"from": 0,  "to": 5,  "result": {"text": "Night"}}},
                {"type": "range", "options": {"from": 6,  "to": 11, "result": {"text": "Morning"}}},
                {"type": "range", "options": {"from": 12, "to": 17, "result": {"text": "Afternoon"}}},
                {"type": "range", "options": {"from": 18, "to": 23, "result": {"text": "Evening"}}},
            ]}},
         "targets": [{"expr": "current_hour", "legendFormat": "Hour"}]},

        {"id": 14, "type": "stat", "gridPos": {"x": 16, "y": 20, "w": 4, "h": 3},
         "title": "Day of Week", "datasource": ds(),
         "options": {"colorMode": "background", "reduceOptions": {"calcs": ["lastNotNull"]}},
         "fieldConfig": {"defaults": {"color": {"mode": "fixed", "fixedColor": "teal"},
            "mappings": [{"type": "value", "options": {
                "0": {"text": "Monday"}, "1": {"text": "Tuesday"},
                "2": {"text": "Wednesday"}, "3": {"text": "Thursday"},
                "4": {"text": "Friday"}, "5": {"text": "Saturday"},
                "6": {"text": "Sunday"},
            }}]}},
         "targets": [{"expr": "current_day_of_week", "legendFormat": "Day"}]},

        {"id": 15, "type": "stat", "gridPos": {"x": 20, "y": 20, "w": 4, "h": 3},
         "title": "Last Metric Push", "datasource": ds(),
         "options": {"colorMode": "background", "reduceOptions": {"calcs": ["lastNotNull"]}},
         "fieldConfig": {"defaults": {"unit": "dateTimeAsIso",
                                      "color": {"mode": "fixed", "fixedColor": "green"}}},
         "targets": [{"expr": "data_version_timestamp", "legendFormat": "Timestamp"}]},

        {"id": 16, "type": "timeseries", "gridPos": {"x": 0, "y": 26, "w": 24, "h": 7},
         "title": "Cat vs Dog Prediction Trend", "datasource": ds(),
         "fieldConfig": {"defaults": {"unit": "short",
                                      "custom": {"lineWidth": 2, "fillOpacity": 8}}},
         "options": {"tooltip": {"mode": "multi"}, "legend": {
             "displayMode": "table", "placement": "right",
             "calcs": ["last", "max", "mean"]}},
         "targets": [
             {"expr": 'api_predictions_total{label="cat"}', "legendFormat": "Cats (cumulative)"},
             {"expr": 'api_predictions_total{label="dog"}', "legendFormat": "Dogs (cumulative)"},
         ]},
    ]

    payload = {
        "dashboard": {
            "id": None,
            "uid": "pet-adoption-ml-v2",
            "title": "Pet Adoption Platform - ML Monitoring",
            "tags": ["mlops", "pet-adoption", "ml", "fastapi"],
            "timezone": "browser",
            "refresh": "10s",
            "time": {"from": "now-1h", "to": "now"},
            "panels": panels,
            "schemaVersion": 37,
            "version": 1,
        },
        "folderId": 0,
        "overwrite": True,
        "message": "Created by setup_grafana.py",
    }

    status, resp = request("POST", "/api/dashboards/db", payload)
    if status == 200:
        url = resp.get("url", "")
        print(f"  Dashboard created! URL: http://localhost:3000{url}")
        return url
    else:
        print(f"  Failed (status {status}): {resp}")
        return None


if __name__ == "__main__":
    print("\n=== Pet Adoption Platform - Grafana Setup ===\n")

    if not wait_for_grafana():
        print("Grafana is not available. Start it first.")
        sys.exit(1)

    time.sleep(2)
    ds_uid = add_datasource()
    time.sleep(1)
    url = create_dashboard(ds_uid)

    print("\n=== Setup Complete! ===")
    print("  Grafana:   http://localhost:3000")
    print("  Login:     admin / admin")
    print("  Dashboard: http://localhost:3000/d/pet-adoption-ml-v2")
