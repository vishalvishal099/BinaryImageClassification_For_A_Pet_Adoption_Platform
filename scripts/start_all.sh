#!/bin/bash
# =============================================================================
# MLOps Platform - Start All Services
# Binary Image Classification for Pet Adoption Platform
# =============================================================================
# Services started:
#   [1] Minikube (K8s cluster)
#   [2] Prometheus (metrics scraping)
#   [3] Grafana (dashboards)
#   [4] K8s Deployment + port-forward
#   [5] ArgoCD (GitOps CD)
#   [6] Traffic generator (populate Grafana)
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Configuration
K8S_PORT=8081
GRAFANA_PORT=3000
PROMETHEUS_PORT=9090
ARGOCD_PORT=8082

echo -e "${CYAN}${BOLD}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║     🐱🐶  MLOps Platform - Binary Image Classification  🐶🐱      ║"
echo "║              Pet Adoption Platform — Full Stack Start              ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ─── Helpers ──────────────────────────────────────────────────────────────────
ok()   { echo -e "  ${GREEN}✅ $1${NC}"; }
warn() { echo -e "  ${YELLOW}⚠️  $1${NC}"; }
info() { echo -e "  ${BLUE}→  $1${NC}"; }
step() { echo -e "\n${CYAN}${BOLD}[$1]${NC}${BOLD} $2${NC}"; }

wait_for_service() {
    local url=$1 name=$2 max=30 i=1
    while [ $i -le $max ]; do
        if curl -s "$url" > /dev/null 2>&1; then ok "$name is ready"; return 0; fi
        sleep 2; ((i++))
    done
    warn "$name not ready yet (continuing)"
}

# =============================================================================
# [1/6] Minikube
# =============================================================================
step "1/6" "Starting Minikube Kubernetes Cluster..."

if minikube status 2>/dev/null | grep -q "apiserver: Running"; then
    ok "Minikube already running"
else
    info "Starting Minikube with Podman driver..."
    minikube start --driver=podman 2>&1 | tail -3
    ok "Minikube started"
fi

# =============================================================================
# [2/6] Prometheus
# =============================================================================
step "2/6" "Starting Prometheus..."

if podman ps --format "{{.Names}}" 2>/dev/null | grep -q "^prometheus$"; then
    ok "Prometheus already running"
else
    podman run -d \
        --name prometheus \
        -p ${PROMETHEUS_PORT}:9090 \
        -v "${PROJECT_DIR}/monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
        docker.io/prom/prometheus:latest > /dev/null 2>&1
    wait_for_service "http://localhost:${PROMETHEUS_PORT}/-/healthy" "Prometheus"
fi

# =============================================================================
# [3/6] Grafana
# =============================================================================
step "3/6" "Starting Grafana..."

if curl -s http://localhost:${GRAFANA_PORT}/api/health > /dev/null 2>&1; then
    ok "Grafana already running"
else
    # Try Homebrew Grafana first, fallback to Podman
    if command -v grafana-server &>/dev/null; then
        brew services start grafana > /dev/null 2>&1 || true
    else
        podman run -d \
            --name grafana \
            -p ${GRAFANA_PORT}:3000 \
            -e GF_SECURITY_ADMIN_PASSWORD=admin \
            docker.io/grafana/grafana:latest > /dev/null 2>&1 || true
    fi
    wait_for_service "http://localhost:${GRAFANA_PORT}/api/health" "Grafana"
fi

# Setup Grafana datasource + dashboard
info "Configuring Grafana datasource and dashboard..."
python3 "${PROJECT_DIR}/scripts/setup_grafana.py" && ok "Grafana dashboard configured"

# =============================================================================
# [4/6] K8s Deployment + port-forward
# =============================================================================
step "4/6" "Deploying to Kubernetes..."

# Apply manifests
kubectl apply -f k8s/local/ > /dev/null 2>&1 && ok "K8s manifests applied"

# Wait for pod
info "Waiting for pod to be ready..."
kubectl wait --for=condition=ready pod \
    -l app=cats-dogs-classifier \
    -n cats-dogs-classifier \
    --timeout=90s > /dev/null 2>&1 && ok "Pod is running" || warn "Pod not ready yet"

# Port-forward K8s API → localhost:8081
pkill -f "port-forward.*${K8S_PORT}" 2>/dev/null || true
sleep 1
nohup kubectl port-forward -n cats-dogs-classifier \
    svc/cats-dogs-classifier ${K8S_PORT}:8000 \
    > /tmp/k8s-pf.log 2>&1 &
sleep 3
wait_for_service "http://localhost:${K8S_PORT}/health" "K8s API (port-forward)"

# =============================================================================
# [5/6] ArgoCD
# =============================================================================
step "5/6" "Starting ArgoCD CD Pipeline..."

# Check if ArgoCD is installed
if kubectl get namespace argocd > /dev/null 2>&1; then
    # Wait for argocd-server pod
    ARGOCD_READY=$(kubectl get pods -n argocd -l app.kubernetes.io/name=argocd-server \
        --no-headers 2>/dev/null | grep "Running" | wc -l | tr -d ' ')

    if [ "$ARGOCD_READY" -gt "0" ]; then
        ok "ArgoCD pods running"
    else
        warn "ArgoCD pods not ready — waiting 30s..."
        sleep 30
    fi

    # Port-forward ArgoCD UI → localhost:8082
    pkill -f "port-forward.*argocd-server" 2>/dev/null || true
    pkill -f "port-forward.*${ARGOCD_PORT}" 2>/dev/null || true
    sleep 1
    nohup kubectl port-forward svc/argocd-server \
        -n argocd ${ARGOCD_PORT}:80 \
        > /tmp/argocd-pf.log 2>&1 &
    sleep 3

    if curl -s http://localhost:${ARGOCD_PORT} > /dev/null 2>&1; then
        ok "ArgoCD UI accessible at http://localhost:${ARGOCD_PORT}"

        # Get admin password
        ARGOCD_PASS=$(kubectl -n argocd get secret argocd-initial-admin-secret \
            -o jsonpath="{.data.password}" 2>/dev/null | base64 -d 2>/dev/null || echo "check-secret")
        info "ArgoCD login → user: admin  |  password: ${ARGOCD_PASS}"

        # Trigger a refresh of the ArgoCD app
        kubectl -n argocd annotate application cats-dogs-classifier \
            argocd.argoproj.io/refresh=hard --overwrite > /dev/null 2>&1 || true
        ok "ArgoCD sync refresh triggered"
    else
        warn "ArgoCD UI not yet reachable — run manually: kubectl port-forward svc/argocd-server -n argocd ${ARGOCD_PORT}:80 &"
    fi
else
    warn "ArgoCD not installed — skipping (run scripts/install_argocd.sh to install)"
fi

# =============================================================================
# [6/6] Traffic Generator — populate Grafana with real data
# =============================================================================
step "6/6" "Generating traffic to populate Grafana dashboards..."

info "Sending 60 prediction requests (cat + dog) to K8s API..."

python3 - <<'PYEOF'
import urllib.request
import urllib.error
import json
import os
import struct
import zlib
import time
import random
import sys

API_URL = "http://localhost:8081/predict"
HEALTH_URL = "http://localhost:8081/health"

# ── Verify API is up ──────────────────────────────────────────────────────────
try:
    with urllib.request.urlopen(HEALTH_URL, timeout=5) as r:
        health = json.loads(r.read())
        if health.get("status") != "healthy":
            print("  ⚠️  API not healthy, skipping traffic generation")
            sys.exit(0)
except Exception as e:
    print(f"  ⚠️  API not reachable: {e} — skipping traffic generation")
    sys.exit(0)

# ── PNG generator (no Pillow needed) ─────────────────────────────────────────
def make_png(r, g, b, size=64):
    """Build a valid RGB PNG in memory without any dependencies."""
    def chunk(name, data):
        c = zlib.crc32(name + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + name + data + struct.pack(">I", c)

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    raw = b""
    for _ in range(size):
        row = b"\x00"  # filter byte
        for _ in range(size):
            noise = random.randint(-15, 15)
            row += bytes([
                max(0, min(255, r + noise)),
                max(0, min(255, g + noise)),
                max(0, min(255, b + noise))
            ])
        raw += row

    compressed = zlib.compress(raw, 6)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )
    return png

def send_request(png_bytes, label, i, total):
    boundary = "----PythonBoundary"
    filename = f"test_{label}_{i}.png"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + png_bytes + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        API_URL,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            cls  = result.get("predicted_class", "?")
            conf = result.get("confidence", 0)
            bar  = "█" * int(conf * 20)
            print(f"  [{i:>3}/{total}] {label:>3} → {cls:<3}  {conf:.2f}  {bar}")
            return True
    except Exception as e:
        print(f"  [{i:>3}/{total}] {label:>3} → ❌ {e}")
        return False

# Cat images: orange-ish tones
CAT_COLORS = [(210, 140, 80), (190, 120, 70), (230, 160, 90), (200, 130, 75)]
# Dog images: brown-ish tones
DOG_COLORS = [(120, 80, 50), (100, 70, 45), (140, 90, 55), (110, 75, 48)]

TOTAL = 60
success = 0
print(f"\n  Sending {TOTAL} requests (30 cat + 30 dog)...\n")

for i in range(1, TOTAL + 1):
    if i % 2 == 0:
        label = "cat"
        r, g, b = random.choice(CAT_COLORS)
    else:
        label = "dog"
        r, g, b = random.choice(DOG_COLORS)

    png = make_png(r, g, b)
    if send_request(png, label, i, TOTAL):
        success += 1
    time.sleep(0.3)  # slight pause so Prometheus captures spread metrics

print(f"\n  ✅ {success}/{TOTAL} requests succeeded")
print(f"  📊 Open Grafana → http://localhost:3000  (admin/admin)")
print(f"  🔄 Dashboard: Cats vs Dogs MLOps\n")
PYEOF

# =============================================================================
# Summary
# =============================================================================
echo -e "\n${CYAN}${BOLD}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                  🎉  All Services Running!  🎉                    ║"
echo "╠═══════════════════════════════════════════════════════════════════╣"
echo "║                                                                   ║"
echo "║  🤖  ML SERVICE                                                   ║"
printf "║  K8s API (predict/health)   → http://localhost:%-4s              ║\n" "${K8S_PORT}"
echo "║                                                                   ║"
echo "║  📊  MONITORING                                                   ║"
printf "║  Grafana  (admin/admin)      → http://localhost:%-4s              ║\n" "${GRAFANA_PORT}"
printf "║  Prometheus                  → http://localhost:%-4s              ║\n" "${PROMETHEUS_PORT}"
echo "║                                                                   ║"
echo "║  �  CD PIPELINE                                                  ║"
printf "║  ArgoCD   (admin/password)   → http://localhost:%-4s              ║\n" "${ARGOCD_PORT}"
echo "║                                                                   ║"
echo "║  �  ARGOCD CREDENTIALS                                           ║"
ARGOCD_PASS=$(kubectl -n argocd get secret argocd-initial-admin-secret \
    -o jsonpath="{.data.password}" 2>/dev/null | base64 -d 2>/dev/null || echo "see kubectl secret")
printf "║  user: admin   password: %-41s ║\n" "${ARGOCD_PASS}"
echo "║                                                                   ║"
echo "║  📝  QUICK TEST                                                   ║"
printf "║  curl http://localhost:%-4s/health                               ║\n" "${K8S_PORT}"
echo "║                                                                   ║"
echo "╠═══════════════════════════════════════════════════════════════════╣"
echo "║  To stop everything:  ./scripts/stop_all.sh                      ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
