#!/bin/bash
# =============================================================================
#  CD Deploy Script — One command to deploy latest code to Minikube
#  Usage: ./scripts/deploy.sh
# =============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

NAMESPACE="cats-dogs-classifier"
DEPLOYMENT="cats-dogs-classifier"
IMAGE="localhost/cats-dogs-classifier:latest"
GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
START_TIME=$(date +%s)

# ─────────────────────────────────────────────────────
#  Banner
# ─────────────────────────────────────────────────────
echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════╗"
echo "║        🚀 CD Pipeline — Deploy to Minikube K8s           ║"
echo "╠══════════════════════════════════════════════════════════╣"
printf "║  Git SHA  : %-44s ║\n" "$GIT_SHA"
printf "║  Image    : %-44s ║\n" "$IMAGE"
printf "║  Namespace: %-44s ║\n" "$NAMESPACE"
echo "╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

step() { echo -e "\n${BLUE}[$1/4]${NC} ${BOLD}$2${NC}"; }
ok()   { echo -e "  ${GREEN}✅ $1${NC}"; }
fail() { echo -e "  ${RED}❌ $1${NC}"; exit 1; }
info() { echo -e "  ${YELLOW}→  $1${NC}"; }

# ─────────────────────────────────────────────────────
#  STEP 1 — Build image
# ─────────────────────────────────────────────────────
step 1 "Building Docker image"
info "Running: podman build -t $IMAGE ."
podman build -t "$IMAGE" . --quiet && ok "Image built: $IMAGE" || fail "Image build failed"

# ─────────────────────────────────────────────────────
#  STEP 2 — Load into Minikube
# ─────────────────────────────────────────────────────
step 2 "Loading image into Minikube"
info "Piping image into minikube cache..."
podman save "$IMAGE" | minikube image load --overwrite=true - \
  && ok "Image loaded into Minikube" || fail "Failed to load image into Minikube"

# ─────────────────────────────────────────────────────
#  STEP 3 — Annotate + Rollout restart
# ─────────────────────────────────────────────────────
step 3 "Triggering K8s rolling update"
info "Annotating deployment with git-sha=$GIT_SHA..."
kubectl annotate deployment "$DEPLOYMENT" -n "$NAMESPACE" \
  git-sha="$GIT_SHA" \
  deployed-at="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --overwrite

info "Restarting deployment..."
kubectl rollout restart deployment/"$DEPLOYMENT" -n "$NAMESPACE"
ok "Rolling update triggered"

# ─────────────────────────────────────────────────────
#  STEP 4 — Wait and verify
# ─────────────────────────────────────────────────────
step 4 "Waiting for rollout to complete"
info "Watching rollout status (timeout: 90s)..."
kubectl rollout status deployment/"$DEPLOYMENT" -n "$NAMESPACE" --timeout=90s \
  && ok "Rollout complete — new pod is live" || fail "Rollout timed out or failed"

# ─────────────────────────────────────────────────────
#  Summary
# ─────────────────────────────────────────────────────
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════════════════════╗"
echo    "║                  ✅ DEPLOYMENT COMPLETE                   ║"
echo    "╠══════════════════════════════════════════════════════════╣"
printf  "║  Git SHA   : %-43s ║\n" "$GIT_SHA"
printf  "║  Duration  : %-43s ║\n" "${ELAPSED}s"
echo    "╠══════════════════════════════════════════════════════════╣"
echo    "║  📊 Visualize:                                           ║"
echo    "║  K8s Dashboard → http://localhost:8001/api/v1/namespaces ║"
echo    "║    /kubernetes-dashboard/services/http:kubernetes-       ║"
echo    "║    dashboard:/proxy/#/workloads?namespace=cats-dogs-     ║"
echo    "║    classifier                                             ║"
echo    "║  Grafana       → http://localhost:3000                   ║"
echo    "║  API Health    → http://localhost:8081/health            ║"
echo -e "╚══════════════════════════════════════════════════════════╝${NC}"

# Show live pod status
echo ""
echo -e "${CYAN}Live Pod Status:${NC}"
kubectl get pods -n "$NAMESPACE" -o wide
