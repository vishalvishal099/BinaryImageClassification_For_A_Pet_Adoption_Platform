#!/bin/bash
# =============================================================================
# MLOps Platform - Stop All Services
# Binary Image Classification for Pet Adoption Platform
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✅ $1${NC}"; }
warn() { echo -e "  ${YELLOW}⚠️  $1${NC}"; }
step() { echo -e "\n${CYAN}${BOLD}[$1]${NC}${BOLD} $2${NC}"; }

echo -e "${CYAN}${BOLD}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║           🛑  Stopping MLOps Platform Services  🛑                ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# =============================================================================
# [1/5] Kill all port-forwards
# =============================================================================
step "1/5" "Stopping port-forwards..."
pkill -f "port-forward" 2>/dev/null && ok "All port-forwards stopped" || ok "No port-forwards running"

# =============================================================================
# [2/5] Stop Podman containers
# =============================================================================
step "2/5" "Stopping Podman containers..."
for container in prometheus grafana mlflow cats-dogs-api; do
    if podman ps --format "{{.Names}}" 2>/dev/null | grep -q "^${container}$"; then
        podman stop "$container" > /dev/null 2>&1
        ok "Stopped: $container"
    else
        echo -e "  ${YELLOW}–${NC}  $container not running"
    fi
done

# =============================================================================
# [3/5] Stop Homebrew Grafana (if used)
# =============================================================================
step "3/5" "Stopping Homebrew Grafana (if running)..."
if command -v brew &>/dev/null && brew services list 2>/dev/null | grep -q "grafana.*started"; then
    brew services stop grafana > /dev/null 2>&1
    ok "Homebrew Grafana stopped"
else
    echo -e "  ${YELLOW}–${NC}  Homebrew Grafana not running"
fi

# =============================================================================
# [4/5] Kill kubectl processes
# =============================================================================
step "4/5" "Killing kubectl background processes..."
pkill -f "kubectl" 2>/dev/null && ok "kubectl processes killed" || ok "No kubectl processes running"

# =============================================================================
# [5/5] Stop Minikube
# =============================================================================
step "5/5" "Stopping Minikube..."
if minikube status 2>/dev/null | grep -q "Running\|apiserver: Running"; then
    minikube stop > /dev/null 2>&1 &
    ok "Minikube stopping in background"
else
    ok "Minikube already stopped"
fi

echo -e "\n${GREEN}${BOLD}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                   ✅  All Services Stopped                        ║"
echo "╠═══════════════════════════════════════════════════════════════════╣"
echo "║  To start again:  ./scripts/start_all.sh                         ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
