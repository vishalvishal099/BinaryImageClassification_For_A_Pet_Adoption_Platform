#!/bin/bash
# =============================================================================
# Local K8s Deployment Script with Auto-Refresh
# =============================================================================
# This script deploys the app to local Minikube and sets up auto-refresh
# when new images are pushed to the registry.
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

NAMESPACE="cats-dogs-classifier"
APP_NAME="cats-dogs-classifier"
IMAGE="ghcr.io/vishalvishal099/cats-dogs-classifier:latest"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       🐕 Cats vs Dogs Classifier - Local K8s Deploy 🐈       ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"

# Function to check if minikube is running
check_minikube() {
    echo -e "\n${YELLOW}[1/5] Checking Minikube status...${NC}"
    if ! minikube status | grep -q "Running"; then
        echo -e "${RED}Minikube is not running. Starting...${NC}"
        minikube start --driver=podman --container-runtime=containerd
    fi
    echo -e "${GREEN}✓ Minikube is running${NC}"
}

# Function to apply K8s manifests
apply_manifests() {
    echo -e "\n${YELLOW}[2/5] Applying K8s manifests...${NC}"
    
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    K8S_DIR="$SCRIPT_DIR/../k8s/local"
    
    # Apply in order
    kubectl apply -f "$K8S_DIR/namespace.yaml"
    kubectl apply -f "$K8S_DIR/configmap.yaml"
    kubectl apply -f "$K8S_DIR/deployment.yaml"
    kubectl apply -f "$K8S_DIR/service.yaml"
    
    echo -e "${GREEN}✓ Manifests applied${NC}"
}

# Function to wait for deployment
wait_for_deployment() {
    echo -e "\n${YELLOW}[3/5] Waiting for deployment to be ready...${NC}"
    kubectl wait --for=condition=available deployment/$APP_NAME \
        -n $NAMESPACE --timeout=120s || {
        echo -e "${RED}Deployment not ready. Checking pod status...${NC}"
        kubectl get pods -n $NAMESPACE
        kubectl describe pods -n $NAMESPACE
        return 1
    }
    echo -e "${GREEN}✓ Deployment is ready${NC}"
}

# Function to show service info
show_service_info() {
    echo -e "\n${YELLOW}[4/5] Getting service URL...${NC}"
    
    # Get the minikube service URL
    SERVICE_URL=$(minikube service $APP_NAME -n $NAMESPACE --url 2>/dev/null || echo "")
    
    if [ -n "$SERVICE_URL" ]; then
        echo -e "${GREEN}✓ Service is available at: ${BLUE}$SERVICE_URL${NC}"
    else
        echo -e "${YELLOW}Getting NodePort...${NC}"
        NODE_PORT=$(kubectl get svc $APP_NAME -n $NAMESPACE -o jsonpath='{.spec.ports[0].nodePort}')
        MINIKUBE_IP=$(minikube ip)
        echo -e "${GREEN}✓ Service is available at: ${BLUE}http://$MINIKUBE_IP:$NODE_PORT${NC}"
    fi
}

# Function to show deployment status
show_status() {
    echo -e "\n${YELLOW}[5/5] Deployment Status:${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    echo -e "\n${GREEN}Pods:${NC}"
    kubectl get pods -n $NAMESPACE -o wide
    
    echo -e "\n${GREEN}Services:${NC}"
    kubectl get svc -n $NAMESPACE
    
    echo -e "\n${GREEN}Current Image:${NC}"
    kubectl get deployment $APP_NAME -n $NAMESPACE -o jsonpath='{.spec.template.spec.containers[0].image}'
    echo ""
}

# Function to force pull latest image
refresh_deployment() {
    echo -e "\n${YELLOW}🔄 Refreshing deployment to pull latest image...${NC}"
    
    # Rollout restart forces K8s to pull the latest image (due to imagePullPolicy: Always)
    kubectl rollout restart deployment/$APP_NAME -n $NAMESPACE
    
    echo -e "${GREEN}✓ Rollout restart initiated${NC}"
    
    # Wait for rollout to complete
    kubectl rollout status deployment/$APP_NAME -n $NAMESPACE --timeout=120s
    
    echo -e "${GREEN}✓ Deployment refreshed with latest image${NC}"
}

# Function to tail logs
tail_logs() {
    echo -e "\n${YELLOW}📋 Tailing pod logs (Ctrl+C to stop)...${NC}"
    kubectl logs -f deployment/$APP_NAME -n $NAMESPACE
}

# Main execution
case "${1:-deploy}" in
    deploy)
        check_minikube
        apply_manifests
        wait_for_deployment
        show_service_info
        show_status
        ;;
    refresh|update)
        refresh_deployment
        show_status
        ;;
    status)
        show_status
        ;;
    logs)
        tail_logs
        ;;
    url)
        minikube service $APP_NAME -n $NAMESPACE --url
        ;;
    delete)
        echo -e "${YELLOW}Deleting deployment...${NC}"
        kubectl delete namespace $NAMESPACE --ignore-not-found
        echo -e "${GREEN}✓ Deployment deleted${NC}"
        ;;
    *)
        echo "Usage: $0 {deploy|refresh|status|logs|url|delete}"
        echo ""
        echo "Commands:"
        echo "  deploy   - Deploy the app to local K8s (default)"
        echo "  refresh  - Pull latest image and restart pods"
        echo "  status   - Show deployment status"
        echo "  logs     - Tail pod logs"
        echo "  url      - Get service URL"
        echo "  delete   - Delete the deployment"
        exit 1
        ;;
esac

echo -e "\n${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    🎉 Operation Complete!                    ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
