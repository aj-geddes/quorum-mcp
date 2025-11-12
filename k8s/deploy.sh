#!/bin/bash
set -e

echo "🚀 Deploying Quorum-MCP to Kubernetes..."

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed or not in PATH"
    exit 1
fi

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH"
    exit 1
fi

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "📦 Building Docker image..."
docker build -t quorum-mcp:latest .

echo "📥 Importing image to k3d cluster (if applicable)..."
if command -v k3d &> /dev/null; then
    # Detect k3d cluster name
    CLUSTER_NAME=$(kubectl config current-context | sed 's/k3d-//')
    if [[ "$CLUSTER_NAME" != "$(kubectl config current-context)" ]]; then
        k3d image import quorum-mcp:latest -c "$CLUSTER_NAME" || echo "⚠️  Failed to import image to k3d cluster, continuing anyway..."
    fi
fi

echo "📝 Checking for secret.yaml..."
if [ ! -f "k8s/secret.yaml" ]; then
    echo "⚠️  secret.yaml not found. Creating from template..."
    echo "⚠️  Please edit k8s/secret.yaml with your actual API keys before deployment!"
    cp k8s/secret.yaml.template k8s/secret.yaml
    echo "❌ Deployment aborted. Please configure k8s/secret.yaml with your API keys."
    exit 1
fi

echo "🏗️  Creating namespace..."
kubectl apply -f k8s/namespace.yaml

echo "📋 Creating ConfigMap..."
kubectl apply -f k8s/configmap.yaml

echo "🔐 Creating Secrets..."
kubectl apply -f k8s/secret.yaml

echo "🚢 Creating Deployment..."
kubectl apply -f k8s/deployment.yaml

echo "🌐 Creating Service..."
kubectl apply -f k8s/service.yaml

echo "🔀 Creating Ingress..."
kubectl apply -f k8s/ingress.yaml

echo "⏳ Waiting for deployment to be ready..."
kubectl wait --for=condition=available --timeout=300s deployment/quorum-mcp -n quorum-mcp

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Deployment status:"
kubectl get pods -n quorum-mcp
echo ""
kubectl get svc -n quorum-mcp
echo ""
kubectl get ingress -n quorum-mcp
echo ""
echo "🌐 Add the following to your /etc/hosts file:"
echo "127.0.0.1 quorum-mcp.local"
echo ""
echo "🎉 Access the Web UI at: http://quorum-mcp.local"
echo ""
echo "📝 To view logs:"
echo "   kubectl logs -f -n quorum-mcp -l app=quorum-mcp"
echo ""
echo "🔍 To check health:"
echo "   curl http://quorum-mcp.local/api/health"
