# Kubernetes Deployment Guide

This directory contains Kubernetes manifests for deploying Quorum-MCP to a local Kubernetes cluster.

## Prerequisites

- Local Kubernetes cluster (e.g., minikube, kind, Docker Desktop with Kubernetes)
- kubectl configured to access your cluster
- Docker installed
- NGINX Ingress Controller installed in your cluster

### Installing NGINX Ingress Controller

If you don't have an ingress controller installed:

```bash
# For minikube
minikube addons enable ingress

# For kind
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml

# For Docker Desktop or other clusters
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/cloud/deploy.yaml
```

## Quick Start

1. **Configure API Keys**

   Copy the secret template and add your API keys:
   ```bash
   cp k8s/secret.yaml.template k8s/secret.yaml
   # Edit k8s/secret.yaml with your actual API keys
   ```

2. **Deploy Everything**

   Run the deployment script:
   ```bash
   chmod +x k8s/deploy.sh
   ./k8s/deploy.sh
   ```

3. **Add to /etc/hosts**

   Add this line to your `/etc/hosts` file:
   ```
   127.0.0.1 quorum-mcp.local
   ```

4. **Access the Web UI**

   Open your browser to: http://quorum-mcp.local

## Manual Deployment

If you prefer to deploy manually:

```bash
# Build Docker image
docker build -t quorum-mcp:latest .

# Create namespace
kubectl apply -f k8s/namespace.yaml

# Create ConfigMap
kubectl apply -f k8s/configmap.yaml

# Create Secrets (make sure you've configured secret.yaml first!)
kubectl apply -f k8s/secret.yaml

# Deploy the application
kubectl apply -f k8s/deployment.yaml

# Create Service
kubectl apply -f k8s/service.yaml

# Create Ingress
kubectl apply -f k8s/ingress.yaml
```

## Configuration Files

- `namespace.yaml` - Creates the quorum-mcp namespace
- `configmap.yaml` - Non-sensitive configuration (Ollama settings, logging, rate limits)
- `secret.yaml.template` - Template for API keys (DO NOT commit secret.yaml!)
- `deployment.yaml` - Main application deployment (2 replicas)
- `service.yaml` - ClusterIP service exposing the application
- `ingress.yaml` - Ingress routing quorum-mcp.local to the service

## Verifying Deployment

Check pod status:
```bash
kubectl get pods -n quorum-mcp
```

View logs:
```bash
kubectl logs -f -n quorum-mcp -l app=quorum-mcp
```

Check service:
```bash
kubectl get svc -n quorum-mcp
```

Check ingress:
```bash
kubectl get ingress -n quorum-mcp
```

Test health endpoint:
```bash
curl http://quorum-mcp.local/api/health
```

## Updating Configuration

### Update ConfigMap

Edit `configmap.yaml` and apply:
```bash
kubectl apply -f k8s/configmap.yaml
kubectl rollout restart deployment/quorum-mcp -n quorum-mcp
```

### Update Secrets

Edit `secret.yaml` and apply:
```bash
kubectl apply -f k8s/secret.yaml
kubectl rollout restart deployment/quorum-mcp -n quorum-mcp
```

### Update Application

Rebuild image and restart:
```bash
docker build -t quorum-mcp:latest .
kubectl rollout restart deployment/quorum-mcp -n quorum-mcp
```

## Scaling

Scale the deployment:
```bash
kubectl scale deployment/quorum-mcp --replicas=3 -n quorum-mcp
```

## Troubleshooting

### Pods not starting

Check pod logs:
```bash
kubectl logs -n quorum-mcp -l app=quorum-mcp
```

Check pod events:
```bash
kubectl describe pod -n quorum-mcp -l app=quorum-mcp
```

### Ingress not working

Check ingress controller is running:
```bash
kubectl get pods -n ingress-nginx
```

Verify ingress configuration:
```bash
kubectl describe ingress quorum-mcp -n quorum-mcp
```

### Can't access quorum-mcp.local

1. Verify /etc/hosts entry:
   ```bash
   cat /etc/hosts | grep quorum-mcp
   ```

2. Check if ingress has an address:
   ```bash
   kubectl get ingress -n quorum-mcp
   ```

3. Test direct service access:
   ```bash
   kubectl port-forward -n quorum-mcp svc/quorum-mcp 8000:80
   # Then visit http://localhost:8000
   ```

## Uninstalling

Remove all resources:
```bash
kubectl delete namespace quorum-mcp
```

Or remove individually:
```bash
kubectl delete -f k8s/ingress.yaml
kubectl delete -f k8s/service.yaml
kubectl delete -f k8s/deployment.yaml
kubectl delete -f k8s/secret.yaml
kubectl delete -f k8s/configmap.yaml
kubectl delete -f k8s/namespace.yaml
```

## Resource Requirements

Each pod requests:
- CPU: 500m (0.5 cores)
- Memory: 512Mi

Each pod limits:
- CPU: 2000m (2 cores)
- Memory: 2Gi

Adjust these in `deployment.yaml` based on your cluster capacity.

## Production Considerations

For production deployments, consider:

1. **Persistent Storage** - Add volumes for session data
2. **Horizontal Pod Autoscaling** - Scale based on CPU/memory
3. **Resource Limits** - Tune based on workload
4. **TLS/HTTPS** - Configure cert-manager for TLS certificates
5. **Monitoring** - Add Prometheus metrics and Grafana dashboards
6. **Logging** - Configure centralized logging (e.g., ELK stack)
7. **Network Policies** - Restrict pod-to-pod communication
8. **Pod Disruption Budgets** - Ensure availability during updates
