#!/bin/bash

echo "Applying Kubernetes manifests..."

# Make sure you're in the root of your repo or adjust the path accordingly
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

echo "Waiting for pods to be ready..."
kubectl rollout status deployment/dr-dialog

echo "Deployment complete. Current pod status:"
kubectl get pods

echo "Service info:"
kubectl get svc dr-dialog-service

NODE_PORT=$(kubectl get svc dr-dialog-service -o=jsonpath='{.spec.ports[0].nodePort}')
EXTERNAL_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 || hostname -I | awk '{print $1}')

echo ""
echo "Your app should be accessible at:"
echo "  http://$EXTERNAL_IP:$NODE_PORT"
