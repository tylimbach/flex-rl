#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Deploying Flex-RL infrastructure...${NC}"

# Step 1: Create GKE cluster with Terraform
echo -e "${BLUE}Step 1: Creating GKE cluster with Terraform...${NC}"
cd $(dirname $0)/../terraform/gke-cluster
terraform init
terraform apply -auto-approve

# Get outputs
CLUSTER_NAME=$(terraform output -raw cluster_name)
STORAGE_BUCKET=$(terraform output -raw storage_bucket)

# Step 2: Configure kubectl
echo -e "${BLUE}Step 2: Configuring kubectl...${NC}"
gcloud container clusters get-credentials $CLUSTER_NAME --zone us-central1-a --project flex-rl

# Step 3: Apply GPU drivers
echo -e "${BLUE}Step 3: Installing NVIDIA device plugin...${NC}"
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/master/nvidia-device-plugin.yml

# Step 4: Build and push Docker image
echo -e "${BLUE}Step 4: Building and pushing Docker image...${NC}"
cd $(dirname $0)/../../
IMAGE_NAME="gcr.io/flex-rl/rl-training:latest"
docker build -t $IMAGE_NAME -f rl/Dockerfile .
docker push $IMAGE_NAME

# Step 5: Deploy MLflow
echo -e "${BLUE}Step 5: Deploying MLflow...${NC}"
kubectl apply -f $(dirname $0)/../kubernetes/mlflow.yaml

# Step 6: Deploy RL training
echo -e "${BLUE}Step 6: Deploying RL training Helm chart...${NC}"
cd $(dirname $0)/../helm/rl-training
helm upgrade --install rl-training . \
  --set storage.bucketName=$STORAGE_BUCKET \
  --set image.repository=gcr.io/flex-rl/rl-training \
  --set image.tag=latest

echo -e "${GREEN}Deployment complete!${NC}"
echo -e "${GREEN}To monitor training progress, run:${NC}"
echo -e "kubectl port-forward svc/mlflow-service 5000:5000"
echo -e "Then visit http://localhost:5000 in your browser."
chmod +x $(dirname $0)/deploy.sh
