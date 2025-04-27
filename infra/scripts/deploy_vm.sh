#!/bin/bash
# Deploy script for CPU-only training setup

set -e

export USE_GKE_GCLOUD_AUTH_PLUGIN=True

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting deployment for CPU-only training setup...${NC}"

# Step 1: Provision CPU-only VM using Terraform
echo -e "${BLUE}Provisioning CPU-only VM with Terraform...${NC}"
cd infra/terraform/cpu-only-vm
terraform init
terraform apply -auto-approve

# Step 2: Get the VM's external IP
VM_IP=$(gcloud compute instances list --filter="name=cpu-only-vm" --format="get(networkInterfaces[0].accessConfigs[0].natIP)")

if [ -z "$VM_IP" ]; then
  echo -e "${RED}Failed to retrieve the VM's external IP.${NC}"
  exit 1
fi

echo -e "${GREEN}VM is running at IP: $VM_IP${NC}"

# Step 3: SSH into the VM and run the training script
echo -e "${BLUE}Running training script on the VM...${NC}"
gcloud compute ssh --zone "us-central1-b" "cpu-only-vm" --command "
  docker run --rm -v /app/data:/app/data gcr.io/flex-rl/rl-training:latest \
  python -m rl.train --config-name=distributed training.device=cpu
"

echo -e "${GREEN}Deployment complete.${NC}"
