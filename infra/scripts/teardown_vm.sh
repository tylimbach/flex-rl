#!/bin/bash
# Teardown script for CPU-only training setup

set +e

export USE_GKE_GCLOUD_AUTH_PLUGIN=True

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting teardown for CPU-only training setup...${NC}"

# Step 1: Destroy CPU-only VM using Terraform
echo -e "${BLUE}Destroying CPU-only VM with Terraform...${NC}"
cd infra/terraform/cpu-only-vm
terraform destroy -auto-approve

echo -e "${GREEN}Teardown complete.${NC}"
