#!/bin/bash
# Complete teardown script to bring costs to zero

# Don't exit immediately on errors - we want to continue cleanup
set +e

export USE_GKE_GCLOUD_AUTH_PLUGIN=True

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function for retrying commands with exponential backoff
function retry_command {
    local max_attempts=5
    local timeout=1
    local attempt=1
    local exitCode=0

    while [[ $attempt -le $max_attempts ]]
    do
        echo -e "${BLUE}Attempt $attempt of $max_attempts: $@${NC}"
        
        "$@"
        exitCode=$?

        if [[ $exitCode == 0 ]]
        then
            echo -e "${GREEN}Command succeeded.${NC}"
            return 0
        fi

        echo -e "${YELLOW}Command failed with exit code $exitCode. Retrying in $timeout seconds...${NC}"
        sleep $timeout
        attempt=$(( attempt + 1 ))
        timeout=$(( timeout * 2 ))
    done

    echo -e "${RED}Command failed after $max_attempts attempts.${NC}"
    return $exitCode
}

echo -e "${RED}===============================================${NC}"
echo -e "${RED}= WARNING: This will delete ALL cloud resources =${NC}"
echo -e "${RED}===============================================${NC}"
echo -e "${YELLOW}All cluster resources, jobs, data, and infrastructure will be permanently deleted.${NC}"
read -r -n 1 -p "Are you sure you want to continue? (y/n) " REPLY
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${GREEN}Teardown canceled.${NC}"
    exit 0
fi

PROJECT_ID="flex-rl"
CLUSTER_NAME="flex-rl-cluster"
ZONE="us-central1-a"
TERRAFORM_DIR=$(dirname $0)/../terraform/gke-cluster

# Step 1: Check if the cluster exists
echo -e "${BLUE}Checking if GKE cluster exists...${NC}"
if gcloud container clusters describe $CLUSTER_NAME --zone $ZONE --project $PROJECT_ID &> /dev/null; then
    echo -e "${BLUE}GKE cluster found. Proceeding with Kubernetes resource deletion...${NC}"
    
    # Configure kubectl with a longer timeout
    echo -e "${BLUE}Fetching cluster endpoint and auth data.${NC}"
    if ! retry_command gcloud container clusters get-credentials $CLUSTER_NAME --zone $ZONE --project $PROJECT_ID; then
        echo -e "${YELLOW}Could not connect to the cluster. Proceeding with infrastructure teardown directly.${NC}"
    else
        # Check if we can actually reach the API server
        echo -e "${BLUE}Testing connection to Kubernetes API server...${NC}"
        if ! retry_command kubectl get nodes --request-timeout=30s; then
            echo -e "${YELLOW}Could not connect to Kubernetes API. Proceeding with infrastructure teardown directly.${NC}"
        else
            # Step 2: Delete Helm releases first (if we can connect to the cluster)
            echo -e "${BLUE}Deleting Helm releases...${NC}"
            if helm list -q --namespace default --request-timeout 60s | grep -q "rl-training"; then
                retry_command helm delete rl-training --timeout 5m
                echo -e "${GREEN}RL training Helm release deleted.${NC}"
            else
                echo -e "${YELLOW}No RL training Helm release found.${NC}"
            fi

            # Step 3: Delete all Kubernetes resources with longer timeouts
            echo -e "${BLUE}Deleting MLflow and other Kubernetes resources...${NC}"
            kubectl delete deployment --all --grace-period=0 --force --timeout=60s || true
            kubectl delete service --all --grace-period=0 --force --timeout=60s || true
            kubectl delete job --all --grace-period=0 --force --timeout=60s || true
            kubectl delete pvc --all --grace-period=0 --force --timeout=60s || true
            kubectl delete pv --all --grace-period=0 --force --timeout=60s || true
            kubectl delete configmap --all --grace-period=0 --force --timeout=60s || true
            echo -e "${GREEN}Kubernetes resources deleted.${NC}"
        fi
    fi
else
    echo -e "${YELLOW}GKE cluster not found. Skipping Kubernetes resource deletion.${NC}"
fi

# Step 4: Run Terraform destroy to remove all infrastructure
echo -e "${BLUE}Disabling deletion protection for the GKE cluster...${NC}"
gcloud container clusters update $CLUSTER_NAME --zone $ZONE --project $PROJECT_ID --no-enable-deletion-protection
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Deletion protection disabled.${NC}"
else
    echo -e "${RED}Failed to disable deletion protection. Please check your permissions.${NC}"
    exit 1
fi

echo -e "${BLUE}Destroying all infrastructure with Terraform...${NC}"
cd $TERRAFORM_DIR
terraform init
terraform destroy -auto-approve
```

# Check if terraform destroy worked
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Terraform destroy failed. Attempting targeted resource deletion...${NC}"
    
    # Try to delete the node pool first, then the cluster
    echo -e "${BLUE}Attempting to delete node pool directly...${NC}"
    gcloud container node-pools delete t4-pool --cluster=$CLUSTER_NAME --zone=$ZONE --quiet || true
    
    echo -e "${BLUE}Attempting to delete cluster directly...${NC}"
    gcloud container clusters delete $CLUSTER_NAME --zone=$ZONE --quiet || true
    
    # Try terraform destroy again
    echo -e "${BLUE}Retrying terraform destroy...${NC}"
    terraform destroy -auto-approve
fi

echo -e "${GREEN}Terraform infrastructure destroyed.${NC}"

# Step 5: Check for any remaining resources (GCS buckets, etc.)
echo -e "${BLUE}Checking for remaining GCS buckets...${NC}"
BUCKETS=$(gsutil ls -p $PROJECT_ID 2>/dev/null | grep "gs://flex-rl" || true)
if [ -n "$BUCKETS" ]; then
    echo -e "${YELLOW}Remaining buckets found. Deleting...${NC}"
    for BUCKET in $BUCKETS; do
        gsutil -m rm -r $BUCKET
        echo -e "${GREEN}Deleted $BUCKET${NC}"
    done
else
    echo -e "${GREEN}No remaining buckets found.${NC}"
fi

# Step 6: Check for remaining GCE resources
echo -e "${BLUE}Checking for any remaining compute instances...${NC}"
INSTANCES=$(gcloud compute instances list --project $PROJECT_ID --format="value(name)" 2>/dev/null | grep "flex-rl" || true)
if [ -n "$INSTANCES" ]; then
    echo -e "${YELLOW}Remaining instances found. Deleting...${NC}"
    for INSTANCE in $INSTANCES; do
        gcloud compute instances delete $INSTANCE --zone $ZONE --project $PROJECT_ID --quiet
        echo -e "${GREEN}Deleted instance $INSTANCE${NC}"
    done
else
    echo -e "${GREEN}No remaining instances found.${NC}"
fi

echo -e "${GREEN}=============================================${NC}"
echo -e "${GREEN}= TEARDOWN COMPLETE - ALL RESOURCES DELETED =${NC}"
echo -e "${GREEN}=============================================${NC}"
echo -e "${YELLOW}Your GCP project should now have no running resources related to flex-rl.${NC}"
echo -e "${YELLOW}To verify, check your GCP console and billing dashboard.${NC}"
chmod +x $(dirname $0)/teardown.sh
