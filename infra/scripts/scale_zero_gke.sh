#!/bin/bash
# Script to easily scale GPU nodes up or down to save costs

CLUSTER_NAME="flex-rl-cluster"
NODE_POOL="t4-pool"
ZONE="us-central1-a"

function show_usage() {
    echo "Usage: $0 [up|down]"
    echo "  up    - Scale GPU nodes to 1 or 2 (for distributed training)"
    echo "  down  - Scale GPU nodes to 0 (to save costs)"
    exit 1
}

if [ "$#" -ne 1 ]; then
    show_usage
fi

case "$1" in
    up)
        echo "Scaling GPU nodes up to 1..."
        gcloud container clusters update $CLUSTER_NAME \
            --node-pool=$NODE_POOL \
            --num-nodes=1 \
            --zone=$ZONE
        
        echo "To scale to 2 nodes for distributed training, run:"
        echo "gcloud container clusters update $CLUSTER_NAME --node-pool=$NODE_POOL --num-nodes=2 --zone=$ZONE"
        ;;
    
    down)
        echo "Scaling GPU nodes down to 0 to save costs..."
        gcloud container clusters update $CLUSTER_NAME \
            --node-pool=$NODE_POOL \
            --num-nodes=0 \
            --zone=$ZONE
        
        echo "Your cluster control plane is still running. To completely delete the cluster and stop all charges:"
        echo "cd infra/terraform/gke-cluster && terraform destroy"
        ;;
    
    *)
        show_usage
        ;;
esac

echo "Operation completed successfully!"
