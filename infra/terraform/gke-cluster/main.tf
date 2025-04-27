provider "google" {
  project = "flex-rl"
  region  = "us-central1"
  zone    = "us-central1-a"
}

# Enable required APIs for the project
resource "google_project_service" "container" {
  service = "container.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "compute" {
  service = "compute.googleapis.com"
  disable_on_destroy = false
}

# We'll use GCS instead of Filestore which has fewer dependencies
resource "google_project_service" "storage" {
  service = "storage.googleapis.com"
  disable_on_destroy = false
}

resource "google_container_cluster" "rl_cluster" {
  name     = "flex-rl-cluster"
  location = "us-central1-a"
  
  # Use a minimal default node pool for control plane
  initial_node_count = 1
  node_config {
    machine_type = "e2-medium"
  }
  
  # Enable workload identity for better security
  workload_identity_config {
    workload_pool = "flex-rl.svc.id.goog"
  }

  # Enable GKE Dataplane V2 for better networking
  networking_mode = "VPC_NATIVE"
  network    = "default"
  subnetwork = "default"
  
  # Wait for API to be enabled
  depends_on = [
    google_project_service.container,
    google_project_service.compute
  ]
}

# Start with a single GPU node to avoid quota issues
resource "google_container_node_pool" "gpu_pool" {
  name       = "t4-pool"
  cluster    = google_container_cluster.rl_cluster.name
  location   = "us-central1-a"
  node_count = 1  # Start with one GPU node, can scale to 2 later

  # Use auto-repair and auto-upgrade for stability
  management {
    auto_repair  = true
    auto_upgrade = true
  }

  node_config {
    machine_type = "n1-standard-8"
    
    # GPU configuration
    guest_accelerator {
      type  = "nvidia-tesla-t4"
      count = 1
    }
    
    # Make this a preemptible VM to save costs
    preemptible  = true
    
    # Required for GPU support
    metadata = {
      "install-nvidia-driver" = "true"
    }
    
    # Labels for workload assignment
    labels = {
      "accelerator" = "t4"
      "workload"    = "rl-training"
    }
    
    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }
  
  # This helps with dependency management
  depends_on = [
    google_container_cluster.rl_cluster
  ]
}

# Use Google Cloud Storage instead of Filestore
resource "google_storage_bucket" "rl_data_bucket" {
  name          = "flex-rl-training-data"
  location      = "US"
  force_destroy = true  # Allow terraform destroy to remove bucket
  
  # Lifecycle settings to automatically manage data
  lifecycle_rule {
    condition {
      age = 30  # Delete files older than 30 days
    }
    action {
      type = "Delete"
    }
  }
  
  depends_on = [
    google_project_service.storage
  ]
}

output "cluster_name" {
  value = google_container_cluster.rl_cluster.name
}

output "storage_bucket" {
  value = google_storage_bucket.rl_data_bucket.name
}
