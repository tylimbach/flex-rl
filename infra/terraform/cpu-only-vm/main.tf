provider "google" {
  project = "flex-rl"
  region  = "us-central1"
  zone    = "us-central1-b"
}

resource "google_compute_instance" "cpu_only_vm" {
  name         = "cpu-only-vm"
  machine_type = "n1-standard-4"
  zone         = "us-central1-b"

  boot_disk {
    initialize_params {
      image = "projects/ubuntu-os-cloud/global/images/family/ubuntu-2004-lts"
      size  = 50
    }
  }

  network_interface {
    network = "default"
    access_config {}
  }

  scheduling {
    preemptible       = true
    on_host_maintenance = "TERMINATE"
    automatic_restart = false
  }

  metadata = {
    training-device = "cpu"
    startup-script = <<-EOT
      #!/bin/bash
      echo "Starting VM setup..."
      apt-get update
      apt-get install -y docker.io
      systemctl start docker
      usermod -aG docker $USER
      newgrp docker <<EONG
      docker pull gcr.io/flex-rl/rl-training:latest
      echo "VM setup complete."
    EOT
  }

  service_account {
    email  = "default"
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
  }
}
      
