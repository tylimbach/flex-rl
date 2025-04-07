provider "google" {
	project = "flex-rl"
	region  = "us-central1"
	zone    = "us-central1-b"
}

resource "google_compute_instance" "llm_dev" {
	name         = "llm-dev"
	machine_type = "n1-standard-8"
	zone         = "us-central1-b"

	boot_disk {
		initialize_params {
			image = "projects/deeplearning-platform-release/global/images/family/pytorch-latest-gpu"
			size  = 100
		}
	}

	network_interface {
		network = "default"
		access_config {}
	}

	guest_accelerator {
		type  = "nvidia-tesla-t4"
		count = 1
	}

	scheduling {
		preemptible       = true
		on_host_maintenance = "TERMINATE"
		automatic_restart = false
	}

	metadata = {
		install-nvidia-driver = "true"
	}

	service_account {
		email  = "default"
		scopes = ["https://www.googleapis.com/auth/cloud-platform"]
	}
}
