# flex-rl — Scalable RL Training & Orchestration Pipeline

**flex-rl** is a reinforcement learning research pipeline focused on scalable, reproducible infrastructure for goal-conditioned and offline RL training. It is built to support flexible, large-scale experimentation on MuJoCo Gym environments, with an emphasis on robust infrastructure and workflow automation.

The project is an ongoing personal effort to explore RL techniques and build production-grade orchestration systems used in modern RL research environments.

---

## 🎯 Project Purpose

The primary goals of this project are to:

- Deepen understanding of reinforcement learning and ML infrastructure
- Experiment with modern multi-skill RL agent training
- Develop scalable, reproducible infrastructure for large-scale experiments

The system is structured to support future expansion into a more general-purpose RL training platform.

---

## 🔥 Key Features

✅ End-to-end RL training pipeline  
✅ Local development support with Docker  
✅ Kubernetes-native orchestration with support for experiment sweeps  
✅ Persistent experiment tracking & reproducibility  
✅ Seamless local → cloud migration path (Minikube → GKE)  
✅ Flexible design for custom environments, Gym extensions, or hardware integration

---

## 🧩 Research & Experimentation Phases

The current research workflow follows three phases:

### Phase 1 — Multi-Skill Pretraining

Pretraining MuJoCo Humanoid agents on core locomotion skills (e.g., walking, turning) using on-policy PPO.

### Phase 2 — Goal-Conditioned Fine-Tuning & Offline RL

Fine-tuning pretrained agents with explicit goal embeddings and large-scale offline datasets.  
Includes support for Behavior Cloning and Decision Transformer-inspired methods.

### Phase 3 — Instruction-Conditioned Behavior

Incorporating lightweight language models to condition agent behavior on natural language instructions.

---

## 🏗️ Infrastructure Overview

The infrastructure is designed for both local development and cloud-scale orchestration:

| Environment         | Use Case                                   |
|--------------------|--------------------------------------------|
| Local (Docker)    | Development & debugging                     |
| Local (Minikube)  | Testing Kubernetes manifests and pipelines  |
| Cloud (GKE, Terraform) | Scalable training & large-scale experiments *(In progress)* |

The local setup mirrors the production cluster environment to enable seamless transitions between development and large-scale training.

---

## 📄 Requirements

- Python 3.10
- Docker
- NVIDIA GPU + drivers (for local GPU training)
- Minikube & kubectl (for local cluster workflow)

---

## ⚙️Setup

Clone the repository:

```bash
git clone https://github.com/tylimbach/flex-rl.git
cd flex-rl
```

Create and activate Python environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(Optional) Install Minikube:

Refer to: https://minikube.sigs.k8s.io/docs/start/

---

## 🚀 Local Development Workflow

For small-scale, local training runs:

Example:

```bash
python train.py --name test_run --config examples/humanoid_walk_forward.yaml
```

Training artifacts will be saved under `./workspace/`.

---

## ☁️ Kubernetes Workflow (Minikube)

> ```Makefile``` provides quick-start commands but you'll likely want to customize arguments or config yaml for personal resource allocation.

### Start Local Minikube Cluster

```bash
make minikube-up
make pvc-up
```

### 2. Build and Load Runtime Image

```bash
make build-runtime
minikube cache add flex-rl-runtime:latest
```

### Submit Training Sweep

```bash
python orchestrator/scripts/submit_sweep.py --sweep examples/humanoid_sweep.yaml
```

This will dynamically submit a Kubernetes Job based on the sweep configuration.

### Monitor Progress

```bash
kubectl get jobs
kubectl logs job/<job-name>
```

Optional: View TensorBoard

```bash
make tensorboard-up
```

### Cleanup

```bash
make jobs-clean
make pvc-down
make minikube-down
make tensorboard-down
```

---

## 🚀 Cloud Workflow (Coming Soon)

The infrastructure is designed to scale to Google Kubernetes Engine (GKE) using Terraform for reproducible infrastructure deployment.

Future steps will include:

- GKE cluster setup via Terraform
- Cloud-based GPU scaling & automated experiment orchestration
- Experiment artifact storage in GCS

## 🌱 Future Roadmap

- 🔨 Support for GKE + Terraform-managed infrastructure
- 🔨 Cloud-based GPU scaling & automated experiment orchestration
- 🚧 MLflow / W&B integration for experiment tracking GUI
- 🚧 Dynamic sweep configuration templates
- 🚧 Optional Airflow/Kubeflow pipeline orchestration
- 🚧 Offline RL dataset packaging

---

## 💡 Project Intent

This repository is primarily a personal research project and infrastructure showcase. It is not currently designed as a plug-and-play RL product but is built to demonstrate scalable, production-grade ML infrastructure practices.

Future development may expand the system into a more general-purpose, modular RL training platform if there is community interest or demand.
