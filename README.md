# flex-rl — Scalable RL Training & Orchestration Infrastructure

**flex-rl** is a reinforcement learning infrastructure project focused on scalable experimentation, reproducibility, and modular design. It supports goal-conditioned and offline RL pipelines on MuJoCo Gym environments and showcases best practices for production-grade training orchestration.

This repository demonstrates infrastructure and experimental design workflows commonly found in modern RL research labs and industry environments. The system is designed to support cloud-based scale while remaining lightweight for local development.

---

## ⚙️ Technologies Used

- [Hydra](https://github.com/facebookresearch/hydra): Configuration management and multi-run orchestration
- [MuJoCo](https://mujoco.org/): Physics simulation engine via Gymnasium
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3): PPO + RL algorithms
- [Kubernetes](https://kubernetes.io/): Job scheduling and orchestration
- [Docker](https://www.docker.com/): Runtime containerization
- [Terraform](https://www.terraform.io/): Infrastructure as code for cloud deployment
- [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine): Cloud-native scaling backend
- [MoviePy](https://zulko.github.io/moviepy/): Media rendering for rollout visualization

---

## 🧱 System Highlights

- 📦 End-to-end RL training pipeline (on-policy PPO, goal-conditioned)
- 🖥️ Local development with Docker or bare-metal GPU
- ☁️ Kubernetes-native orchestration (Minikube or GKE)
- 📁 Persistent snapshotting and experiment metadata
- 🔄 Hydra-driven sweeps and experiment templating
- 🧩 Modular design: custom environments, Gym wrappers, goal samplers

---

## 🧪 Current Experimental Phases

**Phase 1 — Multi-Skill Pretraining**  
Train MuJoCo Humanoid agents on basic locomotion (walk, turn, stand).

**Phase 2 — Goal-Conditioned Fine-Tuning**  
Extend base agents to condition on discrete or continuous goal states. 

**Phase 3 — Offline + Instruction-Conditioned RL**  
Integrate offline datasets and instruction embeddings (in progress).

---

## 🧰 Infrastructure Modes

| Mode             | Description                                     |
|------------------|-------------------------------------------------|
| Local (Bare/Docker) | Fast iteration and debugging                  |
| Minikube (Local K8s) | Kubernetes workflow testing                   |
| GKE (Cloud)      | Cluster-scale orchestration (planned)          |

---

## 📦 Requirements

- Python 3.10+
- NVIDIA GPU (for training)
- Docker (for containerized workflows)
- Minikube + kubectl (for local Kubernetes jobs)

---

### 🔧 Hydra-Based Configuration

All experiments are configured using [Hydra](https://github.com/facebookresearch/hydra). The system supports:
- Modular, hierarchical YAML configs (`env/`, `training/`, `run/`)
- Command-line overrides (e.g. `python train.py training.learning_rate=0.0001`)
- Sweep support and dynamic config interpolation

See the `configs/` folder for examples.

> I am working on migrating the old K8s sweeps to Hydra configs

## 🚀 Quick Start: Local Training

```bash
git clone https://github.com/tylimbach/flex-rl.git
cd flex-rl
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run a training job:
```bash
python train.py experiment_name=test-run
```

Artifacts are saved in `workspace/` with metadata, checkpoints, and config.

---

## ☁️ Kubernetes Workflow (via Minikube)

> `Makefile` provides shortcuts for common cluster operations

### Start Cluster
```bash
make minikube-up
make pvc-up
```

### Build Runtime Image
```bash
make build-runtime
minikube cache add flex-rl-runtime:latest
```

### Submit a Sweep
```bash
python orchestrator/scripts/submit_sweep.py --sweep examples/humanoid_sweep.yaml
```

### Monitor
```bash
kubectl get jobs
kubectl logs job/<job-name>
make tensorboard-up
```

### Cleanup
```bash
make jobs-clean
make pvc-down
make minikube-down
```

---

## 📌 Roadmap

- [ ] GKE deployment via Terraform
- [ ] MLflow or Weights & Biases integration
- [ ] Offline RL dataset tooling
- [ ] Instruction-conditioned rollouts
- [ ] Multi-agent support (stretch goal)

---

## 📚 Project Status

This repository is a research and infrastructure showcase — built to reflect the architecture and workflows of scalable ML systems. It is not a plug-and-play RL library, but a starting point for custom pipelines, research prototypes, and educational study.

Feedback and collaboration welcome.
