# flex-rl — Scalable RL Training & Orchestration Pipeline

**flex-rl** is a research-grade reinforcement learning pipeline focused on scalable, reproducible infrastructure for goal-conditioned and offline RL training. While the experiments emphasize motor skill learning, the primary focus is building maintainable infrastructure for large-scale RL workflows.

The architecture and techniques are inspired by systems like **Gemini
Robotics**, **Gato**, and **Decision Transformer**.

---

## Project Overview

### Infrastructure & Research Goal

The project is designed around two priorities:

- Building scalable RL infrastructure.
- Enabling efficient, multi-skill, goal-conditioned agent training.

The infrastructure supports:

✅ Reproducible experiment workflows
✅ Full Kubernetes orchestration with sweep config support
✅ Persistent experiment tracking and lineage
✅ Seamless local development + production-ready runtime image

---

### Research Context

To start, the underlying research workflow focuses on:

#### Phase 1: Multi-Skill Pretraining

Training MuJoCo Humanoid agents on multiple locomotion skills (walking, turning) using on-policy PPO.

#### Phase 2: Goal-Conditioned Fine-Tuning & Offline Learning

Fine-tuning agents with explicit goal embeddings and large-scale offline data (e.g. Behavior Cloning, Decision Transformer).

#### Phase 3: LLM Integration

Instruction-conditioned behavior using lightweight language models.

---

## Environment Setup

Tested on Ubuntu 22.04 and 24.04.

Not tested on Windows or MacOS.

**Requirements:**

For local development:

- NVIDIA GPU (tested on RTX 3080 Ti)
- Python 3.10
- Docker

For local orchestration:

- Minikube
- kubectl

## Quick Start

All training artifacts are saved under ./workspace/ by default.

### 1. Local Dev Environment

```bash
make build-dev
make shell
```

From inside the dev container:

```bash
python train.py --name test_run --config examples/humanoid_walk_forward.yaml
```

### 2. Kubernetes Sweep Workflow (Local)

> This workflow is in flux. More detailed customization instructions will be added as development progresses.

One-time: Start local registry

```bash
make build-runtime
docker run -d -p 5000:5000 --restart=always --name registry registry:2
```

Start Minikube cluster

```bash
make minikube-up
make pvc-up
```

Build and push runtime image

```bash
make sweep-dev
```

This will:

- Build runtime container
- Push to localhost:5000
- Submit a default configured sweep job

View TensorBoard

```bash
make tensorboard-up
```

Clean up

```bash
make jobs-clean
make pvc-down
make minikube-down
make tensorboard-down
```

---
