# flex-rl — Scalable RL Training & Orchestration Pipeline

flex-rl aims to be a research-grade reinforcement learning pipeline focused on
**scalable, reproducible training and offline learning workflows**.
While the underlying experiments explore goal-conditioned motor agents,
the primary emphasis is on building a robust, maintainable infrastructure to
support large-scale RL research.

---

## Project Overview

### Pipeline & Infrastructure Focus

The flex-rl system is built for:

- **Reproducible RL training & evaluation workflows**
- **Orchestration of multi-experiment sweeps via Kubernetes**
- **Persistent experiment lineage & snapshot tracking**
- **Seamless local dev + production-ready cluster deployment**

The project provides a clean separation between local development environments
and scalable runtime images, following standard infrastructure practices.

---

### Research Context

The research goal is to pretrain and fine-tune multi-skill motor agents using
scalable infrastructure:

**Phase 1: Multi-Skill Pretraining**  
Train a MuJoCo Humanoid agent to perform multiple locomotion tasks (walking,
turning) using reward shaping. The Pipeline makes spawning of single-task and
multi-task training sessions easy.

**Phase 2: Goal-Conditioned Fine-Tuning & Offline Learning**  
Extend policies with explicit goal embeddings, collect large-scale trajectory
data, and fine-tune using behavior cloning or Decision Transformer-style architectures.

**Phase 3: LLM Integration (Planned)**  
Enable instruction-conditioned behavior via lightweight LLM models for goal interpretation and feedback.

---

## Infrastructure Highlights

✅ **Full Kubernetes Orchestration**  
Training, evaluation, and data collection jobs submitted via CLI and sweep YAML configs.

✅ **Local & Cloud-Ready Workflow**  
The project includes both a live-mounted development container and a minimal runtime image for production Kubernetes execution.

✅ **Makefile Automation**  
All common workflows (build, cluster up/down, sweep submission) are automated via Makefile.

✅ **Persistent Experiment Tracking**  
Experiments are saved under `workspace/`, supporting reproducibility and analysis.

---

## Setup and Environment

Tested on Ubuntu 22.04 and 24.04. I have not yet tested Windows or MacOS.

**Requirements:**

- NVIDIA GPU (tested on RTX 3080 Ti)

> To orchestrate sweeps with Kubernetes you need a bit extra

- Docker
- [Kind](https://kind.sigs.k8s.io)
- kubectl

### Quick Start

This project includes a Docker setup for reproducibility and GPU acceleration.

Build and run the dev environment:

```bash
make build-dev
make cluster-up
make shell
```

Start training:

```bash
# train one example experiment
python train.py --name test_run --config examples/humanoid_walk_forward.yaml
# start a training sweep containing multiple experiments
python orchestrator/scripts/submit_sweep.py --sweep examples/humanoid_sweep.yaml
```

**All experiment artifacts will be stored in `./workspace/`.**

---
