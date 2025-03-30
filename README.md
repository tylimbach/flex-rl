# Building Scalable, Generalized Motor Agents

This project explores modern reinforcement learning architectures for building
multi-skill, goal-conditioned embodied agents.
It’s an engineering and research exercise inspired by systems like **Gemini
Robotics**, **Gato**, and **Decision Transformer**, focusing on practical
tradeoffs in RL pipelines.

---

## Project Overview

### Phase 1: Multi-Task Skill Pretraining

The first phase focuses on classic on-policy RL with MuJoCo environments.
Using **Humanoid-v5**, we pretrain a single policy to perform multiple
locomotion skills (e.g. walking, turning) via **reward shaping** and randomized goals.

Although the policy has no explicit awareness of the goal at this stage, this
phase builds a competent low-level controller—a common strategy in modern RL pipelines.

> This mirrors the motor pretraining stage in systems like Gemini Robotics.

The policy will later be fine-tuned with explicit goal inputs and used as a building
block for instruction-conditioned agents.

---

### Phase 2: Goal-Conditioned RL + Offline Learning (In Progress)

After multi-skill competency is achieved:

- Extend the observation space with **goal embeddings**
- Train a goal-conditioned policy that can follow language or symbolic goals
- Shift to **offline RL** for efficient large-scale training:
  - Collect trajectories from multi-skill policy
  - Train using a Decision Transformer-style model
  - Enable skill switching via in-context goal prompts

This architecture enables a **single, unified policy** to generalize over a
variety of tasks without brittle policy switching.

---

### Phase 3: LLM Integration (Planned)

We will explore large language models for high-level reasoning:

- **Instruction parsing** → convert language tasks into RL-executable goals
- **LLM-guided feedback** → assess, correct, or propose trajectories
- Lightweight LLM fine-tuning using models like TinyLlama or GPT-2 Medium

---

## Setup and Environment

Tested on Ubuntu 22.04 and 24.04. I have not yet tested Windows or MacOS.

**Requirements:**

- NVIDIA GPU (tested on RTX 3080 Ti)
- Docker or Python 3.10 venv

### Quick Start

This project includes a Docker setup for reproducibility and GPU acceleration.

```bash
docker build -t rl-env -f docker/Dockerfile .
docker run -it --gpus all -v $(pwd):/workspace -w /workspace rl-env

# train new agent with default config
python3 train.py --name humanoid
# after training, render evaluation of the best model snapshot
python3 play.py --model workspace/humanoid/best
```
