# Building Scalable, Generalized RL Agents

This project explores modern reinforcement learning architectures with the goal of building a scalable, multi-skill, and goal-conditioned embodied agent. It’s an engineering and research exercise that walks through the real tradeoffs behind RL pipelines, moving beyond single-skill policies toward a system that can learn and generalize efficiently.

Starting from classic on-policy RL methods and incorporating emerging offline RL and Transformer-based architectures used by research labs like DeepMind and Meta.

## Project Overview

### MuJoCo PPO Skill Training

The first phase of the project focuses on classic PPO training in MuJoCo environments. I use Gymnasium environments like HalfCheetah and Humanoid to train separate policies for individual skills such as running and recovery.

The goal here was to revisit the foundations of on-policy RL and understand where these methods start to break down when trying to build more general agents.

### Offline RL and Transformer Pipeline (In Progress)

The next phase shifts toward an offline RL architecture:

- Collect millions of trajectory rollouts across various skills
- Train a Transformer-based policy offline, supervised on these trajectories
- Condition behavior on goals or language instructions
- Eliminate the need for brittle policy switching

Offline RL enables GPU-accelerated training and may offer a more flexible unified policy compared to strict policy swapping.

This approach is inspired by modern research systems like Decision Transformer and Gato. The project aims to reproduce a scaled-down but technically sound version of those pipelines. Starting out I'll use one machine, but I plan to expand to a cluster. 

### LLM Integration

A secondary goal is to explore LLM integration:

- Natural language → task translation
- LLM-driven feedback or evaluation loops

A small fine-tuning script for GPT-2 Medium is included to learn the process of building and training LLMs for this purpose.

## Setup and Environment

This project is evolving rapidly, but I will do my best to maintain setup instructions. 
Docker environments are provided for portability and reproducibility.

The environment assumes an NVIDIA GPU (tested on RTX 3080 Ti) and Linux.  
I’ve tested this on Ubuntu 24.04 and Docker images based on 22.04.

Quick start:

```bash
# Build Docker image
docker build -t rl-env -f rl/docker/Dockerfile .

# Run container with GPU
docker run -it --gpus all -v $(pwd):/workspace -w /workspace rl-env
```

Alternatively, you can set up a Python venv using the provided `requirements.txt`.

## What's Next

The project is structured in phases:

- Finish trajectory collection for multiple skills
- Train a unified, goal-conditioned Transformer policy
- Fine-tune and extend the policy as new skills are added
- Explore LLM-driven task instruction and evaluation
- Write a full technical blog post documenting the tradeoffs and design decisions

## Why This Project

This project is part learning exercise, part infrastructure experiment, and part portfolio piece. It’s meant to build practical understanding of how modern RL systems scale—and where the real bottlenecks and challenges are.

If you’re interested in RL systems engineering, distributed compute, or embodied AI, this repo may have something useful for you (especially as it grows!).

