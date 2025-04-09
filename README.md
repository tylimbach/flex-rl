# flex-rl — Adaptable Agents with RL and LLMs

**flex-rl** explores the future of adaptable embodied agents by combining reinforcement learning (RL) with large language models (LLMs). The project’s long-term goal is to enable dynamic skill switching using LLM-guided instructions and real-time feedback, allowing agents to respond fluidly to their environment and external commands.

The system already supports scalable training and inference pipelines for on-policy Recurrent PPO with humanoid agents, LLM inference with PyTorch, and end-to-end infrastructure built for local and cloud-scale experimentation.

<p align="center">
  <img src="assets/run_forward.gif" width="500" alt="Trained humanoid agent sprinting">
</p>
<p align="center"><em>Humanoid agent trained via Recurrent PPO demonstrating sprint behavior</em></p>

## Vision & Roadmap

```bash
[ Environment ]
        ↓
[ Observation Encoding: "Agent: (15, 30), Obstacle: (18, 28), Target: (20, 25)..." ]
        ↓
[ LLM Prompt: "Choose the best action ..." ] ←────────┐
        ↓                                             │
[ LLM Output: "move_right" ]                          │
        ↓                                             │
[ Agent Policy Execution ]                            │
                                                      │
[ External Instruction: "Go around the obstacle!" ] ──┘
```

> **Goal:** Leverage pre-trained, possibly fine-tuned LLMs to select from a library of motor policies in response to environment state and natural language instructions.

The LLM integration is self-managed and an **active work-in progress**, including support for:

- Sharded inference across GPUs using PyTorch FSDP
- Prompt encoding from simulation states
- Real-time instruction injection
- Goal-conditioned inference steering


## Key features (Implemented)

- Train MuJoCo humanoid agents using Recurrent PPO
- Flexible hyperparameter sweeps (CLI overrides + YAML config)
- Local experiment tracking with growing MLFlow integration
- Containerized RL + inference workflows
- Multi-GPU LLM inference using HuggingFace Transformers
- Infrastructure-as-code stack (Terraform, Docker, Helm, Kubernetes)
- Deployment modes for local testing, minikube, and Google Cloud Platform (GCP)

---

## Powered By

- [MuJoCo](https://mujoco.org/): Physics simulation engine with [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) wrappers
- [MLFlow](https://github.com/facebookresearch/hydra): Platform for experiment tracking and GUIs
- [Transformers](https://github.com/huggingface/transformers): Pretrained LLMs and inference pipelines
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3): PPO RL algorithms
- [Docker](https://www.docker.com/): Runtime containerization
- [Helm](https://helm.sh/): [Kubernetes](https://kubernetes.io/) application management
- [Terraform](https://www.terraform.io/): Infrastructure as code for cloud deployment
- [Hydra](https://github.com/facebookresearch/hydra): Hierarchical configuration management on top of [OmegaConf](https://github.com/omry/omegaconf)

---

## Infrastructure Modes

| Mode               | Purpose                                 |
|--------------------|------------------------------------------|
| Local              | Training, debugging, rollouts            |
| Docker             | Reproducible container builds            |
| Minikube           | Local Kubernetes job testing             |
| Cloud                | Multi-GPU inference and autoscaling  |

---

## Project Scope

This is an independent, research-grade prototype. While not intended as a framework, it reflects my efforts to follow real-world infrastructure design for RL pipelines, agent simulation, and distributed inference.

Researchers and engineers are welcome to explore the implementation, adapt patterns, or extend components. Contributions and feedback are encouraged, I'd love to talk to anyone pursuing this area of work.
