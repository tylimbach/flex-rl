# flex-rl — Multi-GPU Inference for Adaptive Embodied Agents

flex-rl's goal is to build adaptable agents from a small set of trained motor policies (RL) by incorporating a large language model (LLM) to steer skill selection in response to real-time environment feedback and external instructions. The system supports scalable experimentation across both local and cloud infrastructure: from bare-metal on-policy PPO training to multi-GPU LLM inference using tensor parallelism on Kubernetes.

```bash
[ Environment ]
        ↓
[ Observation Encoding: "Agent: (15, 30), Ball: (20, 25)..." ]
        ↓
[ LLM Prompt: "Choose the best action ..." ] ←────────┐
        ↓                                             │
[ LLM Output: "move_right" ]                          │
        ↓                                             │
[ Agent Policy Execution ]                            │
                                                      │
[ External Instruction: "Avoid the ball!" ] ──────────┘
```

> Instruction injection and prompting provide context for intelligent selection of the next motor primitive.

## Key features

- Train goal-wrapped agent policies using MuJoCo and Recurrent PPO
- Orchestrate flexible sweeps and experiments via CLI or YAML
- Serve LLM inference via FastAPI or job submission pipelines
- Scale from local dev to K8s-based multi-GPU inference with tensor parallelism
- Infrastructure as Code with Terraform and Helm

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
| Docker             | Containerized reproducibility            |
| Minikube           | Local Kubernetes job testing             |
| Cloud                | Cloud inference + multi-GPU deployment  |

---

## Project Scope

This is an independent research prototype. While not designed for broad reuse, it reflects real-world ML infrastructure with RL-focused design choices, end-to-end orchestration, and custom LLM-agent integration experiments.

Engineers and researchers are encouraged to explore the RL and infra patterns within. Contributions welcome.
