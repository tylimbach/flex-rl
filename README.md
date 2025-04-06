# flex-rl — RL & LLM Agent Pipelines for Scalable Experimentation

**flex-rl** is a personal research project and infrastructure showcase for reinforcement learning and language model integration. It enables scalable RL experimentation via containerized training pipelines and supports local or Kubernetes-based multi-GPU inference of open-weight LLMs.

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

> The core architecture explores using natural language and LLMs as a central decision maker in a multimodal agent loop. By injecting instructions mid-run, agents can adapt behavior without retraining. Traditional RL is leveraged to train a small set of learned motor skills, which the language model composes to translate high-level tasks into actionable motions, resulting in the emergence of complex behaviors.

The system is built with modularity and reproducibility in mind, designed to support:
- Flexible on-policy agent training (PPO)
- Goal-conditioned or instruction-guided environments
- Hydra-driven experiment sweeps
- LLM-driven behavior adaptation
- Cloud-native inference infrastructure (GKE, torchrun)

---

## 🔧 Technologies Used

- **RL & Simulation**: MuJoCo, Gymnasium, Stable Baselines3
- **Infra**: Docker, Kubernetes, torchrun, Terraform, GKE
- **LLMs**: HuggingFace Transformers, Qwen, Mistral, LLaMA 4
- **Config**: Hydra, Omegaconf

---

## 🧱 Architecture Overview

- `rl/`: PPO agents, custom envs, Gym wrappers
- `llm/`: Inference pipelines and configs
- `docker/`: Runtime images for training & inference
- `k8s/`: Kubernetes job templates + Make targets
- `workspace/`: Training logs, rollouts, and metadata

---

## 🧪 Experimental Phases

- **Skill Pretraining** — locomotion PPO training in MuJoCo
- **Goal-Conditioned Fine-Tuning** — via Gym wrappers + config
- **Instruction-Driven Control** — LLM-generated goal inputs (in progress)
- **Offline RL + Datasets** — embedding conditioning (planned)

---

## ☁️ Infrastructure Modes

| Mode               | Purpose                                 |
|--------------------|------------------------------------------|
| Local              | Training, debugging, rollouts            |
| Docker             | Containerized reproducibility            |
| Minikube           | Local Kubernetes job testing             |
| GKE                | Cloud inference + multi-GPU deployment  |

---

## 📚 Project Scope

This is an independent research prototype. While not designed for broad reuse, it reflects real-world ML infrastructure with RL-focused design choices, end-to-end orchestration, and custom LLM-agent integration experiments.

Engineers and researchers are encouraged to explore the RL and infra patterns within. Contributions welcome.
