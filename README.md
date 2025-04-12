# flex-rl

**flex-rl** is a full-stack research and engineering environment for training adaptable embodied agents using reinforcement learning and large language models (LLMs). It combines fast, modular experimentation workflows with a growing infrastructure layer designed for scalable cloud deployment and multi-GPU inference.

## 🔧 Built for Research

<div style="display: flex; flex-wrap: wrap; justify-content: space-between; gap: 8px;">
	<img src="assets/run_forward.gif" width="23%" />
	<img src="assets/run_left.gif" width="23%" />
	<img src="assets/run_right.gif" width="23%" />
	<img src="assets/stand_still.gif" width="23%" />
</div>
<p align="center"><em>Humanoid agent policies trained with Recurrent PPO</em></p>

Launching a training or inference session from a default template takes just a command:

```bash
# train
python -m rl.train env=humanoid_walk_forward experiment_name=walk_forward

# render eval
python -m rl.play model_dir=workspace/walk_forward/best

# prompt a pre-trained local llm 
python -m llm.infer prompt="How are you?"
```

Custom configurations are easily launched with CLI overrides or config variants:

```bash
# provide custom env + hyperparameters
python -m rl.train env=custom_humanoid_env training.n_envs=2 training.learning_rate=0.0003

# save to gif
python -m rl.play model_dir=workspace/default/best gif_path=custom_eval.gif

# load specific model + parameter overrides
python -m llm.infer --config-name=qwen-1.5 torch_dtype=float16
```

Many features are already implemented and maturing:

- [x] **Train MuJoCo humanoids** in directional locomotion with on-policy Recurrent PPO
- [x] **Reward shaping framework** with dynamic weights and environment wrappers
- [x] **Hydra-powered configs** for rapid experimentation and inference
- [x] **Evaluation renders** with the MuJoCo viewer, or directly to mp4/gif
- [x] **Growing MLFlow integration** to track experiments and sweeps
- [x] **Containerized pipelines** for local/remote flexibility
- [x] **FastAPI server** for LLM prompt inference
- [x] **Infrastructure-as-Code** for cloud deployment with Terraform, Helm, and Kubernetes

## 🧠 Vision: Embodied Agents Guided by LLMs

Long-term goal: bridge the gap between low-level RL control and high-level reasoning using LLMs. Agents will respond to natural language commands, switch behaviors mid-rollout, and reason about their goals with real-time LLM modulation.

```bash
[ Environment State ]
          ↓ 
[ Encoded Prompt → LLM ]
          ↓ 
[ LLM Output: goal-conditioned skill ] 
          ↓ 
[ Agent Policy Execution ]
```

LLM integration is an **active work-in-progress** as operations are expanded to the cloud to support large open weight models, like LLaMA 4 Scout

## 🛠️ Tech Stack

| Domain              | Tools                                                                 |
|---------------------|------------------------------------------------------------------------|
| Physics Sim         | MuJoCo + Gymnasium wrappers                                            |
| RL Training         | Stable Baselines3 (Recurrent PPO), custom reward/term/env wrappers     |
| Experiment Tracking | MLFlow, Hydra                                                          |
| Inference           | HuggingFace Transformers, PyTorch, FSDP, FastAPI                       |
| Containerization    | Docker                                                                 |
| Cloud Deployment    | Terraform, Helm, Kubernetes, Google Cloud Provider (GCP)               |

**Deployment Modes:**

- **Local** – training, debugging, rollouts
- **Docker** – reproducible container builds
- **Cloud (WIP)** – multi-GPU inference and autoscaling

## 🧪 Try it Out Soon

> ⚠️ This project is evolving quickly — documentation is in progress.

Both the `rl` and `llm` environments have a dedicated `requirements.txt` and `Dockerfile`. You can clone the repo and run local training or inference today.

If you're interested in testing things out, I'm happy to document the process and help minimize friction. Reach out or open an issue if you’d like to explore or contribute.

## 🤝 Why This Exists

I’m an engineer with a deep interest in simulation and research for robotics and AI, diving in headfirst and building the infrastructure I need as I go. This project is my way of learning:

- Architectures and tooling that make research fast, modular, and reproducible
- How to efficiently scale GPU-accelerated ML workflows to the cloud
- Modern strategies for training and controlling embodied agents

If you’re working on similar problems in RL tooling, sim-to-real robotics, policy learning with LLMs, I’d love to collaborate or chat.
