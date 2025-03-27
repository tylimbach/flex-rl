This project aims to integrate LLMs into real-time agent control loops.

# Components
- MuJoCo physics simulation for agents
- PPO reinforcement learning
- LLM fine-tuning (PyTorch)

Very exploratory in nature, so rapid change is expected. I have some ML background from college, but I'm
learning many of these tools and advancements for the first time. My goals in this project include optimizing
training pipelines and scaling them up.

Training environments are containerized with Docker for portability and eventual distributed training.
Testing was done on a local Ubuntu 24.04 with 3080 Ti GPU, and as well as the Ubuntu 22.04 containerized
environment with a 3080 Ti.

# Progress
- [x] Train HalfCheetah to run in MuJoCo
- [x] Fine-tune GPT2-medium
- [ ] Train a Humanoid to walk, stop, recover
- [ ] Integrate LLM to translate live instructions to actions (may require additional fine-tuning or SOA API)
- [ ] Create a rich MuJoCo sandbox environment
- [ ] Scale up to distributed training pipeline
- [ ] TBD
