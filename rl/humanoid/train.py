import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os

LOAD_DIR = "./workspace_humanoid/checkpoints"
RENDER_EVAL = True
EVAL_FREQ = 100_000

N_STEPS = 8192
GAE_LAMBDA = 0.97
ENT_COEF = 0.001
LEARNING_RATE = 3e-4
TOTAL_TIMESTEPS = 10_000_000

def make_env(render_mode=None):
    return lambda: Monitor(gym.make("Humanoid-v5", render_mode=render_mode))

train_env = DummyVecEnv([make_env()])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

os.makedirs("./workspace_humanoid", exist_ok=True)
train_env.save("./workspace_humanoid/vecnormalize.pkl")

# Load or create model
if os.path.exists(LOAD_DIR):
    print(f"Loading model from {LOAD_DIR}")
    model = PPO.load(LOAD_DIR, env=train_env, device="cpu")
else:
    print("Creating new PPO model for Humanoid")
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=N_STEPS,
        gae_lambda=GAE_LAMBDA,
        ent_coef=ENT_COEF,
        learning_rate=LEARNING_RATE,
        device="cpu",
        verbose=1,
        tensorboard_log="./workspace_humanoid/ppo_tensorboard/"
    )

eval_env = DummyVecEnv([make_env()])
eval_env = VecNormalize.load("./workspace_humanoid/vecnormalize.pkl", eval_env)
eval_env.training = False
eval_env.norm_reward = False

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./workspace_humanoid/checkpoints/",
    log_path="./workspace_humanoid/logs/",
    eval_freq=EVAL_FREQ,
    deterministic=True,
)

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)

model.save("./workspace_humanoid/checkpoints/ppo_humanoid_final")
train_env.close()
eval_env.close()
print("Humanoid training complete.")
