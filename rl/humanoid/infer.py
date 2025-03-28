import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import numpy as np

CHECKPOINT_PATH = "./workspace_humanoid/checkpoints/best_model.zip"
NORMALIZE_PATH = "./workspace_humanoid/vecnormalize.pkl"

def make_env():
    return lambda: Monitor(gym.make("Humanoid-v5", render_mode="human"))

env = DummyVecEnv([make_env()])
env = VecNormalize.load(NORMALIZE_PATH, env)
env.training = False
env.norm_reward = False

model = PPO.load(CHECKPOINT_PATH, env=env, device="cpu")

obs = env.reset()

done, truncated = False, False
total_reward = 0
while not (done or truncated):
    action, _ = model.predict(np.array(obs), deterministic=True)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    env.render()
env.reset()
env.close()

print(f"Reward: {total_reward}")
