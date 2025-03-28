import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import numpy as np

CHECKPOINT_PATH = "./workspace/checkpoints/best_model_cheetah_v5/best_model.zip"
NORMALIZE_PATH = "./workspace/vecnormalize.pkl"

def make_env():
    return lambda: gym.make("HalfCheetah-v5", render_mode="human")

env = DummyVecEnv([lambda: Monitor(make_env()())])
env = VecNormalize.load(NORMALIZE_PATH, env)
env.training = False
env.norm_reward = False

model = PPO.load(CHECKPOINT_PATH, env=env, device="cpu")

obs = env.reset()

done, truncated = False, False
total_reward = 0
while not (done or truncated):
    action, _ = model.predict(np.array(obs), deterministic=False)
    obs, reward, done, info = env.step(action)
    total_reward += reward
    env.render()
env.reset()
env.close()

print(f"Reward: {total_reward}")
