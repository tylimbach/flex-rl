import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os
import argparse

def make_env():
	return lambda: Monitor(gym.make("Humanoid-v5", render_mode="human"))

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--snapshot", type=str, required=True, help="Path to snapshot folder to load")
	args = parser.parse_args()

	env = DummyVecEnv([make_env()])
	env = VecNormalize.load(os.path.join(args.snapshot, "vecnormalize.pkl"), env)
	env.training = False
	env.norm_reward = False

	model = RecurrentPPO.load(os.path.join(args.snapshot, "model.zip"), env=env, device="cuda")

	obs = env.reset()
	lstm_states = None
	episode_starts = np.ones((env.num_envs,), dtype=bool)
	total_reward = 0.0

	done, truncated = False, False
	while not (done or truncated):
		action, lstm_states = model.predict(
			np.array(obs),
			state=lstm_states,
			episode_start=episode_starts,
			deterministic=True
		)
		obs, reward, done, info = env.step(action)
		total_reward += reward[0]
		episode_starts = done
		env.render()

	env.reset()
	env.close()

	print(f"Episode total reward: {total_reward:.2f}")
