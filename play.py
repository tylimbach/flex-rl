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
	parser.add_argument("--model", type=str, required=True, help="Path to model snapshot")
	args = parser.parse_args()

	model_path = os.path.join(args.model, "model.zip")
	vecnorm_path = os.path.join(args.model, "vecnormalize.pkl")

	if not os.path.exists(model_path) or not os.path.exists(vecnorm_path):
		raise FileNotFoundError(f"Missing model or VecNormalize in {args.model}")

	print(f"🔍 Loading model from: {model_path}")
	print(f"🔍 Loading normalization stats from: {vecnorm_path}")

	env = DummyVecEnv([make_env()])
	env = VecNormalize.load(vecnorm_path, env)
	env.training = False
	env.norm_reward = False

	model = RecurrentPPO.load(model_path, env=env, device="cuda")

	running = True
	while running:
		try:
			obs = env.reset()
			lstm_states = None
			episode_starts = np.ones((env.num_envs,), dtype=bool)
			total_reward = 0.0

			done = False
			while not done:
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

			print(f"🎯 Episode total reward: {total_reward:.2f}")
		except Exception:
			env.reset()
			env.close()
