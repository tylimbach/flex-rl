import os
import yaml
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO
from sampling.goal_sampler import GoalSampler
from humanoid_goal_wrapper import HumanoidGoalWrapper


def make_eval_env(env_id, goals):
	def _init():
		sampler = GoalSampler(strategy="balanced", goals=[goals])
		env = Monitor(HumanoidGoalWrapper(gym.make(env_id), goal_sampler=sampler))
		return env
	return _init


def evaluate_snapshot(snapshot_path, eval_episodes=10):
	model_path = os.path.join(snapshot_path, "model.zip")
	vecnorm_path = os.path.join(snapshot_path, "vecnormalize.pkl")
	metadata_path = os.path.join(snapshot_path, "metadata.yaml")

	if not all(os.path.exists(p) for p in [model_path, vecnorm_path, metadata_path]):
		raise FileNotFoundError(f"Missing files in snapshot: {snapshot_path}")

	with open(metadata_path) as f:
		metadata = yaml.safe_load(f)
		cfg = metadata["config"]
		goals = cfg.get("sampling_goals", ["walk forward", "turn left", "turn right", "stand still"])

	results = {}

	for goal in goals:
		env = DummyVecEnv([make_eval_env(cfg["env_id"], goal)])
		env = VecNormalize.load(vecnorm_path, env)
		env.training = False
		env.norm_reward = False

		model = RecurrentPPO.load(model_path, env=env, device="cuda")

		all_rewards = []

		for _ in range(eval_episodes):
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

			all_rewards.append(total_reward)

		mean_reward = np.mean(all_rewards)
		results[goal] = mean_reward
		print(f"✅ Goal '{goal}': Mean reward over {eval_episodes} episodes: {mean_reward:.2f}")

	return results


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("--snapshot", type=str, required=True, help="Path to snapshot folder")
	parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes per goal")
	args = parser.parse_args()

	evaluate_snapshot(args.snapshot, args.episodes)
