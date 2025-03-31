import os
import argparse
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sb3_contrib import RecurrentPPO
import yaml
from sampling.goal_sampler import GoalSampler
from humanoid_goal_wrapper import HumanoidGoalWrapper


SAVE_DIR = "trajectories"


def make_env(env_id, goal_sampler):
	def _init():
		env = Monitor(HumanoidGoalWrapper(gym.make(env_id), goal_sampler=goal_sampler))
		return env
	return _init


def record_trajectories(snapshot_path, num_episodes):
	model_path = os.path.join(snapshot_path, "model.zip")
	vecnorm_path = os.path.join(snapshot_path, "vecnormalize.pkl")
	metadata_path = os.path.join(snapshot_path, "metadata.yaml")

	if not all(os.path.exists(p) for p in [model_path, vecnorm_path, metadata_path]):
		raise FileNotFoundError("Missing snapshot files.")

	with open(metadata_path) as f:
		metadata = yaml.safe_load(f)
		cfg = metadata["config"]
		train_goals = cfg.get("sampling_goals", ["walk forward", "turn left", "turn right", "stand still"])

	goal_sampler = GoalSampler(
		strategy="balanced",
		goals=train_goals
	)

	env = DummyVecEnv([make_env(cfg["env_id"], goal_sampler)])
	env = VecNormalize.load(vecnorm_path, env)
	env.training = False
	env.norm_reward = False

	model = RecurrentPPO.load(model_path, env=env, device="cuda")

	target_dir = os.path.join(os.path.dirname(snapshot_path), SAVE_DIR)
	os.makedirs(target_dir, exist_ok=True)

	print(f"🎯 Recording {num_episodes} episodes to {target_dir}")
	episode_count = 0
	total_rewards = []
	obs = env.reset()

	while episode_count < num_episodes:
		lstm_states = None
		episode_starts = np.ones((env.num_envs,), dtype=bool)

		trajectory = {
			"observations": [],
			"actions": [],
			"rewards": [],
			"dones": [],
			"goal": None
		}

		total_reward = 0
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
			trajectory["observations"].append(obs.copy())
			trajectory["actions"].append(action.copy())
			trajectory["rewards"].append(reward.copy())
			trajectory["dones"].append(done.copy())
			if trajectory["goal"] is None:
				trajectory["goal"] = info[0].get("goal")
			episode_starts = done

		total_rewards.append(total_reward)
		episode_count += 1

		episode_file = os.path.join(target_dir, f"episode_{episode_count:05d}.npz")
		np.savez_compressed(
			episode_file,
			observations=np.array(trajectory["observations"]),
			actions=np.array(trajectory["actions"]),
			rewards=np.array(trajectory["rewards"]),
			dones=np.array(trajectory["dones"]),
			goal=trajectory["goal"]
		)

		print(f"✅ Saved episode {episode_count} | Goal: {trajectory['goal']} | Total reward: {total_reward:.2f}")

	print(f"🎯 Finished recording. Mean reward: {np.mean(total_rewards):.2f}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, required=True, help="Snapshot folder to record from")
	parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to record")
	args = parser.parse_args()

	record_trajectories(args.model, args.episodes)
