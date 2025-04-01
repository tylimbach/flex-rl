import os
import argparse
import numpy as np
import torch
import yaml
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from sampling.goal_sampler import GoalSampler
from humanoid_goal_wrapper import HumanoidGoalWrapper
from train_offline_bc import MLPPolicy, goal_to_idx
import imageio


def make_env(env_id, goal_sampler):
	def _init():
		env = Monitor(HumanoidGoalWrapper(gym.make(env_id, render_mode="rgb_array"), goal_sampler=goal_sampler))
		return env
	return _init


def play_bc(model_path, video_path=None):
	print(f"🔍 Loading offline policy from {model_path}")
	base_dir = os.path.dirname(model_path)
	metadata_path = os.path.join(base_dir, "metadata.yaml")
	vecnorm_path = os.path.join(base_dir, "vecnormalize.pkl")

	with open(metadata_path) as f:
		metadata = yaml.safe_load(f)
	cfg = metadata["config"]
	train_goals = cfg.get("sampling_goals")

	goal_sampler = GoalSampler(strategy="balanced", goals=train_goals)

	env = DummyVecEnv([make_env(cfg["env_id"], goal_sampler)])
	env = VecNormalize.load(vecnorm_path, env)
	env.training = False
	env.norm_reward = False

	obs_dim = env.observation_space.shape[-1]
	action_dim = env.action_space.shape[-1]
	model = MLPPolicy(obs_dim, action_dim).cuda()
	model.load_state_dict(torch.load(model_path))
	model.eval()

	print(f"🎯 Starting BC policy play | Goals: {train_goals}")

	obs = env.reset()
	info = [{}]
	if video_path:
		print(f"🎥 Recording one episode to {video_path}")
		frames = []

		total_reward = 0.0
		done = False

		while not done:
			goal = env.get_attr("goal")[0]
			goal_idx = torch.tensor([goal_to_idx(goal)], dtype=torch.long).cuda()
			obs_tensor = torch.tensor(obs, dtype=torch.float32).cuda()
			action = model(obs_tensor, goal_idx).cpu().detach().numpy()
			obs, reward, done, info = env.step(action)
			total_reward += reward[0]
			frame = env.envs[0].render(mode="rgb_array")
			frames.append(frame)

		imageio.mimsave(video_path, frames, fps=30)
		print(f"🎥 Saved video to {video_path}")
	else:
		while True:
			total_reward = 0.0
			done = False

			while not done:
				goal = env.get_attr("goal")[0]
				goal_idx = torch.tensor([goal_to_idx(goal)], dtype=torch.long).cuda()
				obs_tensor = torch.tensor(obs, dtype=torch.float32).cuda()
				action = model(obs_tensor, goal_idx).cpu().detach().numpy()
				obs, reward, done, info = env.step(action)
				total_reward += reward[0]
				env.render()

			print(f"🎯 Episode reward: {total_reward:.2f} | Goal: {goal}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, required=True, help="Path to offline policy weights")
	parser.add_argument("--video", type=str, help="Optional output video path")
	args = parser.parse_args()

	play_bc(args.model, args.video)
