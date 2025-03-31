import gymnasium as gym
import numpy as np
import torch
import argparse
import os
import yaml
from moviepy import ImageSequenceClip

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from sampling.goal_sampler import GoalSampler
from humanoid_goal_wrapper import HumanoidGoalWrapper


def make_env(env_id, goal_sampler):
	def _init():
		env = Monitor(HumanoidGoalWrapper(gym.make(env_id, render_mode="rgb_array"), goal_sampler=goal_sampler))
		return env
	return _init


def save_video(frames, path, fps=30):
	clip = ImageSequenceClip(frames, fps=fps)
	clip.write_videofile(path, codec="libx264")
	print(f"🎥 Saved video to {path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, required=True, help="Snapshot folder to run")
	parser.add_argument("--video", type=str, help="Optional: Path to save rendered video")
	args = parser.parse_args()

	model_path = os.path.join(args.model, "model.zip")
	vecnorm_path = os.path.join(args.model, "vecnormalize.pkl")
	metadata_path = os.path.join(args.model, "metadata.yaml")

	if not all(os.path.exists(p) for p in [model_path, vecnorm_path, metadata_path]):
		raise FileNotFoundError(f"Missing snapshot files in {args.model}")

	print(f"🔍 Loading model from: {model_path}")
	print(f"🔍 Loading normalization stats from: {vecnorm_path}")
	print(f"🔍 Loading config from: {metadata_path}")

	with open(metadata_path) as f:
		metadata = yaml.safe_load(f)
	cfg = metadata["config"]

	train_goals = cfg.get("sampling_goals")

	goal_sampler = GoalSampler(
		strategy="balanced",
		goals=train_goals
	)

	env = DummyVecEnv([make_env(cfg["env_id"], goal_sampler)])
	env = VecNormalize.load(vecnorm_path, env)
	env.training = False
	env.norm_reward = False

	model = RecurrentPPO.load(model_path, env=env, device="cuda")

	if args.video:
		print(f"🎥 Recording one episode to {args.video}")
	else:
		print("🎮 Starting interactive play")

	obs = env.reset()
	while True:
		try:
			lstm_states = None
			episode_starts = np.ones((env.num_envs,), dtype=bool)
			total_reward = 0.0
			frames = []
			info = [{}]

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

				if args.video:
					frame = env.envs[0].render()
					frames.append(frame)
				else:
					env.render()

			print(f"🎯 Episode reward: {total_reward:.2f} | Goal: {info[0].get('goal')}")

			if args.video:
				save_video(frames, args.video)
				break

		except KeyboardInterrupt:
			env.close()
			break
		finally:
			import gymnasium as gym
			from gymnasium.envs.mujoco import mujoco_rendering

			mujoco_rendering.OffScreenViewer.__del__ = lambda self: None
