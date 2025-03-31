import os
import argparse
import numpy as np
import gymnasium as gym
from moviepy import ImageSequenceClip
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
import yaml
from sampling.goal_sampler import GoalSampler
from humanoid_goal_wrapper import HumanoidGoalWrapper


def make_env(env_id, goal_sampler, render_mode):
	def _init():
		env = Monitor(HumanoidGoalWrapper(gym.make(env_id, render_mode=render_mode), goal_sampler=goal_sampler))
		return env
	return _init


def save_media(frames, path, fps=30):
	if path.endswith(".mp4"):
		clip = ImageSequenceClip(frames, fps=fps)
		clip.write_videofile(path, fps=fps, codec="libx264")
	elif path.endswith(".gif"):
		clip = ImageSequenceClip(frames, fps=fps)
		clip.write_gif(path, fps=fps)
	else:
		raise ValueError("Output must be .mp4 or .gif")

	print(f"🎥 Saved media to {path}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, required=True, help="Path to model snapshot")
	parser.add_argument("--video", type=str, help="Optional: save an episode as video (.mp4)")
	parser.add_argument("--gif", type=str, help="Optional: save an episode as GIF (.gif)")
	args = parser.parse_args()

	model_path = os.path.join(args.model, "model.zip")
	vecnorm_path = os.path.join(args.model, "vecnormalize.pkl")
	config_path = os.path.join(args.model, "metadata.yaml")

	print(f"🔍 Loading model from: {model_path}")
	print(f"🔍 Loading normalization stats from: {vecnorm_path}")
	print(f"🔍 Loading config from: {config_path}")

	with open(config_path) as f:
		metadata = yaml.safe_load(f)
	cfg = metadata["config"]
	train_goals = cfg.get("sampling_goals", ["walk forward", "turn left", "turn right", "stand still"])

	goal_sampler = GoalSampler(
		strategy="balanced",
		goals=train_goals
	)

	render_mode = "rgb_array" if args.video else "human"
	env = DummyVecEnv([make_env(cfg["env_id"], goal_sampler, render_mode)])
	env = VecNormalize.load(vecnorm_path, env)
	env.training = False
	env.norm_reward = False

	model = RecurrentPPO.load(model_path, env=env, device="cuda")

	obs = env.reset()
	if args.video or args.gif:
		output_path = args.video or args.gif
		print(f"🎥 Recording one episode to {output_path}")
		lstm_states = None
		episode_starts = np.ones((env.num_envs,), dtype=bool)
		frames = []
		total_reward = 0
		done = False
		info = [{}]

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
			frame = env.envs[0].render()
			frames.append(frame)

		print(f"🎯 Episode reward: {total_reward:.2f} | Goal: {info[0].get('goal')}")
		save_media(frames, output_path)
	else:
		print(f"🎮 Starting live interactive play...")
		while True:
			try:
				lstm_states = None
				episode_starts = np.ones((env.num_envs,), dtype=bool)
				total_reward = 0
				done = False
				info = [{}]

				while not done:
					action, lstm_states = model.predict(
						np.array(obs),
						state=lstm_states,
						episode_start=episode_starts,
						deterministic=True
					)
					obs, reward, done, info = env.step(action)
					total_reward += reward[0]
					env.render()
					episode_starts = done

				print(f"🎯 Episode reward: {total_reward:.2f} | Goal: {info[0].get('goal')}")
			except Exception:
				env.reset()
				env.close()
