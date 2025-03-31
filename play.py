import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from humanoid_goal_wrapper import HumanoidGoalWrapper
from sampling.goal_sampler import GoalSampler
import numpy as np
import os
import argparse
import yaml


def make_env(env_id, goal_sampler: GoalSampler):
	def _init():
		env = Monitor(
			HumanoidGoalWrapper(
				gym.make(env_id, render_mode="human"),
				goal_sampler=goal_sampler
			)
		)
		return env
	return _init


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, required=True, help="Path to model snapshot")
	args = parser.parse_args()

	model_path = os.path.join(args.model, "model.zip")
	vecnorm_path = os.path.join(args.model, "vecnormalize.pkl")
	config_path = os.path.join(args.model, "metadata.yaml")

	if not os.path.exists(model_path) or not os.path.exists(vecnorm_path):
		raise FileNotFoundError(f"Missing model or VecNormalize in {args.model}")
	if not os.path.exists(config_path):
		raise FileNotFoundError(f"Missing metadata.yaml in {args.model}")

	print(f"🔍 Loading model from: {model_path}")
	print(f"🔍 Loading normalization stats from: {vecnorm_path}")
	print(f"🔍 Loading config from: {config_path}")

	with open(config_path) as f:
		metadata = yaml.safe_load(f)
	cfg = metadata["config"]

	train_goals = cfg.get("train_goals", ["walk forward", "turn left", "turn right", "stand still"])

	goal_sampler = GoalSampler(
		strategy="balanced",
		goals=train_goals
	)

	env = DummyVecEnv([make_env(cfg["env_id"], goal_sampler)])
	env = VecNormalize.load(vecnorm_path, env)
	env.training = False
	env.norm_reward = False

	model = RecurrentPPO.load(model_path, env=env, device="cuda")

	print(f"🎮 Starting interactive play with goals: {train_goals}")

	obs = env.reset()
	while True:
		try:
			lstm_states = None
			episode_starts = np.ones((env.num_envs,), dtype=bool)
			total_reward = 0.0
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
				env.render()

			print(f"🎯 Episode reward: {total_reward:.2f} | Goal: {info[0].get('goal')}")
		except KeyboardInterrupt:
			print("👋 Exiting...")
			break
		except Exception as e:
			print(f"⚠️ Exception: {e}")
	env.close()
