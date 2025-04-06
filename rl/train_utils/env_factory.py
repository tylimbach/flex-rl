import os
from stable_baselines3.common.monitor import Monitor
from rl.envs.humanoid_goal_wrapper import HumanoidGoalWrapper
import gymnasium as gym


def make_env(env_id, goal_sampler, render_mode=None):
	def _init():
		env = Monitor(HumanoidGoalWrapper(gym.make(env_id, render_mode=render_mode), goal_sampler))
		return env
	return _init


def get_unique_experiment_dir(base_path):
	if not os.path.exists(base_path):
		return base_path
	i = 1
	while True:
		new_path = f"{base_path}_{i}"
		if not os.path.exists(new_path):
			return new_path
		i += 1
