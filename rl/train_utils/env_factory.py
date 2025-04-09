import os
from typing import Any
from stable_baselines3.common.monitor import Monitor
from rl.envs.goal import GoalSampler
from rl.envs.humanoid_goal_wrapper import HumanoidGoalWrapper
import gymnasium as gym


def make_env(env_id: str, goal_sampler: GoalSampler, render_mode: str | None = None):
	def _init() -> Monitor[Any, Any]:
		env = Monitor(HumanoidGoalWrapper(gym.make(env_id, render_mode=render_mode), goal_sampler))
		return env
	return _init


def get_unique_experiment_dir(base_path: str) -> str:
	if not os.path.exists(base_path):
		return base_path
	i = 1
	while True:
		new_path = f"{base_path}_{i}"
		if not os.path.exists(new_path):
			return new_path
		i += 1
