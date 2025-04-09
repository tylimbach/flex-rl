from gymnasium import Wrapper, Env
from gymnasium.core import ObsType, ActType
from typing import Any, override

from ..envs import GoalSampler, Goal

class HumanoidGoalWrapper(Wrapper[ObsType, ActType, ObsType, ActType]):
	def __init__(self, env: Env[ObsType, ActType], goal_sampler: GoalSampler):
		super().__init__(env)
		self.goal_sampler = goal_sampler
		self.goal: Goal = goal_sampler.peek()
		self.prev_info: dict[str, Any] = {}

	@override
	def reset(self, **kwargs) -> tuple[ObsType, dict[str, Any]]:
		self.goal = self.goal_sampler.next()
		obs, info = self.env.reset(**kwargs)
		self.prev_info = info.copy()
		info["goal"] = self.goal.name
		return obs, info

	@override
	def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
		obs, reward, terminated, truncated, info = self.env.step(action)
		goal_reward = self.goal.compute_reward(self.prev_info, info)
		reward = float(reward) + goal_reward

		if self.goal.check_termination(info):
			terminated = True

		info["goal"] = self.goal.name
		info["goal_reward"] = goal_reward
		self.prev_info = info.copy()

		return obs, reward, terminated, truncated, info
