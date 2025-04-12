import numpy as np

from gymnasium import Wrapper, Env
from gymnasium.core import ObsType, ActType
from typing import Any, final, override

from .goal import GoalSampler, StepHistory, StepResult

@final
class HumanoidGoalWrapper(Wrapper[ObsType, ActType, ObsType, ActType]):
	def __init__(self, env: Env[ObsType, ActType], goal_sampler: GoalSampler):
		super().__init__(env)
		self.step_history = StepHistory()
		self.goal_sampler = goal_sampler
		self.goal = goal_sampler.peek()

	@override
	def reset(self, **kwargs) -> tuple[ObsType, dict[str, Any]]:
		self.goal = self.goal_sampler.next()

		obs, info = self.env.reset(**kwargs)
		self.step_history.clear()
		self.goal.init(self.env)

		return obs, info

	@override
	def step(self, action: ActType) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
		obs, reward, terminated, truncated, info = self.env.step(action)
		step_result = StepResult(obs, float(reward), terminated, truncated, info)	

		self.step_history.push(step_result)
		reward = self.goal.compute_reward(self.step_history)
		terminated = self.goal.check_termination(step_result)
		
		# update mutable step result
		step_result.reward = reward
		step_result.terminated = terminated

		return obs, reward, terminated, truncated, info
