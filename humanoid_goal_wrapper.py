import gymnasium as gym
import numpy as np
from goal import Goal, GoalSampler


class HumanoidGoalWrapper(gym.Wrapper):
	def __init__(self, env, goal_sampler: GoalSampler):
		super().__init__(env)
		self.goal_sampler = goal_sampler
		self.goal: Goal = goal_sampler.peek()
		self.prev_info = {}

	def reset(self, **kwargs):
		self.goal = self.goal_sampler.next()

		obs, info = self.env.reset(**kwargs)
		self.prev_info = info.copy()

		info["goal"] = self.goal.name if self.goal is not None else None
		return obs, info

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		goal_reward = self.goal.compute_reward(self.prev_info, info)
		reward = float(reward) + goal_reward

		if self.goal.check_termination(info):
			terminated = True

		info["goal"] = self.goal.name
		info["goal_reward"] = goal_reward
		self.prev_info = info.copy()

		return obs, reward, terminated, truncated, info
