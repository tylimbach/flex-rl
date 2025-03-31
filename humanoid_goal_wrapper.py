from typing import Optional, Any, Dict, Tuple
import numpy as np
import gymnasium as gym
from sampling.goal_sampler import GoalSampler

class HumanoidGoalWrapper(gym.Wrapper):
	def __init__(self, env: gym.Env, goal_sampler: GoalSampler, goal_reward_scale: float = 1.0, debug: bool = False):
		super().__init__(env)
		self.goal_sampler = goal_sampler
		self.goal: Optional[str] = None
		self.last_x = 0.0
		self.last_y = 0.0
		self.goal_reward_scale = goal_reward_scale
		self.debug = debug

	def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
		options = kwargs.get("options", {})
		goal = options.get("goal") or self.goal_sampler.next()

		self.goal = goal
		obs, info = self.env.reset(**kwargs)
		self.last_x = info.get("x_position", 0.0)
		self.last_y = info.get("y_position", 0.0)
		info["goal"] = self.goal

		if self.debug:
			print(f"[Reset] Goal: {self.goal}")
			print(f"[Reset] x: {self.last_x:.3f}, y: {self.last_y:.3f}")

		return obs, info

	def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
		obs, reward, terminated, truncated, info = self.env.step(action)
		x_pos = info.get("x_position", 0.0)
		y_pos = info.get("y_position", 0.0)
		delta_x = x_pos - self.last_x
		delta_y = y_pos - self.last_y

		goal_reward = 0.0
		if self.goal == "walk forward":
			goal_reward = delta_x
		elif self.goal == "turn left":
			goal_reward = -delta_y
		elif self.goal == "turn right":
			goal_reward = delta_y
		elif self.goal == "stand still":
			goal_reward = - (abs(delta_x) + abs(delta_y))

		reward = float(reward) + self.goal_reward_scale * goal_reward

		self.last_x = x_pos
		self.last_y = y_pos
		info["goal"] = self.goal
		info["goal_reward"] = goal_reward

		if self.debug:
			print(f"[Step] Goal: {self.goal}")
			print(f"[Step] Δx: {delta_x:.3f}, Δy: {delta_y:.3f}, Goal Reward: {goal_reward:.3f}")
			if terminated:
				print("[Step] Terminated - likely fell")

		return obs, reward, terminated, truncated, info
