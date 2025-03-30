import gymnasium as gym
import numpy as np

GOALS = ["walk forward", "turn left", "turn right"]

class HumanoidGoalWrapper(gym.Wrapper):
	def __init__(self, env):
		super().__init__(env)
		self.goal = None
		self.debug = False  # Set this to False to turn off debug output

	def reset(self, **kwargs):
		self.goal = np.random.choice(GOALS)
		obs, info = self.env.reset(**kwargs)
		info["goal"] = self.goal
		if self.debug:
			print("[Reset] Goal:", self.goal)
			print("[Reset] Initial obs[:10]:", np.array2string(obs[:10], precision=3))
		return obs, info

	def step(self, action):
		obs, reward, terminated, truncated, info = self.env.step(action)
		if self.goal == "walk forward":
			reward += obs[0]
		elif self.goal == "turn left":
			reward += -obs[1]
		elif self.goal == "turn right":
			reward += obs[1]
		info["goal"] = self.goal
		if self.debug:
			print("[Step] Goal:", self.goal)
			print("[Step] Obs[:10]:", np.array2string(obs[:10], precision=3))
			print("[Step] Reward:", reward)
		return obs, reward, terminated, truncated, info
