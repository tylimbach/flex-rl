from typing import Literal, Optional, List
import numpy as np

GOALS = ["walk forward", "turn left", "turn right", "stand still"]

class GoalSampler:
	def __init__(self, strategy: Literal["balanced", "random"] = "balanced", 
	             goals: List[str] = GOALS):
		self.goals = goals or GOALS
		self.strategy = strategy
		self.index = 0

		if strategy not in ["balanced", "random"]:
			raise ValueError(f"Unknown sampling strategy: {strategy}")

	def next(self):
		if self.strategy == "balanced":
			goal = self.goals[self.index]
			self.index = (self.index + 1) % len(self.goals)
			return goal
		elif self.strategy == "random":
			return np.random.choice(self.goals)
