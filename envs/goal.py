from typing import Literal, List 
import random

class Goal:
	def __init__(self, name, reward_fn, termination_fn=None, weight=1.0):
		self.name = name
		self.reward_fn = reward_fn
		self.termination_fn = termination_fn or (lambda info: info.get("terminated", False))
		self.weight = weight

	def compute_reward(self, prev_info, curr_info):
		return self.weight * self.reward_fn(prev_info, curr_info)

	def check_termination(self, curr_info):
		return self.termination_fn(curr_info)


def walk_forward_reward(prev, curr):
	return curr.get("x_position", 0.0) - prev.get("x_position", 0.0)

def walk_backward_reward(prev, curr):
	return - (curr.get("x_position", 0.0) - prev.get("x_position", 0.0))

def turn_left_reward(prev, curr):
	return - (curr.get("y_position", 0.0) - prev.get("y_position", 0.0))

def turn_right_reward(prev, curr):
	return curr.get("y_position", 0.0) - prev.get("y_position", 0.0)

def stand_still_reward(prev, curr):
	dx = curr.get("x_position", 0.0) - prev.get("x_position", 0.0)
	dy = curr.get("y_position", 0.0) - prev.get("y_position", 0.0)
	return - (abs(dx) + abs(dy))

def jump_reward(prev, curr):
	return float(curr.get("z_position", 0.0) > 1.4)

def jump_termination(info):
	return info.get("z_position", 0.0) < 0.8


def get_builtin_goals():
	return {
		"walk_forward": Goal("walk_forward", walk_forward_reward),
		"walk_backward": Goal("walk_backward", walk_forward_reward),
		"turn_left": Goal("turn_left", turn_left_reward),
		"turn_right": Goal("turn_right", turn_right_reward),
		"stand_still": Goal("stand_still", stand_still_reward),
		"jump": Goal("jump", jump_reward, jump_termination)
	}


def load_goals_from_config(goal_cfg_list):
	available = get_builtin_goals()
	loaded = []
	for cfg in goal_cfg_list:
		name = cfg["name"]
		weight = cfg.get("weight", 1.0)
		if name not in available:
			raise ValueError(f"Unknown goal: {name}. Must be one of: {list(available.keys())}")
		goal = available[name]
		goal.weight = weight
		loaded.append(goal)
	return loaded


class GoalSampler:
	def __init__(
		self, 
		goals: List[Goal],
		strategy: Literal["single", "balanced", "random"] = "balanced"
	):
		self.goals: List[Goal] = goals
		self.strategy = strategy
		self.index = 0

		if strategy not in ["single", "balanced", "random"]:
			raise ValueError(f"Unknown sampling strategy: {strategy}")

	def next(self) -> Goal:
		if self.strategy == "single":
			return self.goals[self.index]
		elif self.strategy == "balanced":
			goal = self.goals[self.index]
			self.index = (self.index + 1) % len(self.goals)
			return goal
		elif self.strategy == "random":
			return random.choice(self.goals)
		else:
			raise ValueError(f"Tried sampling with undefined strategy: {self.strategy}")

	def peek(self) -> Goal:
		return self.goals[self.index]

	@classmethod
	def single(cls, goal: Goal) -> "GoalSampler":
		return cls([goal], strategy="single")
