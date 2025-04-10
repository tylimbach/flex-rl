from typing import Any, Callable, Literal, final, Self
import random

RewardFn = Callable[[dict[str, Any], dict[str, Any]], float]
TerminationFn = Callable[[dict[str, Any]], bool]

@final
class Goal:
	def __init__(self, name: str, reward_fn: RewardFn, termination_fn: TerminationFn | None = None, weight: float = 1.0):
		self.name = name
		self.reward_fn = reward_fn
		self.termination_fn = termination_fn or (lambda info: info.get("terminated", False))
		self.weight = weight

	def compute_reward(self, prev_info: dict[str, Any], curr_info: dict[str, Any]) -> float:
		return self.weight * self.reward_fn(prev_info, curr_info)

	def check_termination(self, curr_info: dict[str, Any]) -> bool:
		return self.termination_fn(curr_info)

	@classmethod
	def from_cfg(cls, name: str, weight: float) -> Self:
		builtin = get_builtin_goals().get(name)
		if builtin is not None:
			return cls(name, builtin.reward_fn, builtin.termination_fn, weight)
		raise Exception(f"No goal functions found for name: {name}")


def walk_forward_reward(prev: dict[str, Any], curr: dict[str, Any]) -> float:
	return curr.get("x_position", 0.0) - prev.get("x_position", 0.0)

def walk_backward_reward(prev: dict[str, Any], curr: dict[str, Any]) -> float:
	return - (curr.get("x_position", 0.0) - prev.get("x_position", 0.0))

def turn_left_reward(prev: dict[str, Any], curr: dict[str, Any]) -> float:
	return - (curr.get("y_position", 0.0) - prev.get("y_position", 0.0))

def turn_right_reward(prev: dict[str, Any], curr: dict[str, Any]) -> float:
	return curr.get("y_position", 0.0) - prev.get("y_position", 0.0)

def stand_still_reward(prev: dict[str, Any], curr: dict[str, Any]) -> float:
	dx = curr.get("x_position", 0.0) - prev.get("x_position", 0.0)
	dy = curr.get("y_position", 0.0) - prev.get("y_position", 0.0)
	return - (abs(dx) + abs(dy))

def jump_reward(prev: dict[str, Any], curr: dict[str, Any]) -> float:
	return float(curr.get("z_position", 0.0) > 1.4)

def jump_termination(info: dict[str, Any]) -> bool:
	return info.get("z_position", 0.0) < 0.8


def get_builtin_goals() -> dict[str, Goal]:
	return {
		"walk_forward": Goal("walk_forward", walk_forward_reward),
		"walk_backward": Goal("walk_backward", walk_backward_reward),
		"turn_left": Goal("turn_left", turn_left_reward),
		"turn_right": Goal("turn_right", turn_right_reward),
		"stand_still": Goal("stand_still", stand_still_reward),
		"jump": Goal("jump", jump_reward, jump_termination)
	}


@final
class GoalSampler:
	def __init__(
		self, 
		goals: list[Goal],
		strategy: str = "balanced"
	):
		self.goals = goals
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
