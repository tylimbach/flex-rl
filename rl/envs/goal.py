import numpy as np
from dataclasses import dataclass
from collections import deque
from typing import Any, Callable, final
from gymnasium.core import Env
import random

from omegaconf import OmegaConf

@dataclass
class StepResult:
	obs: Any
	reward: float
	terminated: bool
	truncated: bool
	info: dict[str, Any]

class StepHistory:
	def __init__(self, maxlen: int = 10):
		self._buffer: deque[StepResult] = deque(maxlen=maxlen)

	def push(self, step: StepResult) -> None:
		self._buffer.append(step)

	def clear(self) -> None:
		self._buffer.clear()

	def last(self) -> StepResult | None:
		return self._buffer[-1] if self._buffer else None

	def prev(self, n: int = 1) -> StepResult | None:
		if len(self._buffer) > n:
			return self._buffer[-(n + 1)]
		return None

	def first(self) -> StepResult | None:
		return self._buffer[0] if self._buffer else None

	def delta(self, field: Callable[[StepResult], float], steps: int = 1) -> float:
		a = self.prev(steps)
		b = self.last()
		if a is None or b is None:
			return 0.0
		return field(b) - field(a)

	def smooth(self, field: Callable[[StepResult], float]) -> float:
		if not self._buffer:
			return 0.0
		return sum(field(step) for step in self._buffer) / len(self._buffer)

	def __len__(self):
		return len(self._buffer)

	def __getitem__(self, idx: int) -> StepResult:
		return self._buffer[idx]


RewardFn = Callable[[StepHistory], float]
TerminationFn = Callable[[StepResult], bool]
InitFn = Callable[[Env[Any, Any]], None]

@dataclass
class RewardWeights:
	forward: float = 0.0
	backward: float = 0.0
	turn: float = 0.0
	stand: float = 0.0
	stand_up: float = 0.0
	jump: float = 0.0
	survive: float = 0.0
	ctrl: float = 0.0
	contact: float = 0.0

def terminate_default(cur: StepResult) -> bool:
	return cur.terminated

def compute_reward_from_weights(history: StepHistory, weights: RewardWeights) -> float:
	total = 0.0

	if weights.forward != 0.0:
		total += weights.forward * reward_forward(history)
	if weights.backward != 0.0:
		total += weights.backward * reward_backward(history)
	if weights.turn != 0.0:
		total += weights.turn * reward_turn(history)
	if weights.stand != 0.0:
		total += weights.stand * reward_stand(history)
	if weights.stand_up != 0.0:
		total += weights.stand_up * reward_stand_up(history)
	if weights.jump != 0.0:
		total += weights.jump * reward_jump(history)
	if weights.survive != 0.0:
		total += weights.survive * reward_survive(history)
	if weights.ctrl != 0.0:
		total += weights.ctrl * reward_ctrl(history)
	if weights.contact != 0.0:
		total += weights.contact * reward_contact(history)

	return total

@dataclass
class Goal:
	weights: RewardWeights
	weight: float = 1.0
	termination_fn: TerminationFn = terminate_default
	init_fn: InitFn | None = None

	def compute_reward(self, history: StepHistory) -> float:
		return self.weight * compute_reward_from_weights(history, self.weights)

	def check_termination(self, curr_step: StepResult) -> bool:
		return self.termination_fn(curr_step)

	def init(self, env: Env[Any, Any]) -> None:
		if self.init_fn:
			self.init_fn(env)


@final
class GoalSampler:
	def __init__(self, goals: list[Goal], strategy: str = "balanced"):
		self.goals = goals
		self.strategy = strategy
		self.index = 0
		if strategy not in ["single", "balanced", "random"]:
			raise ValueError(f"Unknown strategy: {strategy}")

	def next(self) -> Goal:
		if self.strategy == "single":
			return self.goals[self.index]
		elif self.strategy == "balanced":
			goal = self.goals[self.index]
			self.index = (self.index + 1) % len(self.goals)
			return goal
		elif self.strategy == "random":
			return random.choice(self.goals)
		raise ValueError(f"Tried sampling with undefined strategy: {self.strategy}")

	def peek(self) -> Goal:
		return self.goals[self.index]

	@classmethod
	def single(cls, goal: Goal) -> "GoalSampler":
		return cls([goal], strategy="single")

def get_default_rewards(info: dict[str, Any]) -> dict[str, float]:
	return {
		"healthy": info.get("reward_survive", 0.0),
		"ctrl": -info.get("reward_ctrl", 0.0),
		"contact": -info.get("reward_contact", 0.0),
	}

def reward_forward(history: StepHistory) -> float:
	return history.delta(lambda s: s.info.get("x_position", 0.0), steps=10)

def reward_backward(history: StepHistory) -> float:
	return -history.delta(lambda s: s.info.get("x_position", 0.0), steps=10)

def reward_turn(history: StepHistory) -> float:
	return history.delta(lambda s: s.info.get("y_position", 0.0), steps=10)

def reward_stand(history: StepHistory) -> float:
	curr = history.last()
	prev = history.prev()
	if curr is None or prev is None:
		return 0.0
	dx = history.delta(lambda s: s.info.get("x_position", 0.0), steps=5)
	dy = history.delta(lambda s: s.info.get("y_position", 0.0), steps=5)
	return - (abs(dx) + abs(dy))

def reward_stand_up(history: StepHistory) -> float:
	curr = history.last()
	if curr is None:
		return 0.0
	obs = curr.obs
	z
	# quat_w = obs[1]
	# ang_vel = sum(abs(v) for v in obs[25:28])
	# return 1.2 * z_pos + 0.8 * quat_w - 0.4 * ang_vel
	return z_pos

def reward_jump(history: StepHistory) -> float:
	curr = history.last()
	if curr is None:
		return 0.0
	obs = curr.obs
	z_vel = obs[24]
	z_pos = obs[0]
	z_pos_thresh = 1.2
	z_pos_reward = z_pos if z_pos > z_pos_thresh else 0.0
	return max(0.0, z_vel) + z_pos_reward

def reward_survive(history: StepHistory) -> float:
	curr = history.last()
	if curr is None:
		return 0.0
	return get_default_rewards(curr.info)["healthy"]

def reward_ctrl(history: StepHistory) -> float:
	curr = history.last()
	if curr is None:
		return 0.0
	return get_default_rewards(curr.info)["ctrl"]

def reward_contact(history: StepHistory) -> float:
	curr = history.last()
	if curr is None:
		return 0.0
	return get_default_rewards(curr.info)["contact"]

def jump_reward(history: StepHistory) -> float:
	curr = history.last()
	if curr is None:
		return 0.0

	obs = curr.obs
	z_vel = obs[24]
	z_pos = obs[0]

	z_pos_thresh = 1.2
	z_pos_reward = z_pos if z_pos > z_pos_thresh else 0.0

	default = get_default_rewards(curr.info)

	return max(0.0, z_vel) + z_pos_reward + default["healthy"]

def terminate_never(curr: StepResult) -> bool:
	return False

def init_fallen(env) -> None:
	base_env = get_base_env(env)

	qpos = base_env.init_qpos.copy()
	qvel = base_env.init_qvel.copy()

	qpos[0] = np.random.uniform(0.3, 0.6)
	qpos[1:5] = np.array([
		np.random.uniform(0.6, 1.0),
		np.random.uniform(-0.3, 0.3),
		np.random.uniform(-0.3, 0.3),
		np.random.uniform(-0.3, 0.3),
	])
	for i in range(5, 17):
		qpos[i] += np.random.uniform(-0.1, 0.1)
	qvel += np.random.normal(0, 0.01, size=qvel.shape)

	base_env.data.qpos[:] = qpos
	base_env.data.qvel[:] = qvel


TERMINATION_FN_REGISTRY: dict[str, Callable[[StepResult], bool]] = {
	"default": terminate_default,
	"never": terminate_never,
}

INIT_FN_REGISTRY: dict[str, Callable[[Env[Any, Any]], None]] = {
	"fallen": init_fallen,
}

def get_base_env(env):
	visited = set()
	while True:
		if hasattr(env, "init_qpos") and hasattr(env, "data"):
			return env
		elif hasattr(env, "env") and id(env.env) not in visited:
			visited.add(id(env))
			env = env.env
		elif hasattr(env, "venv") and id(env.venv) not in visited:
			visited.add(id(env))
			env = env.venv
		else:
			raise ValueError("Could not unwrap to base Mujoco environment with init_qpos")
