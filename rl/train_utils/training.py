from dataclasses import dataclass
from typing import Any
from omegaconf import DictConfig
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecNormalize

from envs.goal import GoalSampler


@dataclass
class TrainingContext:
	cfg: DictConfig
	model: RecurrentPPO
	env: VecNormalize
	goal_sampler: GoalSampler
	checkpoint_dir: str
	workspace_dir: str

@dataclass
class TrainingResult:
	final_reward: float
	model: BaseAlgorithm
	env: Any
	orig_model_path: str | None = None

