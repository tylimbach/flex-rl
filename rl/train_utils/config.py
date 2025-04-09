from dataclasses import dataclass
from typing import final


@dataclass
class SamplingGoal:
	name: str
	weight: float


@dataclass
class EnvConfig:
	env_id: str
	sampling_strategy: str
	sampling_goals: list[SamplingGoal]


@dataclass
class EarlyStopperConfig:
	enabled: bool
	patience: int
	min_delta: float
	reward_key: str
	reward_threshold: float | None
	fail_on_nan: bool
	max_runtime_minutes: int


@dataclass
class TrainingConfig:
	policy: str
	n_envs: int
	total_timesteps: int
	learning_rate: float
	batch_size: int
	n_steps: int
	n_epochs: int
	gamma: float
	gae_lambda: float
	ent_coef: float
	clip_range: float
	policy_net: list[int]
	value_net: list[int]
	clip_obs: float
	device: str
	early_stopper: EarlyStopperConfig


@dataclass
class EvaluationConfig:
	eval_freq: int
	eval_episodes: int


@dataclass
class RuntimeConfig:
	workspace_path: str
	mlflow_uri: str


@dataclass
class TopLevelConfig:
	experiment_name: str | None
	parent_model: str | None
	env: EnvConfig
	training: TrainingConfig
	evaluation: EvaluationConfig
	runtime: RuntimeConfig
