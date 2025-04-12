from omegaconf import OmegaConf
from ..envs.goal import Goal, RewardWeights, TERMINATION_FN_REGISTRY, INIT_FN_REGISTRY, terminate_default
from .config import SamplingGoalConfig

def goal_from_cfg(cfg: SamplingGoalConfig) -> Goal:
	weights = RewardWeights(**OmegaConf.to_container(cfg.reward_weights, resolve=True))
	term_fn = TERMINATION_FN_REGISTRY[cfg.termination_fn] if cfg.get("termination_fn") else terminate_default
	init_fn = INIT_FN_REGISTRY[cfg.init_fn] if cfg.get("init_fn") else None
	return Goal(weights=weights, weight=cfg.weight, termination_fn=term_fn, init_fn=init_fn)
