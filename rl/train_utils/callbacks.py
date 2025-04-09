import logging
import os
from typing import override

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from ..envs import Goal
from .config import EnvConfig, EvaluationConfig
from .early_stopping import EarlyStopper
from .evaluation import evaluate_model_on_goals
from .training import TrainingContext

log = logging.getLogger(__name__)

class SnapshotAndEvalCallback(BaseCallback):
	def __init__(
		self,
		ctx: TrainingContext,
		env_cfg: EnvConfig,
		eval_cfg: EvaluationConfig,
		eval_envs: int,
		eval_episodes: int,
		early_stopper: EarlyStopper | None = None,
		verbose: int = 0
	):
		super().__init__(verbose)
		self.ctx = ctx
		self.env_cfg = env_cfg
		self.eval_cfg = eval_cfg
		self.eval_envs = eval_envs
		self.eval_episodes = eval_episodes
		self.best_reward = -float("inf")
		self.early_stopper = early_stopper

	@override
	def _on_step(self) -> bool:
		if self.n_calls % (self.eval_cfg.eval_freq // self.eval_envs) != 0:
			return True

		reward_by_goal = evaluate_model_on_goals(
			model=self.ctx.model,
			trained_vecnormalize_env=self.ctx.env,
			env_cfg=self.env_cfg,
			goals=self.ctx.goal_sampler.goals,
			eval_episodes=self.eval_episodes,
			n_envs=self.eval_envs
		)
		avg_reward: float = np.mean(list(reward_by_goal.values()))

		if avg_reward > self.best_reward:
			self.best_reward = avg_reward
			step_dir = os.path.join(self.ctx.checkpoint_dir, f"step_{self.num_timesteps}")
			os.makedirs(step_dir, exist_ok=True)
			self.ctx.model.save(os.path.join(step_dir, "model.zip"))
			self.ctx.env.save(os.path.join(step_dir, "vecnormalize.pkl"))
			log.info(f"📸 Saved new best snapshot with avg_reward={avg_reward:.2f} at step {self.num_timesteps}")

		if self.early_stopper and self.early_stopper.should_stop(avg_reward):
			log.warning("⏹️ Early stopping triggered.")
			return False
		return True
