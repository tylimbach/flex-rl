import os
import logging
from stable_baselines3.common.callbacks import BaseCallback

from .early_stopping import EarlyStopper
from .snapshot import save_full_snapshot
from .evaluation import evaluate_model_on_goals, evaluate_snapshot
from typing import Optional

log = logging.getLogger(__name__)

class SnapshotAndEvalCallback(BaseCallback):
	def __init__(self, ctx: TrainingContext, eval_envs: int, eval_episodes: int, early_stopper: Optional[EarlyStopper] = None, verbose=0):
		super().__init__(verbose)
		self.ctx = ctx
		self.eval_envs = eval_envs
		self.eval_episodes = eval_episodes
		self.best_reward = -float("inf")
		self.early_stopper = early_stopper

	def _on_step(self) -> bool:
		if self.n_calls % (self.ctx.cfg.evaluation.eval_freq // self.ctx.cfg.training.n_envs) != 0:
			return True

		goals = ctx.
		goals = load_goals_from_config(self.ctx.cfg.env.sampling_goals)
		reward_by_goal = evaluate_model_on_goals(
			model=self.ctx.model,
			vecnormalize_path=os.path.join(self.ctx.checkpoint_dir, "vecnormalize.pkl"),
			env_cfg=OmegaConf.to_container(self.ctx.cfg.env, resolve=True),
			goals=goals,
			eval_episodes=self.eval_episodes,
			n_envs=self.eval_envs
		)
		avg_reward = np.mean(list(reward_by_goal.values()))

		if avg_reward > self.best_reward:
			self.best_reward = avg_reward
			step_dir = os.path.join(self.ctx.checkpoint_dir, f"step_{self.num_timesteps}")
			os.makedirs(step_dir, exist_ok=True)
			self.ctx.model.save(os.path.join(step_dir, "model.zip"))
			self.ctx.env.save(os.path.join(step_dir, "vecnormalize.pkl"))
			log.info(f"📸 Saved new best snapshot with avg_reward={avg_reward:.2f} at step {self.num_timesteps}")

		if self.early_stopper and not self.early_stopper.check(avg_reward):
			log.warning("⏹️ Early stopping triggered.")
			return False
		return True
