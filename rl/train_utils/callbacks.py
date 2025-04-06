import os
import logging
from stable_baselines3.common.callbacks import BaseCallback

from .early_stopping import EarlyStopper
from .snapshot import save_full_snapshot
from .evaluation import evaluate_snapshot
from typing import Optional

log = logging.getLogger(__name__)

class SnapshotAndEvalCallback(BaseCallback):
	def __init__(
			self,
			model,
			save_freq,
			env,
			save_dir,
			eval_envs=1,
			eval_episodes=10,
			verbose=0,
			early_stopper: Optional[EarlyStopper]=None):
		super().__init__(verbose)
		self.model = model
		self.save_freq = save_freq
		self.env = env
		self.save_dir = save_dir
		self.eval_episodes = eval_episodes
		self.best_reward = float("-inf")
		self.eval_envs = eval_envs
		self.early_stopper = early_stopper

	def _on_step(self):
		if self.n_calls % self.save_freq == 0:
			snap_dir = os.path.join(self.save_dir, f"{self.num_timesteps}")
			os.makedirs(snap_dir, exist_ok=True)

			save_full_snapshot(self.model, self.env, snap_dir, self.num_timesteps)

			results = evaluate_snapshot(snap_dir, eval_episodes=self.eval_episodes)
			mean_reward = sum(results.values()) / len(results)
			log.info(f"Eval result at {self.num_timesteps}: {mean_reward:.2f}")

			if mean_reward > self.best_reward:
				self.best_reward = mean_reward
				best_dir = os.path.join(self.save_dir, "best")
				os.makedirs(best_dir, exist_ok=True)
				save_full_snapshot(
					self.model,
					self.env,
					best_dir,
					self.num_timesteps,
					eval_reward=mean_reward,
				)
				log.info(f"New best model saved at step {self.num_timesteps} with reward {mean_reward:.2f}")

			if self.early_stopper and self.early_stopper.should_stop(mean_reward):
				log.info("🛑 Early stopping triggered from callback.")
				return False

		return True
