import os
from stable_baselines3.common.callbacks import BaseCallback
from .snapshot import save_full_snapshot
from .evaluation import evaluate_snapshot


class SnapshotAndEvalCallback(BaseCallback):
	def __init__(self, model, save_freq, env, save_dir, eval_episodes=10, verbose=0):
		super().__init__(verbose)
		self.model = model
		self.save_freq = save_freq
		self.env = env
		self.save_dir = save_dir
		self.eval_episodes = eval_episodes
		self.best_reward = float("-inf")

	def _on_step(self):
		if self.n_calls % self.save_freq == 0:
			snap_dir = os.path.join(self.save_dir, f"{self.num_timesteps}")
			os.makedirs(snap_dir, exist_ok=True)

			save_full_snapshot(self.model, self.env, snap_dir, self.num_timesteps)

			results = evaluate_snapshot(snap_dir, eval_episodes=self.eval_episodes)
			mean_reward = sum(results.values()) / len(results)
			print(f"Eval result at {self.num_timesteps}: {mean_reward:.2f}")

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
				print(
					f"New best model saved at step {self.num_timesteps} with reward {mean_reward:.2f}"
				)

		return True
