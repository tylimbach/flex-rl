import argparse
import datetime
import os
import shutil
import logging

import gymnasium as gym
import torch
import yaml
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from humanoid_goal_wrapper import HumanoidGoalWrapper
from goal import GoalSampler, load_goals_from_config
from evaluate_snapshot import evaluate_snapshot

import hydra
from omegaconf import DictConfig, OmegaConf


def make_env(env_id, goal_sampler: GoalSampler):
	def _init():
		env = Monitor(HumanoidGoalWrapper(gym.make(env_id), goal_sampler))
		return env

	return _init


def get_unique_experiment_dir(base_path):
	if not os.path.exists(base_path):
		return base_path
	i = 1
	while True:
		new_path = f"{base_path}_{i}"
		if not os.path.exists(new_path):
			return new_path
		i += 1


def save_metadata(path, cfg, parent=None, resumed_at=None):
	metadata = {
		"experiment_name": os.path.basename(path),
		"created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
		"config": cfg,
		"parent": parent,
		"resumed_at_step": resumed_at,
	}
	with open(os.path.join(path, "metadata.yaml"), "w") as f:
		yaml.safe_dump(metadata, f)


def update_snapshot_log(save_dir, step, eval_reward=None, interrupt=False):
	log_path = os.path.join(save_dir, "snapshot_log.yaml")
	if os.path.exists(log_path):
		with open(log_path) as f:
			log = yaml.safe_load(f)
	else:
		log = {"snapshots": [], "cumulative_steps": 0}

	entry = {
		"step": int(step),
		"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
		"interrupt": bool(interrupt),
	}
	if eval_reward is not None:
		entry["eval_reward"] = float(eval_reward)

	log["snapshots"].append(entry)
	log["cumulative_steps"] = int(step)

	with open(log_path, "w") as f:
		yaml.safe_dump(log, f)


def save_full_snapshot(model, env, path, step, eval_reward=None, interrupt=False):
	os.makedirs(path, exist_ok=True)
	model.save(os.path.join(path, "model.zip"))
	env.save(os.path.join(path, "vecnormalize.pkl"))
	experiment_dir = os.path.dirname(os.path.dirname(path))
	metadata_src = os.path.join(experiment_dir, "metadata.yaml")
	metadata_dst = os.path.join(path, "metadata.yaml")
	if os.path.exists(metadata_src):
		shutil.copy(metadata_src, metadata_dst)

	update_snapshot_log(os.path.dirname(path), step, eval_reward, interrupt)
	print(f"✅ Saved snapshot at {path}")


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


def print_lineage(path):
	print(f"\n📄 Lineage for: {path}")
	while path:
		meta_path = os.path.join(path, "metadata.yaml")
		if not os.path.exists(meta_path):
			print(f"❗ No metadata found at {path}")
			break
		with open(meta_path) as f:
			meta = yaml.safe_load(f)
		print(f"\n➡️  Experiment: {meta['experiment_name']}")
		print(f"   Created: {meta['created']}")
		if meta.get("resumed_at_step"):
			print(f"   Resumed at step: {meta['resumed_at_step']}")
		path = meta.get("parent")


def print_summary(base_dir):
	snapshot_log = os.path.join(base_dir, "checkpoints", "snapshot_log.yaml")
	if not os.path.exists(snapshot_log):
		print("❗ No snapshot log found.")
		return

	with open(snapshot_log) as f:
		log = yaml.safe_load(f)

	print("\n📊 Training Summary:")
	print(f"Total Timesteps: {log.get('cumulative_steps', 0)}")

	best_dir = os.path.join(base_dir, "checkpoints", "best")
	if os.path.exists(best_dir):
		print(f"Best Model Saved At: {best_dir}")
	else:
		print("No best model saved.")

	print(f"Full Snapshot Log: {snapshot_log}")


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
	exp_name = cfg.get("experiment_name") or "default_exp"
	workspace = cfg.workspace.base_dir
	exp_base = os.path.join(workspace, exp_name)
	BASE_DIR = get_unique_experiment_dir(exp_base)
	CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
	LOG_DIR = os.path.join(BASE_DIR, "log")

	os.makedirs(BASE_DIR, exist_ok=True)
	os.makedirs(CHECKPOINT_DIR, exist_ok=True)
	os.makedirs(LOG_DIR, exist_ok=True)

	# Save resolved config for reproducibility
	with open(os.path.join(BASE_DIR, "config.yaml"), "w") as f:
		OmegaConf.save(config=cfg, f=f.name)
	log.info(f"🔧 Config:\n{OmegaConf.to_yaml(cfg)}")

	N_ENVS = cfg.training.n_envs
	train_goals = load_goals_from_config(cfg.env.sampling_goals)
	goal_sampler = GoalSampler(strategy=cfg.env.sampling_strategy, goals=train_goals)

	normalize_path = None
	parent_path = None
	resumed_at = None

	if cfg.get("model"):
		print(f"📦 Loading snapshot from: {cfg.model}")
		dummy_env = SubprocVecEnv([make_env(cfg.env.env_id, goal_sampler) for _ in range(N_ENVS)])
		normalize_path = os.path.join(cfg.model, "vecnormalize.pkl")
		env = VecNormalize.load(normalize_path, dummy_env)
		env.training = True
		env.norm_reward = True
		parent_path = cfg.model
		try:
			resumed_at = int(os.path.basename(os.path.normpath(cfg.model)))
		except ValueError:
			pass

		model = RecurrentPPO.load(os.path.join(cfg.model, "model.zip"), env=env, device="cuda")
		model.tensorboard_log = LOG_DIR
	else:
		print("🚀 Starting new training run...")
		env = SubprocVecEnv([make_env(cfg.env.env_id, goal_sampler) for _ in range(N_ENVS)])
		env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=cfg.training.clip_obs)

		model = RecurrentPPO(
			cfg.training.policy,
			env,
			policy_kwargs=dict(
				net_arch=dict(pi=cfg.training.policy_net, vf=cfg.training.value_net),
				activation_fn=torch.nn.ReLU,
			),
			learning_rate=cfg.training.learning_rate,
			n_steps=cfg.training.n_steps,
			batch_size=cfg.training.batch_size,
			n_epochs=cfg.training.n_epochs,
			gamma=cfg.training.gamma,
			gae_lambda=cfg.training.gae_lambda,
			ent_coef=cfg.training.ent_coef,
			clip_range=cfg.training.clip_range,
			verbose=1,
			device="cuda",
			tensorboard_log=LOG_DIR,
		)

	save_metadata(BASE_DIR, OmegaConf.to_container(cfg, resolve=True), parent=parent_path, resumed_at=resumed_at)
	print(f"📝 Metadata saved at {BASE_DIR}/metadata.yaml")

	eval_env = DummyVecEnv([
		make_env(cfg.env.env_id, GoalSampler(strategy="balanced", goals=[goal])) for goal in train_goals
	])

	if normalize_path:
		eval_env = VecNormalize.load(normalize_path, eval_env)
	else:
		eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=cfg.training.clip_obs)

	eval_env.training = False
	eval_env.norm_reward = False

	eval_cb = SnapshotAndEvalCallback(
		model=model,
		save_freq=cfg.training.eval_freq // N_ENVS,
		env=env,
		save_dir=CHECKPOINT_DIR,
		eval_episodes=cfg.training.eval_episodes
	)

	try:
		model.learn(total_timesteps=cfg.training.total_timesteps, callback=[eval_cb])
	except KeyboardInterrupt:
		print("Training interrupted. Saving snapshot...")
		snap_dir = os.path.join(CHECKPOINT_DIR, "manual_interrupt")
		save_full_snapshot(model, env, snap_dir, model.num_timesteps, interrupt=True)
	finally:
		print_summary(BASE_DIR)


if __name__ == "__main__":
	main()
