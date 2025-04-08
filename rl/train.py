import logging
import os
import shutil
from dataclasses import dataclass
from typing import Any

import hydra
import mlflow
import mlflow.utils.mlflow_tags
import torch
from omegaconf import DictConfig, OmegaConf
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.envs import GoalSampler, load_goals_from_config
from rl.train_utils import (
	EarlyStopper,
	SnapshotAndEvalCallback,
	get_unique_experiment_dir,
	make_env,
	print_summary,
	save_full_snapshot,
	save_metadata,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
	log.info(f"Resolved config:\n{OmegaConf.to_yaml(cfg)}")

	exp_name = cfg.get("experiment_name") or "default_exp"
	workspace = os.path.abspath(cfg.runtime.workspace_dir)
	exp_base = os.path.join(workspace, exp_name)
	workspace_dir = get_unique_experiment_dir(exp_base)
	os.makedirs(workspace_dir, exist_ok=True)

	with open(os.path.join(workspace_dir, "config.yaml"), "w") as f:
		OmegaConf.save(config=cfg, f=f.name)
	log.info(f"✅ Saved resolved config to {f.name}")

	result = train(cfg, workspace_dir)
	record(cfg, result)


def record(cfg, result: TrainingResult):
	experiment_name = cfg.get("experiment_name", "default")
	mlflow.set_tracking_uri(cfg.get("mlflow_uri", "http://localhost:5000"))
	mlflow.set_experiment(experiment_name)

	with mlflow.start_run():
		mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
		mlflow.set_tags({
			mlflow.utils.mlflow_tags.MLFLOW_RUN_NAME: experiment_name,
			"policy": cfg.policy,
			"env": cfg.env_id,
		})
		mlflow.log_metric("final_reward", result.final_reward)

		snapshot_dir = "outputs/best_model"
		os.makedirs(snapshot_dir, exist_ok=True)
		result.model.save(os.path.join(snapshot_dir, "model.zip"))
		result.env.save(os.path.join(snapshot_dir, "vecnormalize.pkl"))

		if result.orig_model_path:
			shutil.copy(result.orig_model_path, os.path.join(snapshot_dir, "origination.txt"))
			mlflow.set_tag("resumed_from", result.orig_model_path)

		mlflow.log_artifacts(snapshot_dir, artifact_path="best_model")


def train(cfg: DictConfig, workspace_dir: str) -> TrainingResult:
	checkpoint_dir = os.path.join(workspace_dir, "checkpoints")
	log_dir = os.path.join(workspace_dir, "log")

	os.makedirs(workspace_dir, exist_ok=True)
	os.makedirs(checkpoint_dir, exist_ok=True)
	os.makedirs(log_dir, exist_ok=True)

	N_ENVS = cfg.training.n_envs
	train_goals = load_goals_from_config(cfg.env.sampling_goals)
	goal_sampler = GoalSampler(strategy=cfg.env.sampling_strategy, goals=train_goals)

	normalize_path = None
	parent_path = None
	resumed_at = None

	if cfg.get("parent_model"):
		log.info(f"📦 Loading snapshot from: {cfg.model}")
		dummy_env = DummyVecEnv([make_env(cfg.env.env_id, goal_sampler) for _ in range(N_ENVS)])
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
		model.tensorboard_log = log_dir
	else:
		log.info("🚀 Starting new training run...")
		env = DummyVecEnv([make_env(cfg.env.env_id, goal_sampler) for _ in range(N_ENVS)])
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
			verbose=0,
			device=cfg.training.device,
			tensorboard_log=log_dir,
		)

	save_metadata(workspace_dir, OmegaConf.to_container(cfg, resolve=True), parent=parent_path, resumed_at=resumed_at)
	log.info(f"📝 Metadata saved at {workspace_dir}/metadata.yaml")

	eval_env = DummyVecEnv([
		make_env(cfg.env.env_id, GoalSampler(strategy="balanced", goals=[goal])) for goal in train_goals
	])

	if normalize_path:
		eval_env = VecNormalize.load(normalize_path, eval_env)
	else:
		eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=cfg.training.clip_obs)

	eval_env.training = False
	eval_env.norm_reward = False

	early_stopper = EarlyStopper(cfg.training.early_stopper)
	eval_cb = SnapshotAndEvalCallback(
		model=model,
		save_freq=cfg.evaluation.eval_freq // N_ENVS,
		env=env,
		eval_envs=N_ENVS,
		save_dir=checkpoint_dir,
		eval_episodes=cfg.evaluation.eval_episodes,
		early_stopper=early_stopper
	)

	try:
		model.learn(total_timesteps=cfg.training.total_timesteps, callback=[eval_cb])
	except KeyboardInterrupt:
		log.warning("Training interrupted. Saving snapshot...")
		snap_dir = os.path.join(checkpoint_dir, "manual_interrupt")
		save_full_snapshot(model, env, snap_dir, model.num_timesteps, interrupt=True)
	finally:
		print_summary(workspace_dir)
		return TrainingResult(eval_cb.best_reward, eval_cb.model, cfg.env.env_id, cfg.parent_model)


if __name__ == "__main__":
	main()
