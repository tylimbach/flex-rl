import logging
import os
import shutil

import hydra
import mlflow
import mlflow.utils.mlflow_tags
import torch
from omegaconf import OmegaConf
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .train_utils.training import TrainingContext


from .envs import GoalSampler, Goal
from .train_utils import (
	EarlyStopper,
	SnapshotAndEvalCallback,
	get_unique_experiment_dir,
	make_env,
	print_summary,
	save_full_snapshot,
	save_metadata,
	TrainingResult,
	TopLevelConfig,
	goal_from_cfg
)

log: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: TopLevelConfig) -> None:
	log.info(f"Resolved config:\n{OmegaConf.to_yaml(cfg)}")

	exp_name = cfg.experiment_name or "default_exp"
	workspace = os.path.abspath(cfg.runtime.workspace_path)
	exp_base = os.path.join(workspace, exp_name)
	workspace_dir = get_unique_experiment_dir(exp_base)
	os.makedirs(workspace_dir, exist_ok=True)

	with open(os.path.join(workspace_dir, "config.yaml"), "w") as f:
		OmegaConf.save(config=OmegaConf.structured(cfg), f=f.name)
	log.info(f"✅ Saved resolved config to {f.name}")

	result = train(cfg, workspace_dir)
	record(cfg, result)

def record(cfg: TopLevelConfig, result: TrainingResult) -> None:
	experiment_name = cfg.experiment_name or "default"
	mlflow.set_tracking_uri(cfg.runtime.mlflow_uri)
	experiment = mlflow.set_experiment(experiment_name)

	with mlflow.start_run():
		# mlflow.log_params(cfg)
		mlflow.set_tags({
			mlflow.utils.mlflow_tags.MLFLOW_RUN_NAME: experiment_name,
			"policy": cfg.training.policy,
			"env": cfg.env.env_id,
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

def train(cfg: TopLevelConfig, workspace_dir: str) -> TrainingResult:
	checkpoint_dir = os.path.join(workspace_dir, "checkpoints")
	log_dir = os.path.join(workspace_dir, "log")

	os.makedirs(workspace_dir, exist_ok=True)
	os.makedirs(checkpoint_dir, exist_ok=True)
	os.makedirs(log_dir, exist_ok=True)

	n_envs: int = cfg.training.n_envs
	train_goals = [goal_from_cfg(x) for x in cfg.env.sampling_goals]
	goal_sampler = GoalSampler(strategy=cfg.env.sampling_strategy, goals=train_goals)

	normalize_path = None
	parent_path = None
	resumed_at = None

	if cfg.parent_model:
		log.info(f"📦 Loading snapshot from: {cfg.parent_model}")
		dummy_env = DummyVecEnv([make_env(cfg.env.env_id, goal_sampler) for _ in range(n_envs)])
		normalize_path = os.path.join(cfg.parent_model, "vecnormalize.pkl")
		env = VecNormalize.load(normalize_path, dummy_env)
		env.training = True
		env.norm_reward = True
		parent_path = cfg.parent_model
		try:
			resumed_at = int(os.path.basename(os.path.normpath(cfg.parent_model)))
		except ValueError:
			pass

		model = RecurrentPPO.load(os.path.join(cfg.parent_model, "model.zip"), env=env, device="cuda")
		model.tensorboard_log = log_dir
	else:
		log.info("🚀 Starting new training run...")
		env = DummyVecEnv([make_env(cfg.env.env_id, goal_sampler) for _ in range(n_envs)])
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

	raw_cfg = OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True)
	save_metadata(workspace_dir, raw_cfg, parent=parent_path, resumed_at=resumed_at)
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
	ctx = TrainingContext(cfg, model, env, goal_sampler, checkpoint_dir, workspace_dir)
	eval_cb = SnapshotAndEvalCallback(
		ctx=ctx,
		env_cfg=cfg.env,
		eval_cfg=cfg.evaluation,
		eval_envs=n_envs,
		eval_episodes=cfg.evaluation.eval_episodes,
		early_stopper=early_stopper
	)

	try:
		model.learn(total_timesteps=cfg.training.total_timesteps, callback=[eval_cb])
	except KeyboardInterrupt:
		log.warning("Training interrupted manually. Saving snapshot...")
		snap_dir = os.path.join(checkpoint_dir, "manual_interrupt")
		save_full_snapshot(model, env, snap_dir)
	except Exception as e:
		log.warning(f"Training interrupted due to error {e}. Saving snapshot...")
		snap_dir = os.path.join(checkpoint_dir, "crash")
		save_full_snapshot(model, env, snap_dir)
	finally:
		print_summary(workspace_dir)
		return TrainingResult(eval_cb.best_reward, eval_cb.ctx.model, env, cfg.parent_model)

if __name__ == "__main__":
	main()
