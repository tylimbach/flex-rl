import os
import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import torch

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from envs import GoalSampler, load_goals_from_config
from train_utils.env_factory import make_env, get_unique_experiment_dir
from train_utils.snapshot import save_full_snapshot
from train_utils.metadata import save_metadata
from train_utils.summary import print_summary
from train_utils.callbacks import SnapshotAndEvalCallback

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
	log.info(f"Resolved config:\n{OmegaConf.to_yaml(cfg)}")
	exp_name = cfg.get("experiment_name") or "default_exp"
	workspace = os.path.abspath(cfg.runtime.workspace_dir)
	exp_base = os.path.join(workspace, exp_name)
	BASE_DIR = get_unique_experiment_dir(exp_base)
	CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
	LOG_DIR = os.path.join(BASE_DIR, "log")

	os.makedirs(BASE_DIR, exist_ok=True)
	os.makedirs(CHECKPOINT_DIR, exist_ok=True)
	os.makedirs(LOG_DIR, exist_ok=True)

	with open(os.path.join(BASE_DIR, "config.yaml"), "w") as f:
		OmegaConf.save(config=cfg, f=f.name)
	log.info(f"✅ Saved resolved config to {f.name}")

	N_ENVS = cfg.training.n_envs
	train_goals = load_goals_from_config(cfg.env.sampling_goals)
	goal_sampler = GoalSampler(strategy=cfg.env.sampling_strategy, goals=train_goals)

	normalize_path = None
	parent_path = None
	resumed_at = None

	if cfg.get("model"):
		log.info(f"📦 Loading snapshot from: {cfg.model}")
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
		log.info("🚀 Starting new training run...")
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
			device=cfg.training.device,
			tensorboard_log=LOG_DIR,
		)

	save_metadata(BASE_DIR, OmegaConf.to_container(cfg, resolve=True), parent=parent_path, resumed_at=resumed_at)
	log.info(f"📝 Metadata saved at {BASE_DIR}/metadata.yaml")

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
		save_freq=cfg.evaluation.eval_freq // N_ENVS,
		env=env,
		save_dir=CHECKPOINT_DIR,
		eval_episodes=cfg.evaluation.eval_episodes
	)

	try:
		model.learn(total_timesteps=cfg.training.total_timesteps, callback=[eval_cb])
	except KeyboardInterrupt:
		log.warning("Training interrupted. Saving snapshot...")
		snap_dir = os.path.join(CHECKPOINT_DIR, "manual_interrupt")
		save_full_snapshot(model, env, snap_dir, model.num_timesteps, interrupt=True)
	finally:
		print_summary(BASE_DIR)


if __name__ == "__main__":
	main()
