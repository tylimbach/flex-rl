import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from humanoid_goal_wrapper import HumanoidGoalWrapper
import torch
import os
import argparse
import yaml
import shutil
import datetime


def make_env(id):
	return lambda: Monitor(HumanoidGoalWrapper(gym.make(id)))


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
		yaml.dump(metadata, f)


def update_snapshot_log(save_dir, step, eval_reward=None, interrupt=False):
	log_path = os.path.join(save_dir, "snapshot_log.yaml")
	if os.path.exists(log_path):
		with open(log_path) as f:
			log = yaml.safe_load(f)
	else:
		log = {"snapshots": [], "cumulative_steps": 0}

	entry = {
		"step": step,
		"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
		"interrupt": interrupt
	}
	if eval_reward is not None:
		entry["eval_reward"] = eval_reward

	log["snapshots"].append(entry)
	log["cumulative_steps"] = step

	with open(log_path, "w") as f:
		yaml.dump(log, f)


def save_full_snapshot(model, env, path, step, eval_reward=None, interrupt=False):
	os.makedirs(path, exist_ok=True)
	model.save(os.path.join(path, "model.zip"))
	env.save(os.path.join(path, "vecnormalize.pkl"))
	update_snapshot_log(os.path.dirname(path), step, eval_reward, interrupt)
	print(f"✅ Saved snapshot at {path}")


class SnapshotCallback(BaseCallback):
	def __init__(self, save_freq, env, save_dir, verbose=0):
		super().__init__(verbose)
		self.save_freq = save_freq
		self.env = env
		self.save_dir = save_dir

	def _on_step(self):
		if self.n_calls % self.save_freq == 0:
			snap_dir = os.path.join(self.save_dir, f"{self.num_timesteps}")
			save_full_snapshot(self.model, self.env, snap_dir, self.num_timesteps)
		return True


class BestSnapshotCallback(BaseCallback):
	def __init__(self, model, env, save_dir, verbose=0):
		super().__init__(verbose)
		self.model = model
		self.env = env
		self.save_dir = save_dir

	def _on_step(self):
		return True

	def __call__(self, _locals, _globals):
		snap_dir = os.path.join(self.save_dir, "best")
		save_full_snapshot(self.model, self.env, snap_dir, self.model.num_timesteps)
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


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--name", type=str, required=True, help="Experiment name")
	parser.add_argument("--config", type=str, default="config/humanoid_goal.yaml", help="Path to config file (YAML)")
	parser.add_argument("--model", type=str, help="Optional snapshot folder to resume training")
	parser.add_argument("--lineage", type=str, help="Optional: Print lineage of an experiment path and exit")
	args = parser.parse_args()

	if args.lineage:
		print_lineage(args.lineage)
		exit(0)

	exp_base = f"./workspace/{args.name}"
	BASE_DIR = get_unique_experiment_dir(exp_base)
	CHECKPOINT_DIR = f"{BASE_DIR}/checkpoints"
	TENSORBOARD_DIR = f"{BASE_DIR}/tensorboard"
	EXPERIMENT_CONFIG = f"{BASE_DIR}/config.yaml"

	os.makedirs(BASE_DIR, exist_ok=True)
	os.makedirs(CHECKPOINT_DIR, exist_ok=True)
	os.makedirs(TENSORBOARD_DIR, exist_ok=True)

	if not os.path.exists(args.config):
		raise FileNotFoundError(f"Config file {args.config} not found")

	shutil.copy(args.config, EXPERIMENT_CONFIG)
	print(f"✅ Copied config to {EXPERIMENT_CONFIG}")

	with open(EXPERIMENT_CONFIG) as f:
		cfg = yaml.safe_load(f)

	N_ENVS = cfg["n_envs"]

	if args.model:
		print(f"Loading snapshot from {args.model}")
		dummy_env = SubprocVecEnv([make_env(cfg["env_id"]) for _ in range(N_ENVS)])
		env = VecNormalize.load(os.path.join(args.model, "vecnormalize.pkl"), dummy_env)
		env.training = True
		env.norm_reward = True
		normalize_path = os.path.join(args.model, "vecnormalize.pkl")
		parent_path = args.model
		resumed_at = int(os.path.basename(os.path.normpath(args.model))) if os.path.basename(os.path.normpath(args.model)).isdigit() else None

		model = RecurrentPPO.load(os.path.join(args.model, "model.zip"), env=env, device="cuda")
		model.tensorboard_log = TENSORBOARD_DIR
		print(f"✅ Loaded model and VecNormalize from {args.model}")
	else:
		print("Starting fresh training run...")
		env = SubprocVecEnv([make_env(cfg["env_id"]) for _ in range(N_ENVS)])
		env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=cfg["clip_obs"])
		normalize_path = None
		parent_path = None
		resumed_at = None

		model = RecurrentPPO(
			cfg["policy"],
			env,
			policy_kwargs=dict(
				net_arch=dict(pi=cfg["policy_net"], vf=cfg["value_net"]),
				activation_fn=torch.nn.ReLU
			),
			learning_rate=cfg["learning_rate"],
			n_steps=cfg["n_steps"],
			batch_size=cfg["batch_size"],
			n_epochs=cfg["n_epochs"],
			gamma=cfg["gamma"],
			gae_lambda=cfg["gae_lambda"],
			ent_coef=cfg["ent_coef"],
			clip_range=cfg["clip_range"],
			verbose=1,
			device="cuda",
			tensorboard_log=TENSORBOARD_DIR
		)

	save_metadata(BASE_DIR, cfg, parent=parent_path, resumed_at=resumed_at)
	print(f"📝 Metadata saved at {BASE_DIR}/metadata.yaml")

	snapshot_cb = SnapshotCallback(
		save_freq=cfg["snapshot_freq"],
		env=env,
		save_dir=CHECKPOINT_DIR
	)

	eval_env = DummyVecEnv([make_env(cfg["env_id"])])
	if normalize_path:
		eval_env = VecNormalize.load(normalize_path, eval_env)
	else:
		eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=cfg["clip_obs"])

	eval_env.training = False
	eval_env.norm_reward = False

	best_cb = BestSnapshotCallback(model, env, CHECKPOINT_DIR)

	eval_cb = EvalCallback(
		eval_env,
		callback_on_new_best=best_cb,
		log_path=CHECKPOINT_DIR,
		eval_freq=cfg["eval_freq"] // N_ENVS,
		n_eval_episodes=5,
		deterministic=True,
		render=False
	)

	try:
		model.learn(total_timesteps=cfg["total_timesteps"], callback=[snapshot_cb, eval_cb])
	except KeyboardInterrupt:
		print("Training interrupted. Saving snapshot...")
		snap_dir = os.path.join(CHECKPOINT_DIR, "manual_interrupt")
		save_full_snapshot(model, env, snap_dir, model.num_timesteps, interrupt=True)
	finally:
		print_summary(BASE_DIR)
