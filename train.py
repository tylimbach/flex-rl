import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from humanoid_goal_wrapper import HumanoidGoalWrapper
import torch
import os
import argparse

N_ENVS = 8
LOG_DIR = "./workspace_humanoid"
CHECKPOINT_DIR = f"{LOG_DIR}/checkpoints"
TENSORBOARD_DIR = f"{LOG_DIR}/tensorboard"

def make_env():
	return lambda: Monitor(HumanoidGoalWrapper(gym.make("Humanoid-v5")))

def save_full_snapshot(model, env, label):
	snap_dir = os.path.join(CHECKPOINT_DIR, str(label))
	os.makedirs(snap_dir, exist_ok=True)
	model.save(os.path.join(snap_dir, "model.zip"))
	env.save(os.path.join(snap_dir, "vecnormalize.pkl"))
	print(f"✅ Saved snapshot at {snap_dir}")

class SnapshotCallback(BaseCallback):
	def __init__(self, save_freq, env, verbose=0):
		super().__init__(verbose)
		self.save_freq = save_freq
		self.env = env

	def _on_step(self):
		if self.n_calls % self.save_freq == 0:
			save_full_snapshot(self.model, self.env, self.num_timesteps)
		return True

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, help="Path to specific snapshot folder to load")
	args = parser.parse_args()

	if args.model:
		print(f"Loading snapshot from {args.model}")
		dummy_env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
		env = VecNormalize.load(os.path.join(args.model, "vecnormalize.pkl"), dummy_env)
		env.training = True
		env.norm_reward = True
		normalize_path = os.path.join(args.model, "vecnormalize.pkl")
	else:
		print("Starting fresh training run...")
		env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
		env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
		normalize_path = None

	policy_kwargs = dict(
		net_arch=dict(pi=[256, 256], vf=[256, 256]),
		activation_fn=torch.nn.ReLU
	)

	model = RecurrentPPO(
		"MlpLstmPolicy", env,
		policy_kwargs=policy_kwargs,
		learning_rate=3e-4,
		n_steps=2048,
		batch_size=4096,
		n_epochs=10,
		gamma=0.99,
		gae_lambda=0.95,
		ent_coef=0.001,
		clip_range=0.2,
		verbose=1,
		device="cuda"
	)

	if args.model:
		model = RecurrentPPO.load(os.path.join(args.model, "model.zip"), env=env, device="cuda")
		print(f"✅ Loaded model and VecNormalize from {args.model}")

	snapshot_cb = SnapshotCallback(save_freq=100_000, env=env)

	# Eval env with correct VecNormalize
	eval_env = DummyVecEnv([make_env()])
	if normalize_path:
		eval_env = VecNormalize.load(normalize_path, eval_env)
	else:
		eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=10.)

	eval_env.training = False
	eval_env.norm_reward = False

	eval_cb = EvalCallback(
		eval_env,
		best_model_save_path=os.path.join(CHECKPOINT_DIR, "best"),
		log_path=CHECKPOINT_DIR,
		eval_freq=100_000,
		n_eval_episodes=5,
		deterministic=True,
		render=False
	)

	try:
		model.learn(total_timesteps=100_000_000, callback=[snapshot_cb, eval_cb])
	except KeyboardInterrupt:
		print("Training interrupted. Saving snapshot...")
		save_full_snapshot(model, env, "manual_interrupt")
