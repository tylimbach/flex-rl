import os
import yaml
import numpy as np
import logging
import math

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

from rl.envs import GoalSampler, load_goals_from_config
from .env_factory import make_env

log = logging.getLogger(__name__)

def evaluate_snapshot(snapshot_path: str, eval_episodes: int = 10, n_envs: int = 1) -> dict:
	model_path = os.path.join(snapshot_path, "model.zip")
	vecnorm_path = os.path.join(snapshot_path, "vecnormalize.pkl")
	metadata_path = os.path.join(snapshot_path, "metadata.yaml")

	if not all(os.path.exists(p) for p in [model_path, vecnorm_path, metadata_path]):
		raise FileNotFoundError(f"Missing files in snapshot: {snapshot_path}")

	with open(metadata_path) as f:
		metadata = yaml.safe_load(f)
		cfg = metadata["config"]["env"]
		goals = load_goals_from_config(cfg.get("sampling_goals"))

	results = {}
	for goal in goals:
		sampler = GoalSampler.single(goal)

		env_fns = [make_env(cfg["env_id"], sampler) for _ in range(n_envs)]
		env = DummyVecEnv(env_fns)
		env = VecNormalize.load(vecnorm_path, env)
		env.training = False
		env.norm_reward = False

		model = RecurrentPPO.load(model_path, env=env, device="cuda")

		obs = env.reset()
		lstm_states = None  # this is the correct initial value
		episode_starts = np.ones((n_envs,), dtype=bool)

		episodes_needed = eval_episodes
		batch_size = n_envs
		num_batches = math.ceil(episodes_needed / batch_size)

		episode_rewards = []

		for _ in range(num_batches):
			obs = env.reset()
			lstm_states = None
			episode_starts = np.ones((n_envs,), dtype=bool)
			current_rewards = [0.0] * n_envs
			done_flags = [False] * n_envs

			while not all(done_flags):
				action, lstm_states = model.predict(
					obs, state=lstm_states, episode_start=episode_starts, deterministic=True
				)
				obs, reward, done, info = env.step(action)
				episode_starts = done

				for i in range(n_envs):
					if not done_flags[i]:
						current_rewards[i] += reward[i]
						if done[i]:
							done_flags[i] = True

			episode_rewards.extend(current_rewards)

			if len(episode_rewards) >= eval_episodes:
				episode_rewards = episode_rewards[:eval_episodes]
				break

		mean_reward = np.mean(episode_rewards)

		results[goal] = mean_reward
		log.info(f"✅ Goal '{goal}': Mean reward over {eval_episodes} episodes: {mean_reward:.2f}")

	return results
