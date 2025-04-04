import os
import yaml
import numpy as np
import logging

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

from envs import GoalSampler, load_goals_from_config
from train_utils.env_factory import make_env

log = logging.getLogger(__name__)

def evaluate_snapshot(snapshot_path: str, eval_episodes: int = 10) -> dict:
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
		env = DummyVecEnv([make_env(cfg["env_id"], sampler)])
		env = VecNormalize.load(vecnorm_path, env)
		env.training = False
		env.norm_reward = False

		model = RecurrentPPO.load(model_path, env=env, device="cuda")

		all_rewards = []

		for _ in range(eval_episodes):
			obs = env.reset()
			lstm_states = None
			episode_starts = np.ones((env.num_envs,), dtype=bool)
			total_reward = 0.0
			done = False

			while not done:
				action, lstm_states = model.predict(
					np.array(obs),
					state=lstm_states,
					episode_start=episode_starts,
					deterministic=True
				)
				obs, reward, done, info = env.step(action)
				total_reward += reward[0]
				episode_starts = done

			all_rewards.append(total_reward)

		mean_reward = np.mean(all_rewards)
		results[goal] = mean_reward
		log.info(f"✅ Goal '{goal}': Mean reward over {eval_episodes} episodes: {mean_reward:.2f}")

	return results
