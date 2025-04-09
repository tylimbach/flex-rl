import os
from stable_baselines3.common.base_class import BaseAlgorithm
import yaml
import numpy as np
import logging
import math

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO

from ..envs import GoalSampler, Goal
from .config import EnvConfig
from .env_factory import make_env

log = logging.getLogger(__name__)

def evaluate_model_on_goals(
	model: BaseAlgorithm,
	trained_vecnormalize_env: VecNormalize,
	env_cfg: EnvConfig,
	goals: list[Goal],
	eval_episodes: int,
	n_envs: int = 1
) -> dict[Goal, float]:
	results = {}
	for goal in goals:
		sampler = GoalSampler.single(goal)
		env_fns = [make_env(env_cfg["env_id"], sampler) for _ in range(n_envs)]
		eval_env = DummyVecEnv(env_fns)
		eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, clip_obs=trained_vecnormalize_env.clip_obs)

		# Copy normalization statistics from training env
		eval_env.obs_rms = trained_vecnormalize_env.obs_rms
		eval_env.ret_rms = trained_vecnormalize_env.ret_rms
		eval_env.training = False
		eval_env.norm_reward = False

		episode_rewards = []
		for _ in range(math.ceil(eval_episodes / n_envs)):
			obs = eval_env.reset()
			lstm_states = None
			episode_starts = np.ones((n_envs,), dtype=bool)
			curr_rewards = [0.0] * n_envs
			done_flags = [False] * n_envs

			while not all(done_flags):
				action, lstm_states = model.predict(
					obs, state=lstm_states, episode_start=episode_starts, deterministic=True
				)
				obs, reward, done, _ = eval_env.step(action)
				episode_starts = done
				for i in range(n_envs):
					if not done_flags[i]:
						curr_rewards[i] += reward[i]
						if done[i]:
							done_flags[i] = True
			episode_rewards.extend(curr_rewards)
		episode_rewards = episode_rewards[:eval_episodes]
		mean_reward = np.mean(episode_rewards)
		log.info(f"✅ Goal '{goal}': Mean reward over {eval_episodes} episodes: {mean_reward:.2f}")
		results[goal] = mean_reward
	return results


# def evaluate_snapshot(snapshot_path: str, eval_episodes: int = 10, n_envs: int = 1) -> dict[Goal, float]:
# 	model_path = os.path.join(snapshot_path, "model.zip")
# 	vecnorm_path = os.path.join(snapshot_path, "vecnormalize.pkl")
# 	metadata_path = os.path.join(snapshot_path, "metadata.yaml")
#
# 	if not all(os.path.exists(p) for p in [model_path, vecnorm_path, metadata_path]):
# 		raise FileNotFoundError(f"Missing files in snapshot: {snapshot_path}")
#
# 	with open(metadata_path) as f:
# 		metadata = yaml.safe_load(f)
# 		cfg = metadata["config"]["env"]
# 		goals = load_goals_from_config(cfg.get("sampling_goals"))
#
# 	model = RecurrentPPO.load(model_path, device="cuda")
#
# 	return evaluate_model_on_goals(model, vecnorm_path, metadata["config"]["env"], goals, eval_episodes, n_envs)
