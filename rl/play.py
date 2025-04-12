import logging
import os

import hydra
import numpy as np
from omegaconf import OmegaConf
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .envs import GoalSampler
from .train_utils import TopLevelConfig, make_env, save_media, goal_from_cfg

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@hydra.main(config_path="configs", config_name="play", version_base="1.3")
def main(cfg):
	model_path = os.path.join(cfg.model_dir, "model.zip")
	vecnorm_path = os.path.join(cfg.model_dir, "vecnormalize.pkl")
	config_path = os.path.join(cfg.model_dir, "../config.yaml")

	log.info(f"🔍 Loading model from: {model_path}")
	log.info(f"🔍 Loading normalization from: {vecnorm_path}")
	log.info(f"🔍 Loading config from: {config_path}")

	raw = OmegaConf.load(config_path)
	env_cfg = OmegaConf.structured(TopLevelConfig(**raw))

	goals = [goal_from_cfg(x) for x in env_cfg.env.sampling_goals]
	goal_sampler = GoalSampler(strategy="balanced", goals=goals)
	render_mode = "rgb_array" if cfg.mp4_path or cfg.gif_path else "human"

	env = DummyVecEnv([make_env(env_cfg.env.env_id, goal_sampler, render_mode)])
	env = VecNormalize.load(vecnorm_path, env)
	env.training = False
	env.norm_reward = False

	model = RecurrentPPO.load(model_path, env=env, device=cfg.device)
	obs = env.reset()

	if cfg.mp4_path or cfg.gif_path:
		output_path = cfg.mp4_path or cfg.gif_path
		log.info(f"🎥 Recording to {output_path}")
		lstm_states = None
		episode_starts = np.ones((env.num_envs,), dtype=bool)
		frames, total_reward, done, info = [], 0, False, [{}]

		while not done:
			action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
			obs, reward, done, info = env.step(action)
			total_reward += reward[0]
			episode_starts = done
			frames.append(env.envs[0].render())

		log.info(f"🎯 Episode reward: {total_reward:.2f} | Goal: {info[0].get('goal')}")
		save_media(frames, output_path)
	else:
		log.info("🎮 Starting live play session...")
		while True:
			try:
				lstm_states = None
				episode_starts = np.ones((env.num_envs,), dtype=bool)
				total_reward, done, info = 0, False, [{}]

				while not done:
					action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
					obs, reward, done, info = env.step(action)
					total_reward += reward[0]
					env.render()
					episode_starts = done

				log.info(f"🎯 Episode reward: {total_reward:.2f} | Goal: {info[0].get('goal')}")
			except Exception:
				env.reset()
				env.close()


if __name__ == "__main__":
	main()
