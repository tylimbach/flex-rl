import argparse
import logging
import os

import numpy as np
import yaml
from omegaconf import OmegaConf
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from rl.envs import GoalSampler, load_goals_from_config
from rl.train_utils import make_env, save_media

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--model", type=str, required=True)
	parser.add_argument("--video", type=str, help="Save episode to .mp4")
	parser.add_argument("--gif", type=str, help="Save episode to .gif")
	args = parser.parse_args()

	model_path = os.path.join(args.model, "model.zip")
	vecnorm_path = os.path.join(args.model, "vecnormalize.pkl")
	config_path = os.path.join(args.model, "metadata.yaml")

	log.info(f"🔍 Loading model from: {model_path}")
	log.info(f"🔍 Loading normalization from: {vecnorm_path}")
	log.info(f"🔍 Loading config from: {config_path}")

	with open(config_path) as f:
		cfg = OmegaConf.create(yaml.safe_load(f)["config"])
		goals = load_goals_from_config(cfg.env.sampling_goals)

	goal_sampler = GoalSampler(strategy="balanced", goals=goals)
	render_mode = "rgb_array" if args.video or args.gif else "human"
	env = DummyVecEnv([make_env(cfg.env.env_id, goal_sampler, render_mode)])
	env = VecNormalize.load(vecnorm_path, env)
	env.training = False
	env.norm_reward = False

	model = RecurrentPPO.load(model_path, env=env, device="cuda")
	obs = env.reset()

	if args.video or args.gif:
		output_path = args.video or args.gif
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
