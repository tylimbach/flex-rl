import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
import os

RENDER_EVAL = True
EVAL_FREQ = 50_000

N_STEPS = 4096
GAE_LAMBDA = 0.95
ENT_COEF = 0.001
LEARNING_RATE = 3e-4

SCHEDULE_TIMESTEPS = 1_000_000
TOTAL_TIMESTEPS = 10_000_000

class RenderEvalCallback(EvalCallback):
    def _on_step(self) -> bool:
        # Use default evaluation behavior
        result = super()._on_step()
        # After evaluation is done, render a single episode (if this was an eval step)
        if self.eval_env.render_mode == "human" and self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            obs = self.eval_env.reset()
            done = False
            while not (done):
                action, _ = self.model.predict(np.array(obs) if not isinstance(obs, dict) else obs, deterministic=True)
                step_result = self.eval_env.step(action)
                obs, reward, done, info = step_result
                if RENDER_EVAL:
                    self.eval_env.render()
        return result

env_id = "HalfCheetah-v4"
train_env = DummyVecEnv([lambda: gym.make("HalfCheetah-v4")])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)
train_env.save("./workspace/vecnormalize.pkl")

eval_env = DummyVecEnv([lambda: gym.make("HalfCheetah-v4", render_mode="human")])
eval_env = VecNormalize.load("./workspace/vecnormalize.pkl", eval_env)
eval_env.training = False
eval_env.norm_reward = False 

model = PPO(
    "MlpPolicy",
    train_env,
    n_steps=N_STEPS,
    gae_lambda=GAE_LAMBDA,
    ent_coef=ENT_COEF,
    learning_rate=LEARNING_RATE,
    verbose=1,
    tensorboard_log="./workspace/ppo_tensorboard/"
)

eval_callback = RenderEvalCallback(
    eval_env,
    best_model_save_path="./workspace/checkpoints/",
    log_path="./workspace/logs/",
    eval_freq=EVAL_FREQ,
    deterministic=True,
)

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)

os.makedirs("./workspace/checkpoints", exist_ok=True)
model.save("./workspace/checkpoints/ppo_halfcheetah_final")

print("✅ Training complete.")
