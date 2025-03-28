import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import os

LOAD = True
LOAD_DIR = "./workspace/checkpoints/best_model_cheetah_v5.zip"
RENDER_EVAL = True
EVAL_FREQ = 50_000

N_STEPS = 4096
GAE_LAMBDA = 0.95
ENT_COEF = 0.001
LEARNING_RATE = 3e-4

SCHEDULE_TIMESTEPS = 1_000_000
TOTAL_TIMESTEPS = 3_000_000

class RenderEvalCallback(EvalCallback):
    def _on_step(self) -> bool:
        result = super()._on_step()
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

def make_env(render_mode=None):
    return lambda: gym.make("HalfCheetah-v5", render_mode=render_mode)

train_env = DummyVecEnv([make_env()])
train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

if os.path.exists(LOAD_DIR) and LOAD:
    print(f"Loading model from {LOAD_DIR}")
    model = PPO.load(LOAD_DIR, env=train_env, device="cpu")
else:
    print("Creating new PPO model")
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=N_STEPS,
        gae_lambda=GAE_LAMBDA,
        ent_coef=ENT_COEF,
        learning_rate=LEARNING_RATE,
        verbose=1,
        device="cpu",
        tensorboard_log="./workspace/ppo_tensorboard/"
    )

os.makedirs("./workspace", exist_ok=True)
train_env.save("./workspace/vecnormalize.pkl")

# Setup evaluation environment
eval_env = DummyVecEnv([lambda: Monitor(make_env(render_mode="human")())])
eval_env = VecNormalize.load("./workspace/vecnormalize.pkl", eval_env)
eval_env.training = False
eval_env.norm_reward = False

eval_callback = RenderEvalCallback(
    eval_env,
    best_model_save_path="./workspace/checkpoints/best_model_cheetah_v5",
    log_path="./workspace/logs/",
    eval_freq=EVAL_FREQ,
    deterministic=True,
)

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)

model.save("./workspace/checkpoints/ppo_halfcheetah_final")

train_env.close()
eval_env.close()
print("Training complete.")
