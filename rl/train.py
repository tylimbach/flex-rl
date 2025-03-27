import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
import os

RENDER = False

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
                if RENDER:
                    self.eval_env.render()
        return result


env_id = "HalfCheetah-v4"

train_env = gym.make(env_id)

if (RENDER):
    eval_env = gym.make(env_id, render_mode="human")
else:
    eval_env = gym.make(env_id, render_mode="human")

model = PPO("MlpPolicy", train_env, verbose=1, tensorboard_log="./workspace/ppo_tensorboard/")

# Use custom render-enabled EvalCallback
eval_callback = RenderEvalCallback(
    eval_env,
    best_model_save_path="./workspace/checkpoints/",
    log_path="./workspace/logs/",
    eval_freq=10_000,
    deterministic=True,
)

# Train model
model.learn(total_timesteps=500_000, callback=eval_callback)

# Save final model
os.makedirs("./workspace/checkpoints", exist_ok=True)
model.save("./workspace/checkpoints/ppo_halfcheetah_final")

print("✅ Training complete.")
