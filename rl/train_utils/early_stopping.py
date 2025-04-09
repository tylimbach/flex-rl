import time
import logging
import math
from typing import final

from .config import EarlyStopperConfig

log = logging.getLogger(__name__)

@final
class EarlyStopper:
	def __init__(self, cfg: EarlyStopperConfig):
		self.enabled = cfg.enabled
		self.patience = cfg.patience
		self.min_delta = cfg.min_delta
		self.reward_threshold = cfg.reward_threshold
		self.fail_on_nan = cfg.fail_on_nan
		self.max_runtime = cfg.max_runtime_minutes * 60 if cfg.max_runtime_minutes else None

		self.best_reward = -math.inf
		self.counter = 0
		self.start_time = time.time()

	def should_stop(self, current_reward: float) -> bool:
		if not self.enabled:
			return False

		if self.fail_on_nan and (math.isnan(current_reward) or math.isinf(current_reward)):
			log.info("❌ Early stopping due to NaN/Inf reward.")
			return True

		if self.reward_threshold is not None and current_reward >= self.reward_threshold:
			log.info(f"✅ Reached reward threshold {self.reward_threshold}.")
			return True

		if current_reward > self.best_reward + self.min_delta:
			self.best_reward = current_reward
			self.counter = 0
		else:
			self.counter += 1
			if self.counter >= self.patience:
				log.info(f"🛑 Early stopping after {self.counter} rounds without improvement.")
				return True

		if self.max_runtime and (time.time() - self.start_time > self.max_runtime):
			log.info("⏱️ Max runtime exceeded.")
			return True

		return False
