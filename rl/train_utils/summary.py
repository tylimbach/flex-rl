import os
import logging

from omegaconf import DictConfig
import yaml

log: logging.Logger = logging.getLogger(__name__)

def print_summary(base_dir: str) -> None:
	snapshot_log_path = os.path.join(base_dir, "checkpoints", "snapshot_log.yaml")
	if not os.path.exists(snapshot_log_path):
		log.info("❗ No snapshot log found.")
		return

	with open(snapshot_log_path) as f:
		snapshot_log: DictConfig = yaml.safe_load(f)

	log.info("\\n📊 Training Summary:")
	log.info(f"Total Timesteps: {snapshot_log.get('cumulative_steps', 0)}")

	best_dir = os.path.join(base_dir, "checkpoints", "best")
	if os.path.exists(best_dir):
		log.info(f"Best Model Saved At: {best_dir}")
	else:
		log.info("No best model saved.")

	log.info(f"Full Snapshot Log: {snapshot_log_path}")
