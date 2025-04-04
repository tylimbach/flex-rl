import os
import yaml


def print_summary(base_dir):
	snapshot_log = os.path.join(base_dir, "checkpoints", "snapshot_log.yaml")
	if not os.path.exists(snapshot_log):
		print("❗ No snapshot log found.")
		return

	with open(snapshot_log) as f:
		log = yaml.safe_load(f)

	print("\\n📊 Training Summary:")
	print(f"Total Timesteps: {log.get('cumulative_steps', 0)}")

	best_dir = os.path.join(base_dir, "checkpoints", "best")
	if os.path.exists(best_dir):
		print(f"Best Model Saved At: {best_dir}")
	else:
		print("No best model saved.")

	print(f"Full Snapshot Log: {snapshot_log}")
