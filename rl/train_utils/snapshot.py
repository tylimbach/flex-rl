import datetime
import os
import shutil

import yaml


def update_snapshot_log(save_dir, step, eval_reward=None, interrupt=False):
	log_path = os.path.join(save_dir, "snapshot_log.yaml")
	if os.path.exists(log_path):
		with open(log_path) as f:
			log = yaml.safe_load(f)
	else:
		log = {"snapshots": [], "cumulative_steps": 0}

	entry = {
		"step": int(step),
		"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
		"interrupt": bool(interrupt),
	}
	if eval_reward is not None:
		entry["eval_reward"] = float(eval_reward)

	log["snapshots"].append(entry)
	log["cumulative_steps"] = int(step)

	with open(log_path, "w") as f:
		yaml.safe_dump(log, f)


def save_full_snapshot(model, env, path, step, eval_reward=None, interrupt=False):
	os.makedirs(path, exist_ok=True)
	model.save(os.path.join(path, "model.zip"))
	env.save(os.path.join(path, "vecnormalize.pkl"))
	experiment_dir = os.path.dirname(os.path.dirname(path))
	metadata_src = os.path.join(experiment_dir, "metadata.yaml")
	metadata_dst = os.path.join(path, "metadata.yaml")
	if os.path.exists(metadata_src):
		shutil.copy(metadata_src, metadata_dst)

	update_snapshot_log(os.path.dirname(path), step, eval_reward, interrupt)
	print(f"✅ Saved snapshot at {path}")
