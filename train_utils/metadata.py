import os
import yaml
import datetime


def save_metadata(path, cfg, parent=None, resumed_at=None):
	metadata = {
		"experiment_name": os.path.basename(path),
		"created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
		"config": cfg,
		"parent": parent,
		"resumed_at_step": resumed_at,
	}
	with open(os.path.join(path, "metadata.yaml"), "w") as f:
		yaml.safe_dump(metadata, f)


def print_lineage(path):
	print(f"\\n📄 Lineage for: {path}")
	while path:
		meta_path = os.path.join(path, "metadata.yaml")
		if not os.path.exists(meta_path):
			print(f"❗ No metadata found at {path}")
			break
		with open(meta_path) as f:
			meta = yaml.safe_load(f)
		print(f"\\n➡️  Experiment: {meta['experiment_name']}")
		print(f"   Created: {meta['created']}")
		if meta.get("resumed_at_step"):
			print(f"   Resumed at step: {meta['resumed_at_step']}")
		path = meta.get("parent")
