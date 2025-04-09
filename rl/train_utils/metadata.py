import datetime
import os

from omegaconf import DictConfig
import yaml


def save_metadata(path: str, cfg: DictConfig, parent: str | None, resumed_at: int | None):
	metadata = {
		"experiment_name": os.path.basename(path),
		"created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
		"config": cfg,
		"parent": parent,
		"resumed_at_step": resumed_at,
	}
	with open(os.path.join(path, "metadata.yaml"), "w") as f:
		yaml.safe_dump(metadata, f)
