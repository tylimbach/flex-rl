from dataclasses import dataclass
import datetime
import os
import shutil
from typing import Any, cast
from gymnasium.core import ActType, ObsType, Env

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecNormalize
from dataclasses import dataclass, field, asdict
import yaml

@dataclass
class SnapshotEntry:
	step: int
	timestamp: str
	interrupt: bool
	eval_reward: float


@dataclass
class SnapshotLog:
	snapshots: list[SnapshotEntry] = field(default_factory=list)
	cumulative_steps: int = 0


def from_dict(d: dict[str, Any]) -> SnapshotLog:
	snapshots = [SnapshotEntry(**cast(dict[str, Any], s)) for s in d.get("snapshots", [])]
	return SnapshotLog(
		snapshots=snapshots,
		cumulative_steps=d.get("cumulative_steps", 0)
	)


def update_snapshot_log(save_dir: str, step: int, eval_reward: float, interrupt: bool = False) -> None:
	log_path = os.path.join(save_dir, "snapshot_log.yaml")

	if os.path.exists(log_path):
		with open(log_path) as f:
			log = from_dict(cast(dict[str, Any], yaml.safe_load(f) or {}))
	else:
		log = SnapshotLog()

	entry = SnapshotEntry(
		step=int(step),
		timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
		interrupt=bool(interrupt),
		eval_reward=float(eval_reward)
	)

	log.snapshots.append(entry)
	log.cumulative_steps = step

	with open(log_path, "w") as f:
		yaml.safe_dump(asdict(log), f)


def save_full_snapshot(model: BaseAlgorithm, env: VecNormalize, path: str) -> None:
	os.makedirs(path, exist_ok=True)
	model.save(os.path.join(path, "model.zip"))
	env.save(os.path.join(path, "vecnormalize.pkl"))
	experiment_dir = os.path.dirname(os.path.dirname(path))
	metadata_src = os.path.join(experiment_dir, "metadata.yaml")
	metadata_dst = os.path.join(path, "metadata.yaml")
	if os.path.exists(metadata_src):
		shutil.copy(metadata_src, metadata_dst)

	print(f"✅ Saved snapshot at {path}")
