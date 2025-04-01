import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
	def __init__(self, data_dir, max_length=1000, flattened=False):
		self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
		self.max_length = max_length
		self.flattened = flattened
		self.trajectories = []
		self.index = []
		self._load_data()

	def _load_data(self):
		for file_idx, file in enumerate(self.files):
			data = np.load(file)
			trajectory = {
				"observations": torch.tensor(data["observations"], dtype=torch.float32),
				"actions": torch.tensor(data["actions"], dtype=torch.float32),
				"rewards": torch.tensor(data["rewards"], dtype=torch.float32),
				"dones": torch.tensor(data["dones"], dtype=torch.bool),
				"goal": str(data["goal"]),
			}
			self.trajectories.append(trajectory)

			if self.flattened:
				for t in range(len(trajectory["observations"])):
					self.index.append((file_idx, t))

	def __len__(self):
		if self.flattened:
			return len(self.index)
		else:
			return len(self.trajectories)

	def __getitem__(self, index):
		if self.flattened:
			episode_idx, timestep = self.index[index]
			sample = self.trajectories[episode_idx]
			return {
				"observation": sample["observations"][timestep],
				"action": sample["actions"][timestep],
				"reward": sample["rewards"][timestep],
				"done": sample["dones"][timestep],
				"goal": sample["goal"]
			}
		else:
			traj = self.trajectories[index]
			length = min(len(traj["observations"]), self.max_length)
			return {
				"observations": traj["observations"][:length],
				"actions": traj["actions"][:length],
				"rewards": traj["rewards"][:length],
				"dones": traj["dones"][:length],
				"goal": traj["goal"]
			}


if __name__ == "__main__":
	dataset = TrajectoryDataset("workspace/humanoid_balanced/trajectories", flattened=True)
	print(f"✅ Loaded {len(dataset)} samples")
	print(dataset[0])

	dataset_seq = TrajectoryDataset("workspace/humanoid_balanced/trajectories", flattened=False)
	print(f"✅ Loaded {len(dataset_seq)} trajectories")
	print(dataset_seq[0])
