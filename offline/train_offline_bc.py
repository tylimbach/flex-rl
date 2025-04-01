import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from trajectory_dataset import TrajectoryDataset


class MLPPolicy(nn.Module):
	def __init__(self, obs_dim, action_dim, goal_embedding_dim=16):
		super().__init__()
		self.goal_embedding = nn.Embedding(4, goal_embedding_dim)
		self.net = nn.Sequential(
			nn.Linear(obs_dim + goal_embedding_dim, 256),
			nn.ReLU(),
			nn.Linear(256, 256),
			nn.ReLU(),
			nn.Linear(256, action_dim)
		)
	
	def forward(self, obs, goal_idx):
		goal_emb = self.goal_embedding(goal_idx)
		inp = torch.cat([obs, goal_emb], dim=-1)
		return self.net(inp)


def goal_to_idx(goal):
	mapping = {
		"walk forward": 0,
		"turn left": 1,
		"turn right": 2,
		"stand still": 3
	}
	return mapping[goal]


def train_offline_bc(dataset_dir, save_dir, batch_size=512, epochs=10):
	os.makedirs(save_dir, exist_ok=True)
	dataset = TrajectoryDataset(dataset_dir, flattened=True)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

	sample = dataset[0]
	obs_dim = sample["observation"].shape[-1]
	action_dim = sample["action"].shape[-1]

	model = MLPPolicy(obs_dim, action_dim).cuda()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	loss_fn = nn.MSELoss()

	for epoch in range(epochs):
		total_loss = 0
		model.train()
		for batch in dataloader:
			obs = torch.stack([torch.tensor(o) for o in batch["observation"]]).cuda().squeeze(1)
			actions = torch.stack([torch.tensor(a) for a in batch["action"]]).cuda().squeeze(1)
			goal_idx = torch.tensor([goal_to_idx(g) for g in batch["goal"]]).cuda()

			pred = model(obs, goal_idx)
			loss = loss_fn(pred, actions)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			total_loss += loss.item()

		print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.6f}")

	torch.save(model.state_dict(), os.path.join(save_dir, "offline_bc_policy.pth"))
	print(f"✅ Offline BC policy saved to {save_dir}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", type=str, required=True, help="Path to trajectory directory")
	parser.add_argument("--out", type=str, default="workspace/offline_bc_policy", help="Output directory")
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--batch", type=int, default=512)
	args = parser.parse_args()

	train_offline_bc(args.data, args.out, args.batch, args.epochs)
