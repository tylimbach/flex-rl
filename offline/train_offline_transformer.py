import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from trajectory_dataset import TrajectoryDataset


class TrajectoryTransformer(nn.Module):
	def __init__(self, obs_dim, action_dim, goal_embedding_dim=16, d_model=128, nhead=4, num_layers=3):
		super().__init__()
		self.goal_embedding = nn.Embedding(4, goal_embedding_dim)
		self.input_proj = nn.Linear(obs_dim + goal_embedding_dim, d_model)
		encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
		self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
		self.output_proj = nn.Linear(d_model, action_dim)

	def forward(self, obs, goal_idx):
		goal_emb = self.goal_embedding(goal_idx).unsqueeze(1).repeat(1, obs.size(1), 1)
		x = torch.cat([obs, goal_emb], dim=-1)
		x = self.input_proj(x)
		x = self.transformer(x)
		return self.output_proj(x)


def goal_to_idx(goal):
	return {"walk forward": 0, "turn left": 1, "turn right": 2, "stand still": 3}[goal]


def train_transformer(dataset_dir, save_dir, batch_size=8, epochs=10):
	os.makedirs(save_dir, exist_ok=True)
	dataset = TrajectoryDataset(dataset_dir, flattened=False)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

	sample = dataset[0]
	obs_dim = sample["observations"].shape[-1]
	action_dim = sample["actions"].shape[-1]

	model = TrajectoryTransformer(obs_dim, action_dim).cuda()
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
	loss_fn = nn.MSELoss()

	for epoch in range(epochs):
		model.train()
		total_loss = 0

		for batch in dataloader:
			obs = torch.stack([torch.tensor(b["observations"]) for b in batch]).cuda()
			actions = torch.stack([torch.tensor(b["actions"]) for b in batch]).cuda()
			goals = torch.tensor([goal_to_idx(b["goal"]) for b in batch]).cuda()

			pred = model(obs, goals)
			loss = loss_fn(pred, actions)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			total_loss += loss.item()

		print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.6f}")

	torch.save(model.state_dict(), os.path.join(save_dir, "transformer_policy.pth"))
	print(f"✅ Transformer policy saved to {save_dir}")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--data", type=str, required=True, help="Path to trajectory directory")
	parser.add_argument("--out", type=str, default="workspace/transformer_policy", help="Output directory")
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--batch", type=int, default=8)
	args = parser.parse_args()

	train_transformer(args.data, args.out, args.batch, args.epochs)
