import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
import numpy as np
import torch.nn as nn

class OvercookedTransitionDataset(Dataset):
    def __init__(self, data_folder):
        self.samples = []
        for fname in os.listdir(data_folder):
            if fname.endswith(".pkl"):
                with open(os.path.join(data_folder, fname), "rb") as f:
                    traj = pickle.load(f)
                    self.samples.extend(traj)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        obs_agent1 = np.array(sample["observations"][0])
        obs_agent2 = np.array(sample["observations"][1])
        obsp1_agent1 = np.array(sample["next_observations"][0])
        obsp1_agent2 = np.array(sample["next_observations"][1])

        obs = np.concatenate([obs_agent1, obs_agent2])       # (192,)
        obsp1 = np.concatenate([obsp1_agent1, obsp1_agent2]) # (192,)

        # One-hot encode actions (discrete, 6 actions possibles)
        action1, action2 = sample["actions"]
        action1_oh = np.eye(6)[action1]
        action2_oh = np.eye(6)[action2]

        x_input = np.concatenate([obs, action1_oh, action2_oh])  # (204,)

        return torch.tensor(x_input, dtype=torch.float32), torch.tensor(obsp1, dtype=torch.float32)


class OvercookedPredictor(nn.Module):
    def __init__(self, input_dim=204, output_dim=192):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.model(x)

def train_predictor(model, dataloader, epochs=20, lr=1e-3, device="cpu"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        model.train()

        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x_batch.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

if __name__ == "__main__":
    # Dossier contenant les .pkl de transitions
    data_folder = "./overcooked_data"

    # Créer dataset et dataloader
    dataset = OvercookedTransitionDataset(data_folder)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialiser et entraîner le modèle
    model = OvercookedPredictor()
    train_predictor(model, dataloader, epochs=20, lr=1e-3)

    # Sauvegarde (optionnel)
    torch.save(model.state_dict(), "overcooked_predictor.pth")
