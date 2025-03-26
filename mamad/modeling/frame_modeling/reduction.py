import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),  # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32x32 -> 16x16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16x16 -> 8x8
            nn.ReLU(),
            nn.Flatten()
        )

        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)

        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 16 -> 32
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),   # 32 -> 64
            nn.Sigmoid()  # To bound output between 0 and 1
        )

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z)
        x = x.view(-1, 128, 8, 8)
        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


class OvercookedFrameDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.samples = []
        self.transform = transform

        for fname in os.listdir(data_dir):
            if fname.endswith(".pkl"):
                with open(os.path.join(data_dir, fname), "rb") as f:
                    traj = pickle.load(f)
                    for t in traj:
                        # could also add frame_t+1
                        self.samples.append(t["frame_t"])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame = self.samples[idx]  # shape: (H, W, 3)
        img = Image.fromarray(frame).resize((64, 64))
        if self.transform:
            img = self.transform(img)
        return img


def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div


def train_vae(vae, dataloader, epochs=20, lr=1e-3, device="cpu"):
    vae.to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        vae.train()

        for batch in dataloader:
            batch = batch.to(device)
            recon, mu, logvar = vae(batch)
            loss = loss_function(recon, batch, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.2f}")


# Visualize a batch
def show_reconstructions(vae, dataloader, device="cpu"):
    vae.to(device)
    batch = next(iter(dataloader)).to(device)
    with torch.no_grad():
        recon_batch, _, _ = vae(batch)

    batch = batch.cpu().numpy()
    recon_batch = recon_batch.cpu().numpy()

    n = min(8, batch.shape[0])
    plt.figure(figsize=(16, 4))
    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(np.transpose(batch[i], (1, 2, 0)))
        ax.set_title("Original")
        plt.axis("off")

        # Reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(np.transpose(recon_batch[i], (1, 2, 0)))
        ax.set_title("Reconstruction")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = OvercookedFrameDataset(
        "./overcooked_frames", transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    vae = ConvVAE(latent_dim=32)
    train_vae(vae, dataloader, epochs=25,
              device="cuda" if torch.cuda.is_available() else "cpu")

    # Save encoder/decoder
    torch.save(vae.state_dict(), "vae_overcooked.pth")

    # Load VAE model
    vae = ConvVAE(latent_dim=32)
    vae.load_state_dict(torch.load("vae_overcooked.pth", map_location="cpu"))
    vae.eval()

    # Dataset (resize to 64x64 and normalize to [0, 1])
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = OvercookedFrameDataset(
        "./overcooked_frames", transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Run the visualization
    show_reconstructions(vae, dataloader)
