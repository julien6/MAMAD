import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from mamad.modeling.prepare_data import OvercookedTransitionDataset, OvercookedPredictor

def evaluate_model(model, dataset, device="cpu", batch_size=128):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    mse_total = 0.0
    mae_total = 0.0
    cosine_total = 0.0
    n = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        with torch.no_grad():
            y_pred = model(x_batch)

        mse = F.mse_loss(y_pred, y_batch, reduction='sum').item()
        mae = F.l1_loss(y_pred, y_batch, reduction='sum').item()
        cosine = F.cosine_similarity(y_pred, y_batch, dim=1).sum().item()

        batch_size = x_batch.size(0)
        mse_total += mse
        mae_total += mae
        cosine_total += cosine
        n += batch_size

    mse_avg = mse_total / n
    mae_avg = mae_total / n
    cosine_avg = cosine_total / n

    print("Evaluation results:")
    print(f"  MSE:   {mse_avg:.6f}")
    print(f"  MAE:   {mae_avg:.6f}")
    print(f"  Cosine Similarity: {cosine_avg:.6f}")
    return mse_avg, mae_avg, cosine_avg

# Reload dataset & model
dataset = OvercookedTransitionDataset("./overcooked_data")
model = OvercookedPredictor()
model.load_state_dict(torch.load("overcooked_predictor.pth"))

# Evaluate
evaluate_model(model, dataset)
