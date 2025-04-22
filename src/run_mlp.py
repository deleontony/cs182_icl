import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from base_models import NeuralNetwork
import numpy as np
import random

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

set_seed(42)

# Example: 1D sine wave
def generate_sine_data(n_points=1000, noise_std=0.0):
    x = np.random.uniform(-10, 10, size=(n_points, 1))
    y = np.sin(x) + np.random.normal(0, noise_std, size=(n_points, 1))
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    for x_batch, y_batch in dataloader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
    return loss.item()

def evaluate(model, dataloader, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            losses.append(loss.item())
    return np.mean(losses)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 1
    val_losses = []

    for run in range(10):
        print(f"\n--- Run {run + 1} ---")
        set_seed(run)  # Different seed for each run

        # Model & optimizer
        model = NeuralNetwork(in_size=input_dim, hidden_size=64, out_size=1).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Data (fixed for fairness — same train/val across runs)
        x_train, y_train = generate_sine_data(800)
        x_val, y_val = generate_sine_data(200)
        train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
        val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=64)

        for epoch in range(100):
            train_loss = train(model, train_loader, optimizer, criterion, device)
            val_loss = evaluate(model, val_loader, criterion, device)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

        val_losses.append(val_loss)

    print("\n📊 Average Validation Loss over 10 runs:", np.mean(val_losses))

if __name__ == "__main__":
    main()