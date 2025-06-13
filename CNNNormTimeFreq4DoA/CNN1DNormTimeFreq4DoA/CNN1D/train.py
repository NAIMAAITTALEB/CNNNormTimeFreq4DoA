import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from dataset import WidebandULADataset
from model import CNN1D_Doa
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
    def forward(self, pred, target):
        weight = 1.0 + self.alpha * torch.abs(target)
        return torch.mean(weight * (pred - target) ** 2)

if __name__ == "__main__":
    dataset = WidebandULADataset("dataset")
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])
    loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    model = CNN1D_Doa(num_antennas=8, num_samples=512)
    loss_fn = WeightedMSELoss(alpha=0.01)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(10):
        total_loss = 0
        for x, y in loader:
            preds = model(x)
            loss = loss_fn(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

        # Optional: validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                preds = model(x)
                val_loss += loss_fn(preds, y).item()
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
        model.train()

    torch.save(model.state_dict(), "cnn1d_wideband.pt")
