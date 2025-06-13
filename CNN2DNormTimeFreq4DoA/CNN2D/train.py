from dataset import WidebandULASTFTDataset
from model import CNN2D_Doa
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
    def forward(self, pred, target):
        weight = 1.0 + self.alpha * torch.abs(target)
        return torch.mean(weight * (pred - target) ** 2)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

if __name__ == "__main__":
    dataset = WidebandULASTFTDataset("dataset", n_fft=128, hop=64)
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_set, val_set = random_split(dataset, [n_train, n_val])
    loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)

    model = CNN2D_Doa(num_antennas=8, n_freq=65, n_time=8)
    loss_fn = WeightedMSELoss(alpha=0.01)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
    num_epochs = 100
    best_val = float("inf")
    best_epoch = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, y in loader:
            preds = model(x)
            loss = loss_fn(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(loader)

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                preds = model(x)
                val_loss += loss_fn(preds, y).item()
        avg_val_loss = val_loss / len(val_loader)

        # Scheduler step
        scheduler.step(avg_val_loss)

        # Show learning rate, train/val loss
        print(f"Epoch {epoch+1:03d}, LR: {get_lr(optimizer):.2e}, "
              f"Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f} "
              f"{'(BEST)' if avg_val_loss < best_val else ''}")

        # Save best model
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), "cnn2d_wideband_best.pt")
            print(f"  >>> New best model saved at epoch {best_epoch}.")

        # Early stopping (optional)
        if avg_train_loss < 100 and avg_val_loss < 100:
            print("Target Loss reached (< 100). Early stopping.")
            break

    # Optionally, save the last model as well
    torch.save(model.state_dict(), "cnn2d_wideband_last.pt")
    print(f"Training complete. Best Validation Loss: {best_val:.4f} at epoch {best_epoch}.")
