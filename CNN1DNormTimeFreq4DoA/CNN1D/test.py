import torch
from dataset import WidebandULADataset
from model import CNN1D_Doa
from torch.utils.data import DataLoader

if __name__ == "__main__":
    dataset = WidebandULADataset("dataset")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = CNN1D_Doa(num_antennas=8, num_samples=512)
    model.load_state_dict(torch.load("cnn1d_wideband.pt", map_location="cpu"))
    model.eval()

    errors = []
    for x, y in loader:
        pred = model(x).detach().numpy()[0][0]
        true = y.numpy()[0][0]
        error = abs(pred - true)
        errors.append(error)
        print(f"Pred: {pred:.2f}° / True: {true:.2f}° → Error: {error:.2f}°")

    print(f"Mean Absolute Error: {sum(errors)/len(errors):.2f}°")
