import os
import numpy as np
import torch
from torch.utils.data import Dataset

class WidebandULADataset(Dataset):
    def __init__(self, folder, normalize=True):
        self.signal_files = sorted([f for f in os.listdir(folder) if f.startswith("signal_")])
        self.folder = folder
        self.normalize = normalize

    def __len__(self):
        return len(self.signal_files)

    def __getitem__(self, idx):
        signal = np.load(os.path.join(self.folder, self.signal_files[idx]))
        doa_file = self.signal_files[idx].replace("signal_", "doa_")
        doa = np.load(os.path.join(self.folder, doa_file))[0]
        if self.normalize:
            signal = (signal - signal.mean()) / (signal.std() + 1e-6)
        # Output shape: [antennas, samples], label: [angle]
        return torch.tensor(signal, dtype=torch.float32), torch.tensor([doa], dtype=torch.float32)
