import os
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import stft

class WidebandULASTFTDataset(Dataset):
    def __init__(self, folder, n_fft=128, hop=64, normalize=True):
        self.signal_files = sorted([f for f in os.listdir(folder) if f.startswith("signal_")])
        self.folder = folder
        self.n_fft = n_fft
        self.hop = hop
        self.normalize = normalize

    def __len__(self):
        return len(self.signal_files)

    def __getitem__(self, idx):
        signal = np.load(os.path.join(self.folder, self.signal_files[idx]))  # [antennas, samples]
        # STFT for each antenna
        stfts = []
        for ch in signal:
            _, _, Zxx = stft(ch, nperseg=self.n_fft, noverlap=self.n_fft - self.hop)
            stfts.append(np.abs(Zxx))  # [n_freq, n_time]
        x = np.stack(stfts)  # [antennas, n_freq, n_time]
        if self.normalize:
            x = (x - x.mean()) / (x.std() + 1e-6)
        doa_file = self.signal_files[idx].replace("signal_", "doa_")
        doa = np.load(os.path.join(self.folder, doa_file))[0]
        return torch.tensor(x, dtype=torch.float32), torch.tensor([doa], dtype=torch.float32)