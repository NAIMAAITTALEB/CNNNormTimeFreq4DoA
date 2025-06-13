import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block (optional)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    """A basic residual block for CNN2D"""
    def __init__(self, in_ch, out_ch, kernel_size=3, dropout=0.2):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch)
        self.use_res = (in_ch == out_ch)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.use_res:
            out += identity
        out = self.act(out)
        return out

class CNN2D_Doa(nn.Module):
    def __init__(self, num_antennas=8, n_freq=65, n_time=8, dropout=0.3):
        super().__init__()
        # Feature extractor with residual blocks
        self.block1 = ResidualBlock(num_antennas, 32, dropout=dropout)
        self.block2 = ResidualBlock(32, 64, dropout=dropout)
        self.block3 = ResidualBlock(64, 128, dropout=dropout)
        self.block4 = ResidualBlock(128, 256, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool2d(1)  # Global pooling
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)  # [batch, 256, 1, 1]
        x = x.view(x.size(0), -1)  # [batch, 256]
        x = self.dropout(F.leaky_relu(self.fc1(x), negative_slope=0.1))
        return self.fc2(x)  # [batch, 1]
