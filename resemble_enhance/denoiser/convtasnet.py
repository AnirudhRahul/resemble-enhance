import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, hidden_channels, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, channels, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class TCN(nn.Module):
    def __init__(self, channels, hidden_channels=512, kernel_size=3, num_layers=6):
        super().__init__()
        self.blocks = nn.ModuleList(
            [ConvBlock(channels, hidden_channels, kernel_size, 2 ** i) for i in range(num_layers)]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class ConvTasNet(nn.Module):
    """Simplified ConvTasNet operating on raw waveforms."""

    def __init__(self, enc_dim=256, hidden_channels=512, kernel_size=3, num_layers=6):
        super().__init__()
        stride = kernel_size * 2 // 2
        self.encoder = nn.Conv1d(1, enc_dim, kernel_size * 2, stride=stride, padding=kernel_size, bias=False)
        self.tcn = TCN(enc_dim, hidden_channels, kernel_size, num_layers)
        self.mask = nn.Conv1d(enc_dim, enc_dim, 1)
        self.decoder = nn.ConvTranspose1d(
            enc_dim, 1, kernel_size * 2, stride=stride, padding=kernel_size, bias=False
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (b 1 t)
        enc = self.encoder(x)
        m = torch.sigmoid(self.mask(self.tcn(enc)))
        out = self.decoder(enc * m)
        return out.squeeze(1)
