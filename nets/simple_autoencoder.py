import torch
import torch.nn as nn

class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super.__init__()
        self.encoder = None
        self.decoder = None

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1), # Output: [16, 128, 128]
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), # Output: [32, 64, 64]
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1), # Output: [64, 32, 32]
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: [32, 64, 64]
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: [16, 128, 128]
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=3, stride=2, padding=1, output_padding=1), # Output: [3, 256, 256]

            nn.Sigmoid()
        )
    def forward(self, x):
        pass
