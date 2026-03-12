import torch
import torch.nn as nn

class SimpleAutoencoder(nn.Module):
    def __init__(self):
        super.__init__()
        self.encoder = None
        self.decoder = None

    def forward(self, x):
        pass
