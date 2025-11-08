import torch
import torch.nn as nn
import torch.fft
import sys

sys.path.append(".")
from utils.psf import psf
from utils.zernike import zern_abb


class S2P(nn.Module):
    def __init__(self, in_dim, hid_dim, layers, out_dim, device="cpu"):
        super(S2P, self).__init__()
        layers_list = [nn.Linear(in_dim, hid_dim), nn.ReLU()]
        layers_list.append(nn.LayerNorm(hid_dim))
        for _ in range(layers):
            layers_list.append(nn.Linear(hid_dim, hid_dim))
            layers_list.append(nn.ReLU())
            layers_list.append(nn.LayerNorm(hid_dim))
        layers_list.append(nn.Linear(hid_dim, out_dim))

        self.model = nn.Sequential(*layers_list)
        self.device = device
        self.to(device)

    def forward(self, y):
        return self.model(y)
