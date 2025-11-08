# wavefun/pupil/__init__.py
import torch
import math


def polymask(N, poly=-1, scale=1, device="cpu"):
    rx, ry = torch.meshgrid(
        torch.linspace(-1, 1, N), torch.linspace(-1, 1, N), indexing="xy"
    )
    rho, theta = torch.sqrt(rx**2 + ry**2), torch.atan2(ry, rx) + math.pi
    if poly != -1 and poly >= 3:
        alph = math.pi / poly
        Ualph = theta - torch.floor((theta + alph) / (2 * alph)) * 2 * alph
        rho = rho / (scale * math.cos(alph) / (torch.cos(Ualph)))
    else:
        rho /= scale
    return ((rho < 1) * 1).to(device)
