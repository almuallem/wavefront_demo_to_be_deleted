import torch
import sys

sys.path.append("../")
from utils.psf import psf


def corr(sig1, sig2):
    return torch.sum(sig1.real * sig2.real, dim=(-1, -2)) / (
        torch.norm(sig1.real, dim=(-1, -2)) * torch.norm(sig2.real, dim=(-1, -2))
    )


def strehl(phase1, phase2, pupil):
    diff = psf(pupil, pupil.unsqueeze(0) * 0.0)
    corrs = psf(pupil, (phase1 - phase2) * pupil.unsqueeze(0))

    return torch.amax(corrs, dim=(-1, -2)) / torch.amax(diff, dim=(-1, -2))
