# wavefun/pupil/__init__.py
import torch
import numpy as np


def oracle_s2p(phase, pupil, clean_psf, std=1e-2):
    noise1 = torch.randn(*clean_psf.shape) * std
    noise2 = torch.randn(*clean_psf.shape) * std
    clean_psf_flip = torch.fliplr(torch.flipud(clean_psf))
    phase_flip_conj = -torch.fliplr(torch.flipud(phase))
    success_failure = torch.le(
        torch.norm(torch.clip(noise1, 0, np.inf)) ** 2,
        torch.norm(torch.clip(clean_psf_flip - clean_psf_flip + noise2, 0, np.inf)),
    )
    out_phase = phase * 0
    out_phase[success_failure] = phase[success_failure]
    out_phase[torch.logical_not(success_failure)] = phase_flip_conj[
        torch.logical_not(success_failure)
    ]
    return out_phase
