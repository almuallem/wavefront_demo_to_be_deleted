# wavefun/psf/__init__.py
import torch
import math
from utils.fourier import ft2


def psf(pupil, phi, snr=torch.tensor([math.inf]), zoom=-1, device="cpu"):
    pupil = pupil.to(device)
    phi = phi.to(device)
    N = pupil.shape[0]

    psfs = torch.abs(ft2(pupil * torch.exp(1j * phi), device=device)) ** 2
    if zoom > 0:
        psf_clean = psfs[
            :, N // 2 - zoom : N // 2 + zoom, N // 2 - zoom : N // 2 + zoom
        ] / torch.sum(
            psfs[:, N // 2 - zoom : N // 2 + zoom, N // 2 - zoom : N // 2 + zoom],
            axis=(-1, -2),
            keepdims=True,
        )
    else:
        psf_clean = psfs / torch.sum(psfs, axis=(-1, -2), keepdims=True)

    if snr == math.inf:
        return psf_clean
    else:
        return noise_psf(psf_clean, device=device)


def noise_psf(psfs, device="cpu"):
    print("TODO")
