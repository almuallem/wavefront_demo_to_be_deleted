# wavefun/pupil/__init__.py
import torch
import math
import numpy as np


def zern_abb(N, vec, scale=1, tilt_corr=True, device="cpu"):
    """_summary_

    Args:
        N       : length of one side, output is NxN
        vec     : Zernike coefficient vector, (num_outputs, num_zernikes)
        scale   : scale should be [0,1], represents how much Zernike polynomials fill window. Defaults to 1.
        tilt_corr: set first few Zernikes to 0. Defaults to True.
        device  : torch device. Defaults to "cpu".

    Returns:
        _type_: _description_
    """
    vec = vec * 1.0  # local copy
    if tilt_corr:  # approximate tilt correction (turn off first 3 zernikes)
        vec[:, :3] = 0.0
    x, y = torch.meshgrid(
        torch.linspace(-1, 1, N), torch.linspace(-1, 1, N), indexing="ij"
    )
    r = torch.sqrt(x**2 + y**2) / scale
    theta = torch.atan2(y, x)
    polynomials = []  # list to store polynomials
    for n in range(25):  # just a really big number, it will terminate sooner
        for m in range(-n, n + 1, 2):
            Rnm = torch.zeros_like(r)
            for s in range((n - np.abs(m)) // 2 + 1):
                c = (
                    (-1) ** s
                    * math.factorial(n - s)
                    / (
                        math.factorial(s)
                        * math.factorial((n + np.abs(m)) // 2 - s)
                        * math.factorial((n - np.abs(m)) // 2 - s)
                    )
                )
                Rnm += c * r ** (n - 2 * s)
            if m >= 0:
                Znm = Rnm * torch.cos(m * theta)
            else:
                Znm = Rnm * torch.sin(abs(m) * theta)

            # Zernike polynomials are defined only within the unit disk
            Znm[r > 1] = 0

            polynomials.append(Znm)
            if len(polynomials) >= vec.shape[1]:
                break
        if len(polynomials) >= vec.shape[1]:
            break
    zernikes = torch.stack(polynomials)
    weighted_sum = torch.einsum("mi,ijk->mjk", vec, zernikes)
    return weighted_sum.to(device)


def zernike_gram_schmidt(support, scale, N, Z):
    zerns = zern_abb(N, torch.eye(Z), scale=scale, tilt_corr=False)
    xi = zerns * 0
    xi[0] = (zerns[0] * support) / torch.norm(zerns[0] * support)
    for i in np.arange(1, Z):
        xi[i] = zerns[i] * support
        for j in range(i - 1):
            xi[i] = xi[i] - torch.sum(xi[j] * xi[i] * support) * xi[j]
        xi[i] = xi[i] / torch.norm(xi[i] * support)
    return xi
