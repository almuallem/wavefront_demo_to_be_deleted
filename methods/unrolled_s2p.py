import torch
import torch.nn as nn
import torch.fft
import sys

sys.path.append(".")
from utils.psf import psf
from utils.zernike import zern_abb
from utils.zernike import zernike_gram_schmidt


class HIO_net(nn.Module):
    def __init__(self, K, pupil, phase_size, psf_size, hidden_size=128):
        super(HIO_net, self).__init__()
        self.K = K
        g_theta_k_list = []

        for k in range(K):
            if k == 0:
                g_theta_k = nn.Sequential(
                    nn.Linear(psf_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, phase_size),
                )
            else:
                g_theta_k = nn.Sequential(
                    nn.Linear(phase_size + 2 * psf_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, phase_size),
                )
            g_theta_k_list.append(g_theta_k)

        self.g_theta_k_list = nn.ModuleList(g_theta_k_list)
        self.pupil = pupil

    def forward(self, y):
        y_k_minus_1 = y  # Initialize y_0 with the input y.
        batch_size = y.shape[0]
        psize = self.pupil.shape[0]
        phase_out = torch.zeros((batch_size, psize**2))
        for k in range(1, self.K + 1):
            y_k_minus_2 = y_k_minus_1 * 1.0 if k > 1 else None
            if k == 1:
                nu_k = self.g_theta_k_list[k - 1](y_k_minus_1)
            else:
                nu_k = self.g_theta_k_list[k - 1](
                    torch.cat((y, y_k_minus_1, nu_k), dim=-1)
                )
            phase_out[:, self.pupil.reshape(-1) > 0.1] = nu_k * 1.0
            y_k = psf(
                self.pupil,
                phase_out.reshape(batch_size, psize, psize),
                zoom=30,
            ).reshape(batch_size, -1)
            y_k = torch.nn.functional.normalize(y_k)
            y_k_minus_1 = y_k * 1.0
        return nu_k


class HIO_zern_net(nn.Module):
    def __init__(self, K, pupil, psf_size, scale, hidden_size=128, zsize=21):
        super(HIO_zern_net, self).__init__()
        self.K = K
        g_theta_k_list = []
        self.scale = scale * 1.0
        self.xi = zernike_gram_schmidt(pupil, scale, 128, zsize)

        for k in range(K):
            if k == 0:
                g_theta_k = nn.Sequential(
                    nn.Linear(psf_size, hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, zsize),
                )
            else:
                g_theta_k = nn.Sequential(
                    nn.Linear(zsize + 2 * psf_size, hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(hidden_size),
                    nn.Linear(hidden_size, zsize),
                )
            g_theta_k_list.append(g_theta_k)

        self.g_theta_k_list = nn.ModuleList(g_theta_k_list)
        self.pupil = pupil

    def forward(self, y, zoom=-1):
        y_k_minus_1 = y  # Initialize y_0 with the input y.
        batch_size = y.shape[0]
        psize = self.pupil.shape[0]
        for k in range(1, self.K + 1):
            if k == 1:
                z_k = self.g_theta_k_list[k - 1](y_k_minus_1)
            else:
                z_k = self.g_theta_k_list[k - 1](
                    torch.cat((y, y_k_minus_1, z_k), dim=-1)
                )
            # phase_out[:, self.pupil.reshape(-1) > 0.1] = nu_k * 1.0
            # phase_k = zern_abb(psize, z_k, self.scale) * self.pupil
            phase_k = torch.einsum("Bd,dxy->Bxy", z_k, self.xi) * self.pupil
            y_k = psf(
                self.pupil,
                phase_k,
                zoom=zoom,
            ).reshape(batch_size, -1)
            y_k = y_k
            y_k_minus_1 = y_k * 1.0
        return phase_k.reshape(batch_size, -1)
