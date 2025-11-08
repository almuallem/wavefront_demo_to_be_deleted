import sys

sys.path.append(".")
import torch
import torch.nn as nn
from utils.zernike import zern_abb
import numpy as np
import matplotlib.pyplot as plt


class INR_xi_encoded(nn.Module):
    def __init__(
        self,
        num_zern_features,
        hidden_dim,
        output_dim,
        support,
        scale=1,
        final=False,
        N_proj_approx=256,
    ):
        self.num_zern_features = num_zern_features
        super(INR_xi_encoded, self).__init__()
        self.zernike_xi_weights, self.xi_funcs = self.zernike_gram_schmidt(
            support, scale, N_proj_approx
        )

        # Define the MLP
        if not final:
            self.mlp = nn.Sequential(
                nn.Linear(self.num_zern_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.num_zern_features, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Tanh(),
            )

    def zernike_gram_schmidt(self, support, scale, N):
        zerns = zern_abb(
            N, torch.eye(self.num_zern_features), scale=scale, tilt_corr=False
        )
        xi_approx = zerns * 0
        xi_approx[0] = (zerns[0] * support) / torch.norm(zerns[0] * support)
        for i in np.arange(1, self.num_zern_features):
            xi_approx[i] = zerns[i] * support
            for j in range(i - 1):
                xi_approx[i] = (
                    xi_approx[i]
                    - torch.sum(xi_approx[j] * xi_approx[i] * support) * xi_approx[j]
                )
            xi_approx[i] = xi_approx[i] / torch.norm(xi_approx[i] * support)
        proj_weight_mat = torch.zeros((self.num_zern_features, self.num_zern_features))
        for i in range(self.num_zern_features):
            for j in range(i):
                proj_weight_mat[i, j] = torch.sum(xi_approx[i] * xi_approx[j])
        return proj_weight_mat, xi_approx

    def forward(self):
        return self.mlp(self.xi_funcs.reshape(self.num_zern_features, -1).permute(1, 0))
