import torch
from zernike import zern_abb
from pupil import polymask
from psf import psf
from matplotlib import pyplot as plt
from methods.proj_methods import gerchberg_saxton, fienup_hio
from fourier import ft2
from metrics import strehl, corr
import torch.optim as optim
import torch.nn as nn
from methods.xi_encoded_inr import INR_xi_encoded
from dataset_prep import pupil_dataset
from dataset_prep import s2p_alphas_betas
from zernike import zernike_gram_schmidt
import time


def prepare_data(pupil_shape, N):
    print("Preparing the dataset for ", pupil_shape, " pupil. This may takes a few minutes...")
    pupil, scale = pupil_dataset(pupil_shape, N)
    dataset = torch.load(f"datasets/table1_{pupil_shape}_{N}_demo.pt", weights_only=False)
    tset = torch.load(f"datasets/y_phi_{pupil_shape}_{N}_train.pt", weights_only=False)
    vset = torch.load(f"datasets/y_phi_{pupil_shape}_{N}_val.pt", weights_only=False)
    L = len(vset) + len(tset)

    psfs = []
    phases = []
    for psf_idx, phase in dataset:
        psfs.append(psf_idx)
        phases.append(phase)
    psfs_train = []
    phases_train = []
    for psf_idx, phase in tset:
        psfs_train.append(psf_idx)
        phases_train.append(phase)
    for psf_idx, phase in vset:
        psfs_train.append(psf_idx)
        phases_train.append(phase)

    psfs = torch.stack(psfs)
    phases = torch.stack(phases)
    psfs_train = torch.stack(psfs_train)
    phases_train = torch.stack(phases_train)
    
    print("Calculating S2P basis")
    xi, alphas, betas, psf_basis, psf_mu = s2p_alphas_betas(
        psfs_train, phases_train, pupil_shape, N, zoom=30, xi_count=21
    )
    psf_flat_cen = psfs[
        :, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30
    ].reshape(psfs.shape[0], -1) - torch.mean(
        psfs[:, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30].reshape(
            psfs.shape[0], -1
        ),
        dim=0,
        keepdim=True,
    )
    # for newly generated data
    betas = psf_flat_cen @ torch.linalg.pinv(psf_basis[:100].reshape(100, -1))
    alphas = phases.reshape(phases.shape[0], -1) @ torch.linalg.pinv(xi.reshape(21, -1))

    print("Data prepeation finished!")
    return pupil, scale, dataset, tset, vset, psfs, phases, psfs_train, phases_train, xi, alphas, betas, psf_basis, psf_flat_cen, betas, alphas


def gen_dataset(N, pupil_shape):
    L = 10
    N = 128
    pupil, scale = pupil_dataset(pupil_shape, N)
    alphas = 4 * (torch.rand(L, 21) - 0.5)
    alphas[:, :3] = 0
    phases = zern_abb(
        N, alphas, 0.4681
    )  # 0.4681 for fairness across pupil shapes (explained in supp)
    psfs = psf(pupil, phases)

    tset = torch.utils.data.TensorDataset(
        psfs[:L],
        phases[:L],
    )
    torch.save(tset, f"datasets/table1_{pupil_shape}_{N}_demo.pt")
