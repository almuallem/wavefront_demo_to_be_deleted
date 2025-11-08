import sys

sys.path.append(".")
from utils.pupil import polymask
import torch
from utils.zernike import zernike_gram_schmidt
from utils.zernike import zern_abb


def pupil_dataset(pupil_shape, N):
    if pupil_shape == "circle":
        scale = 0.3
        pupil = polymask(N, -1, scale)
    if pupil_shape == "pentagon":
        scale = 0.345
        pupil = polymask(N, 5, scale)
    if pupil_shape == "triangle":
        scale = 0.4681
        pupil = polymask(N, 3, scale)
    return pupil, scale


def phase_dataset_crop(dataset, pupil_shape, N, zoom=30):
    psf_list = []
    phase_list = []
    pupil = pupil_dataset(pupil_shape, N)
    for psf, phase in dataset:
        cropped_input = psf[
            :, N // 2 - zoom : N // 2 + zoom, N // 2 - zoom : N // 2 + zoom
        ].reshape(-1)
        cropped_output = phase[pupil > 0.1]
        psf_list.append(cropped_input)
        phase_list.append(cropped_output)

    psf = torch.stack(psf_list)
    phase = torch.stack(phase_list)

    mod_dataset = torch.utils.data.TensorDataset(
        psf.reshape(len(dataset), -1),
        phase.reshape(len(dataset), -1),
    )
    return mod_dataset


def s2p_alphas_betas(
    psfs, phases, pupil_shape, N, zoom=-1, psf_basis_M=100, xi_count=21
):
    psfs = psfs * 1.0
    pupil, scale = pupil_dataset(pupil_shape, N)
    xi = zernike_gram_schmidt(pupil, scale, N, xi_count)
    xi_flat = xi.reshape(xi_count, -1)
    # coeff_mat = (zern_flat @ torch.linalg.pinv(xi_flat[:alphas.shape[1]]))
    xi_alphas = phases.reshape(psfs.shape[0], -1) @ torch.linalg.pinv(
        xi_flat[:xi_count]
    )
    if zoom > 0:
        n1, n2 = psfs.shape[1], psfs.shape[2]
        psfs = psfs[:, n1 // 2 - zoom : n1 // 2 + zoom, n2 // 2 - zoom : n2 // 2 + zoom]
    u, s, vt = torch.linalg.svd(
        (
            psfs.reshape(psfs.shape[0], -1)
            - torch.mean(psfs.reshape(psfs.shape[0], -1), dim=0, keepdim=True)
        ).permute(1, 0)
    )
    if zoom > 0:
        psf_basis = u.T.reshape(u.shape[0], zoom * 2, zoom * 2)
        betas = torch.diag(s) @ vt[: (zoom * 2) ** 2, :]
    if zoom < 0:
        psf_basis = u.T.reshape(u.shape[0], N, N)
        betas = torch.diag(s) @ vt[: N**2, :]

    return (
        xi,
        xi_alphas,
        betas[:psf_basis_M].T,
        psf_basis[:psf_basis_M],
        torch.mean(psfs, dim=(0)),
    )
