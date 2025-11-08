import sys

sys.path.append(".")

import torch
from utils.zernike import zern_abb
from utils.pupil import polymask
from utils.psf import psf
from matplotlib import pyplot as plt
from methods.proj_methods import gerchberg_saxton, fienup_hio
from utils.fourier import ft2
from utils.metrics import strehl, corr
import torch.optim as optim
import torch.nn as nn
from methods.xi_encoded_inr import INR_xi_encoded
from utils.dataset_prep import pupil_dataset
from utils.dataset_prep import s2p_alphas_betas
from utils.zernike import zernike_gram_schmidt
import math


def run_experiment(pupil_shape, N, inr=False):
    pupil, scale = pupil_dataset(pupil_shape, N)
    dataset = torch.load(f"datasets/table1_{pupil_shape}_{N}.pt", weights_only=False)
    tset = torch.load(f"datasets/y_phi_{pupil_shape}_{N}_train.pt", weights_only=False)
    vset = torch.load(f"datasets/y_phi_{pupil_shape}_{N}_val.pt", weights_only=False)
    L = len(vset) + len(tset)
    snr_db = 0
    snr_linear = 10 ** (snr_db / 10)  # Convert dB to linear scale

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
    signal_power = torch.mean(
        psfs[:, N // 2 - 10 : N // 2 + 10, N // 2 - 10 : N // 2 + 10] ** 2,
        dim=(-1, -2),
        keepdim=True,
    )
    psfs = psfs
    phases = torch.stack(phases)
    psfs_train = torch.stack(psfs_train)
    phases_train = torch.stack(phases_train)
    noise = torch.randn(*psfs.shape)
    noise_power = torch.mean(
        noise[:, N // 2 - 10 : N // 2 + 10, N // 2 - 10 : N // 2 + 10] ** 2,
        dim=(-1, -2),
        keepdim=True,
    )
    scaling_factor = torch.sqrt(signal_power / (noise_power * snr_linear))
    noisy_psfs = torch.clip(psfs + noise * scaling_factor, 0, math.inf)
    plt.imshow(noisy_psfs[0])
    plt.show()
    noisy_dataset = torch.utils.data.TensorDataset(
        noisy_psfs,
        phases,
    )
    # Verify the resulting SNR
    actual_noise_power = torch.mean((noise * scaling_factor) ** 2, dim=(-1, -2))
    actual_signal_power = torch.mean(psfs**2, dim=(-1, -2))
    actual_snr_db = 10 * torch.log10(actual_signal_power / actual_noise_power)
    print(f"Average resulting SNR: {torch.mean(actual_snr_db).item()} dB")
    actual_noise_power = torch.mean(
        (
            noise[:, N // 2 - 10 : N // 2 + 10, N // 2 - 10 : N // 2 + 10]
            * scaling_factor
        )
        ** 2,
        dim=(-1, -2),
    )
    actual_signal_power = torch.mean(
        psfs[:, N // 2 - 10 : N // 2 + 10, N // 2 - 10 : N // 2 + 10] ** 2, dim=(-1, -2)
    )
    actual_snr_db = 10 * torch.log10(actual_signal_power / actual_noise_power)
    print(f"Average resulting SNR: {torch.mean(actual_snr_db).item()} dB")

    xi, alphas, betas, psf_basis, psf_mu = s2p_alphas_betas(
        psfs_train, phases_train, pupil_shape, N, zoom=30, xi_count=21
    )
    psf_flat_cen = noisy_psfs[
        :, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30
    ].reshape(noisy_psfs.shape[0], -1) - torch.mean(
        psfs[:, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30].reshape(
            psfs.shape[0], -1
        ),
        dim=0,
        keepdim=True,
    )
    # # for newly generated data
    betas = psf_flat_cen @ torch.linalg.pinv(psf_basis[:100].reshape(100, -1))
    alphas = phases.reshape(phases.shape[0], -1) @ torch.linalg.pinv(xi.reshape(21, -1))

    # # Projection methods
    # Gerchberg-Saxton
    # out = gerchberg_saxton(torch.sqrt(noisy_psfs), pupil.unsqueeze(0), 500)
    # psf_hat = psf(pupil, out)
    # print(
    #     f"Gerchberg-Saxton, PR: {torch.mean(corr(psf_hat, psfs))} +- {torch.std(corr(psf_hat, psfs))}, WE:  {torch.mean(strehl(out, phases, pupil))} +- {torch.std(strehl(out, phases, pupil))}"
    # )

    # # Fienup's Hybrid Input-Output
    # out = fienup_hio(torch.sqrt(noisy_psfs), pupil.unsqueeze(0), 500, beta=0.5)
    # psf_hat = psf(pupil, out)
    # print(
    #     f"HIO, PR: {torch.mean(corr(psf_hat, psfs))} +- {torch.std(corr(psf_hat, psfs))}, WE:  {torch.mean(strehl(out, phases, pupil))} +- {torch.std(strehl(out, phases, pupil))}"
    # )

    # xi-Encoded INR, very VERY slow :( off by default
    if inr:
        psfs_inr = []
        phases_inr = []
        k = 0
        for psf_idx, phase in noisy_dataset:
            inr = INR_xi_encoded(21, 75, 1, pupil, scale, N_proj_approx=N)
            criterion = nn.MSELoss()
            opt_inr = optim.Adam(inr.parameters(), lr=20e-3)
            k += 1
            for i in range(1200):
                opt_inr.zero_grad()
                phase_est = inr()
                psf_est = psf(pupil, phase_est.reshape(N, N).unsqueeze(0))[0]
                loss = criterion(psf_idx, psf_est)
                loss.backward()
                opt_inr.step()
                # with torch.no_grad():
                #     if i % 100 == 0:
                #         plt.subplot(131)
                #         plt.imshow(psfs[k - 1])
                #         plt.subplot(132)
                #         plt.imshow(psf_idx)
                #         plt.subplot(133)
                #         plt.imshow(psf_est)
                #         plt.show()
            with torch.no_grad():
                psfs_inr.append(psf_est)
                phases_inr.append(phase_est.reshape(N, N))

            psfs_inrt = torch.stack(psfs_inr)
            phases_inrt = torch.stack(phases_inr)
            if k % 50 == 0:
                print(
                    f"{k}, xi-Encoded, PR: {torch.mean(corr(psfs_inrt[:k], psfs[:k]))} +- {torch.std(corr(psfs_inrt[:k], psfs[:k]))}, WE:  {torch.mean(strehl(phases_inrt[:k], phases[:k], pupil))} +- {torch.std(strehl(phases_inrt[:k], phases[:k], pupil))}"
                )
        print(
            f"{k}, xi-Encoded, PR: {torch.mean(corr(psfs_inrt[:k], psfs[:k]))} +- {torch.std(corr(psfs_inrt[:k], psfs[:k]))}, WE:  {torch.mean(strehl(phases_inrt[:k], phases[:k], pupil))} +- {torch.std(strehl(phases_inrt[:k], phases[:k], pupil))}"
        )

    # S2P-like methods
    s2p = torch.load(
        f"datasets/s2p_noisy_{pupil_shape}_{N}_train.pt", weights_only=False
    )
    alphas = s2p(betas)
    out = torch.einsum("dB, Bxy->dxy", alphas, xi)
    psf_hat = psf(pupil, out)
    print(
        f"S2P, PR: {torch.mean(corr(psf_hat, psfs))} +- {torch.std(corr(psf_hat, psfs))}, WE:  {torch.mean(strehl(out, phases, pupil))} +- {torch.std(strehl(out, phases, pupil))}"
    )

    mlp1 = torch.load(
        f"datasets/mlp1_noisy_{pupil_shape}_{N}_train.pt", weights_only=False
    )
    alphas = mlp1(
        noisy_psfs[:, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30].reshape(
            psfs.shape[0], -1
        )
    )
    xi = zernike_gram_schmidt(pupil, scale, N, 21)
    out = torch.einsum("dB, Bxy->dxy", alphas, xi)
    psf_hat = psf(pupil, out)
    print(
        f"mlp1, PR: {torch.mean(corr(psf_hat, psfs))} +- {torch.std(corr(psf_hat, psfs))}, WE:  {torch.mean(strehl(out, phases, pupil))} +- {torch.std(strehl(out, phases, pupil))}"
    )

    mlp2 = torch.load(
        f"datasets/mlp2_noisy_{pupil_shape}_{N}_train.pt", weights_only=False
    )
    out = mlp2(
        noisy_psfs[:, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30].reshape(
            psfs.shape[0], -1
        )
    )
    phases_out = phases * 0.0
    phases_out[:, pupil > 0.1] = out * 1.0
    psf_hat = psf(pupil, phases_out)
    print(
        f"mlp2, PR: {torch.mean(corr(psf_hat, psfs))} +- {torch.std(corr(psf_hat, psfs))}, WE:  {torch.mean(strehl(phases_out, phases, pupil))} +- {torch.std(strehl(phases_out, phases, pupil))}"
    )

    us2p = torch.load(
        f"datasets/unrolled_noisy_s2p_{pupil_shape}_{N}_train.pt", weights_only=False
    )
    out = us2p(
        noisy_psfs[:, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30].reshape(
            psfs.shape[0], -1
        ),
        zoom=30,
    )
    psf_hat = psf(pupil, out.reshape(psfs.shape[0], N, N))
    print(
        f"Unrolled S2P, PR: {torch.mean(corr(psf_hat, psfs))} +- {torch.std(corr(psf_hat, psfs))}, WE:  {torch.mean(strehl(out.reshape(psfs.shape[0], N, N), phases, pupil))} +- {torch.std(strehl(out.reshape(psfs.shape[0], N, N), phases, pupil))}"
    )


def gen_dataset(N, pupil_shape):
    L = 1000
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
    torch.save(tset, f"datasets/table1_{pupil_shape}_{N}.pt")


if __name__ == "__main__":
    # pentagon_scale, triangle_scale, circle_scale = 0.345, 0.4681, 0.3
    N = 128
    # gen_dataset(N, "circle")
    # gen_dataset(N, "pentagon")
    # gen_dataset(N, "triangle")
    # You'll have to generate the datasets first

    run_experiment("triangle", N, inr=False)
    run_experiment("pentagon", N, inr=False)
    run_experiment("circle", N, inr=False)
