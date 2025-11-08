import sys

sys.path.append("..")

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


def run_experiment(pupil_shape, N, inr=False):
    print("Running a noiseless experiment with ", pupil_shape, " pupil of size ", str(N))
    pupil, scale = pupil_dataset(pupil_shape, N)
    dataset = torch.load(f"../datasets/table1_{pupil_shape}_{N}_demo.pt", weights_only=False)
    tset = torch.load(f"../datasets/y_phi_{pupil_shape}_{N}_train.pt", weights_only=False)
    vset = torch.load(f"../datasets/y_phi_{pupil_shape}_{N}_val.pt", weights_only=False)
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

    # Projection methods
    # Gerchberg-Saxton
    out = gerchberg_saxton(torch.sqrt(psfs), pupil.unsqueeze(0), 500)
    psf_hat = psf(pupil, out)
    print(
        f"Gerchberg-Saxton, PR: {torch.mean(corr(psf_hat, psfs))} +- {torch.std(corr(psf_hat, psfs))}, WE:  {torch.mean(strehl(out, phases, pupil))} +- {torch.std(strehl(out, phases, pupil))}"
    )

    phigs, psfgs = out * 1.0, psf_hat * 1.0

    # Fienup's Hybrid Input-Output
    out = fienup_hio(torch.sqrt(psfs), pupil.unsqueeze(0), 500, beta=0.5)
    psf_hat = psf(pupil, out)
    print(
        f"HIO, PR: {torch.mean(corr(psf_hat, psfs))} +- {torch.std(corr(psf_hat, psfs))}, WE:  {torch.mean(strehl(out, phases, pupil))} +- {torch.std(strehl(out, phases, pupil))}"
    )

    phihio, psfhio = out * 1.0, psf_hat * 1.0

    # xi-Encoded INR, very VERY slow :( off by default
    if inr:
        psfs_inr = []
        phases_inr = []
        k = 0
        for psf_idx, phase in dataset:
            k = k + 1
            inr = INR_xi_encoded(21, 75, 1, pupil, scale, N_proj_approx=N)
            criterion = nn.MSELoss()
            opt_inr = optim.Adam(inr.parameters(), lr=80e-3)

            for i in range(300):
                # if i % 199 == 0 and i > 0:
                #     print(i, k, loss.item())
                opt_inr.zero_grad()
                phase_est = inr()
                psf_est = psf(pupil, phase_est.reshape(N, N).unsqueeze(0))[0]
                loss = criterion(psf_idx, psf_est)
                loss.backward()
                opt_inr.step()
            with torch.no_grad():
                psfs_inr.append(psf_est)
                phases_inr.append(phase_est.reshape(N, N))

            psfs_inrt = torch.stack(psfs_inr)
            phases_inrt = torch.stack(phases_inr)
            if k > 1:
                print(
                    f"{k}, xi-Encoded, PR: {torch.mean(corr(psfs_inrt, psfs[:k]))} +- {torch.std(corr(psfs_inrt, psfs[:k]))}, WE:  {torch.mean(strehl(phases_inrt, phases[:k], pupil))} +- {torch.std(strehl(phases_inrt, phases[:k], pupil))}"
                )

    # phiinr, psfinr = phases_inrt * 1.0, psfs_inrt * 1.0

    # # S2P-like methods
    s2p = torch.load(f"../datasets/s2p_{pupil_shape}_{N}_train.pt", weights_only=False)
    alphas = s2p(betas)
    out = torch.einsum("dB, Bxy->dxy", alphas, xi)
    psf_hat = psf(pupil, out)
    print(
        f"S2P, PR: {torch.mean(corr(psf_hat, psfs))} +- {torch.std(corr(psf_hat, psfs))}, WE:  {torch.mean(strehl(out, phases, pupil))} +- {torch.std(strehl(out, phases, pupil))}"
    )
    phis2p, psfs2p = out * 1.0, psf_hat * 1.0

    ##########################START of methods that were not included in Nick's saved checkpoints

    # mlp1 = torch.load(f"datasets/mlp1_{pupil_shape}_{N}_train.pt", weights_only=False)
    # alphas = mlp1(
    #     psfs[:, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30].reshape(
    #         psfs.shape[0], -1
    #     )
    # )
    # xi = zernike_gram_schmidt(pupil, scale, N, 21)
    # out = torch.einsum("dB, Bxy->dxy", alphas, xi)
    # psf_hat = psf(pupil, out)
    # print(
    #     f"mlp1, PR: {torch.mean(corr(psf_hat, psfs))} +- {torch.std(corr(psf_hat, psfs))}, WE:  {torch.mean(strehl(out, phases, pupil))} +- {torch.std(strehl(out, phases, pupil))}"
    # )
    # phimlp1, psfmlp1 = out * 1.0, psf_hat * 1.0

    # mlp2 = torch.load(f"datasets/mlp2_{pupil_shape}_{N}_train.pt", weights_only=False)
    # out = mlp2(
    #     psfs[:, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30].reshape(
    #         psfs.shape[0], -1
    #     )
    # )
    # phases_out = phases * 0.0
    # phases_out[:, pupil > 0.1] = out * 1.0
    # psf_hat = psf(pupil, phases_out)
    # print(
    #     f"mlp2, PR: {torch.mean(corr(psf_hat, psfs))} +- {torch.std(corr(psf_hat, psfs))}, WE:  {torch.mean(strehl(phases_out, phases, pupil))} +- {torch.std(strehl(phases_out, phases, pupil))}"
    # )
    # phimlp2, psfmlp2 = phases_out * 1.0, psf_hat * 1.0

    # us2p = torch.load(
    #     f"datasets/unrolled_s2p_{pupil_shape}_{N}_train.pt", weights_only=False
    # )
    # out = us2p(
    #     torch.nn.functional.normalize(
    #         psfs[:, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30].reshape(
    #             psfs.shape[0], -1
    #         )
    #     ),
    #     zoom=30,
    # )
    # psf_hat = psf(pupil, out.reshape(psfs.shape[0], N, N))
    # print(
    #     f"Unrolled S2P, PR: {torch.mean(corr(psf_hat, psfs))} +- {torch.std(corr(psf_hat, psfs))}, WE:  {torch.mean(strehl(out.reshape(psfs.shape[0], N, N), phases, pupil))} +- {torch.std(strehl(out.reshape(psfs.shape[0], N, N), phases, pupil))}"
    # )
    # phius2p, psfus2p = out.reshape(psfs.shape[0], N, N) * 1.0, psf_hat * 1.0

    ##########################End of methods that are not included in Nick's saved points

    # with torch.no_grad():
    #     for i in range(10):
    #         plt.subplot(2, 7, 1)
    #         plt.imshow(psfgs[i])
    #         plt.subplot(2, 7, 2)
    #         plt.imshow(psfhio[i])
    #         plt.subplot(2, 7, 3)
    #         plt.imshow(psfinr[i])
    #         plt.subplot(2, 7, 4)
    #         plt.imshow(psfs2p[i])
    #         plt.subplot(2, 7, 5)
    #         plt.imshow(psfmlp1[i])
    #         plt.subplot(2, 7, 6)
    #         plt.imshow(psfmlp2[i])
    #         plt.subplot(2, 7, 7)
    #         plt.imshow(psfus2p[i])

    #         plt.subplot(2, 7, 8)
    #         plt.imshow(phigs[i])
    #         plt.subplot(2, 7, 9)
    #         plt.imshow(phihio[i])
    #         plt.subplot(2, 7, 10)
    #         plt.imshow(phiinr[i])
    #         plt.subplot(2, 7, 11)
    #         plt.imshow(phis2p[i])
    #         plt.subplot(2, 7, 12)
    #         plt.imshow(phimlp1[i])
    #         plt.subplot(2, 7, 13)
    #         plt.imshow(phimlp2[i])
    #         plt.subplot(2, 7, 14)
    #         plt.imshow(phius2p[i])
    #         plt.show()


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
    torch.save(tset, f"../datasets/table1_{pupil_shape}_{N}_demo.pt")


if __name__ == "__main__":
    # pentagon_scale, triangle_scale, circle_scale = 0.345, 0.4681, 0.3
    N = 128
    # gen_dataset(N, "circle")
    gen_dataset(N, "pentagon")
    gen_dataset(N, "triangle")
    # You'll have to generate the datasets first

    # run_experiment("triangle", N, inr=False)
    run_experiment("pentagon", N, inr=False)
    run_experiment("triangle", N, inr=False)
