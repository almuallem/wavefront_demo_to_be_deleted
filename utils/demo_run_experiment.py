import sys

sys.path.append("..")

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

def run_experiment(pupil_shape, N, inr=False, data = None):
    print("Running a noiseless experiment with ", pupil_shape, " pupil of size ", str(N))

    if data is not None:
        pupil, scale, dataset, tset, vset, psfs, phases, psfs_train, phases_train, xi, alphas, betas, psf_basis, psf_flat_cen, betas, alphas = data
        print ("Data already prepared!. Excuting the methods...")
    else:
        pupil, scale, dataset, tset, vset, psfs, phases, psfs_train, phases_train, xi, alphas, betas, psf_basis, psf_flat_cen, betas, alphas = prepare_data(pupil_shape, N)
    # pupil, scale = pupil_dataset(pupil_shape, N)
    # dataset = torch.load(f"datasets/table1_{pupil_shape}_{N}_demo.pt", weights_only=False)
    # tset = torch.load(f"datasets/y_phi_{pupil_shape}_{N}_train.pt", weights_only=False)
    # vset = torch.load(f"datasets/y_phi_{pupil_shape}_{N}_val.pt", weights_only=False)
    # L = len(vset) + len(tset)

    # print("Preparing the dataset...")
    # psfs = []
    # phases = []
    # for psf_idx, phase in dataset:
    #     psfs.append(psf_idx)
    #     phases.append(phase)
    # psfs_train = []
    # phases_train = []
    # for psf_idx, phase in tset:
    #     psfs_train.append(psf_idx)
    #     phases_train.append(phase)
    # for psf_idx, phase in vset:
    #     psfs_train.append(psf_idx)
    #     phases_train.append(phase)

    # psfs = torch.stack(psfs)
    # phases = torch.stack(phases)
    # psfs_train = torch.stack(psfs_train)
    # phases_train = torch.stack(phases_train)

    # print("Calculating S2P basis")
    # xi, alphas, betas, psf_basis, psf_mu = s2p_alphas_betas(
    #     psfs_train, phases_train, pupil_shape, N, zoom=30, xi_count=21
    # )
    # psf_flat_cen = psfs[
    #     :, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30
    # ].reshape(psfs.shape[0], -1) - torch.mean(
    #     psfs[:, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30].reshape(
    #         psfs.shape[0], -1
    #     ),
    #     dim=0,
    #     keepdim=True,
    # )
    # # for newly generated data
    # betas = psf_flat_cen @ torch.linalg.pinv(psf_basis[:100].reshape(100, -1))
    # alphas = phases.reshape(phases.shape[0], -1) @ torch.linalg.pinv(xi.reshape(21, -1))

    # Dictionary to store results: {'method': [pr_mean, pr_std, we_mean, we_std, time]}
    results = {}
    
    # Projection methods
    # Gerchberg-Saxton
    print("Running the Gerchberg Saxton")
    start_time = time.perf_counter()
    out = gerchberg_saxton(torch.sqrt(psfs), pupil.unsqueeze(0), 500)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    psf_hat = psf(pupil, out)
    
    # print(
    #     f"Gerchberg-Saxton, PR: {torch.mean(corr(psf_hat, psfs))} +- {torch.std(corr(psf_hat, psfs))}, WE:  {torch.mean(strehl(out, phases, pupil))} +- {torch.std(strehl(out, phases, pupil))}. Elapsed time: {elapsed_time} seconds"
    # )

    pr_mean_gs = torch.mean(corr(psf_hat, psfs)).item()
    pr_std_gs = torch.std(corr(psf_hat, psfs)).item()
    we_mean_gs = torch.mean(strehl(out, phases, pupil)).item()
    we_std_gs = torch.std(strehl(out, phases, pupil)).item()

    results['Gerchberg-Saxton'] = [pr_mean_gs, pr_std_gs, we_mean_gs, we_std_gs, elapsed_time]

    phigs, psfgs = out * 1.0, psf_hat * 1.0

    # Fienup's Hybrid Input-Output
    print("Running Fienup's HIO")
    start_time = time.perf_counter()
    out = fienup_hio(torch.sqrt(psfs), pupil.unsqueeze(0), 500, beta=0.5)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    psf_hat = psf(pupil, out)
    # print(
    #     f"HIO, PR: {torch.mean(corr(psf_hat, psfs))} +- {torch.std(corr(psf_hat, psfs))}, WE:  {torch.mean(strehl(out, phases, pupil))} +- {torch.std(strehl(out, phases, pupil))}. Elapsed time: {elapsed_time} seconds"
    # )

    pr_mean_hio = torch.mean(corr(psf_hat, psfs)).item()
    pr_std_hio = torch.std(corr(psf_hat, psfs)).item()
    we_mean_hio = torch.mean(strehl(out, phases, pupil)).item()
    we_std_hio = torch.std(strehl(out, phases, pupil)).item()

    results['HIO'] = [pr_mean_hio, pr_std_hio, we_mean_hio, we_std_hio, elapsed_time]

    phihio, psfhio = out * 1.0, psf_hat * 1.0

    # xi-Encoded INR, very VERY slow :( off by default
    if inr:
        print("Running xi-Encoded INR")
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

    s2p = torch.load(f"datasets/s2p_{pupil_shape}_{N}_train.pt", weights_only=False)
    print("Running the S2P method")
    start_time = time.perf_counter()
    alphas = s2p(betas)
    out = torch.einsum("dB, Bxy->dxy", alphas, xi)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    psf_hat = psf(pupil, out)
    # print(
    #     f"S2P, PR: {torch.mean(corr(psf_hat, psfs))} +- {torch.std(corr(psf_hat, psfs))}, WE:  {torch.mean(strehl(out, phases, pupil))} +- {torch.std(strehl(out, phases, pupil))}.  Elapsed time: {elapsed_time} second."
    # )

    pr_mean_s2p = torch.mean(corr(psf_hat, psfs)).item()
    pr_std_s2p = torch.std(corr(psf_hat, psfs)).item()
    we_mean_s2p = torch.mean(strehl(out, phases, pupil)).item()
    we_std_s2p = torch.std(strehl(out, phases, pupil)).item()
    
    results['S2P'] = [pr_mean_s2p, pr_std_s2p, we_mean_s2p, we_std_s2p, elapsed_time]

    phis2p, psfs2p = out * 1.0, psf_hat * 1.0
    # with torch.no_grad():
    #     for i in range(4):
    #         plt.figure(figsize=(18, 10))
    #         plt.subplot(2, 4, 1)
    #         plt.title("Ground Truth PSF")
    #         plt.imshow(psfs[i]) # Ground Truth PSF
            
    #         plt.subplot(2, 4, 2)
    #         plt.title("PSF Gerchberg Saxton")
    #         plt.imshow(psfgs[i])
            
    #         plt.subplot(2, 4, 3)
    #         plt.title("PSF HIO")
    #         plt.imshow(psfhio[i])

    #         plt.subplot(2, 4, 4)
    #         plt.title("PSF S2P")
    #         plt.imshow(psfs2p[i])

    #         # Row 2: Phases
    #         plt.subplot(2, 4, 5)
    #         plt.title("Ground Truth Phase")
    #         plt.imshow(phases[i] * pupil) # Ground Truth Phase

    #         plt.subplot(2, 4, 6)
    #         plt.title("Phase Gerchberg Saxton")
    #         plt.imshow(phigs[i])

    #         plt.subplot(2, 4, 7)
    #         plt.title("Phase HIO")
    #         plt.imshow(phihio[i])

    #         plt.subplot(2, 4, 8)
    #         plt.title("Phase S2P")
    #         plt.imshow(phis2p[i])
            
    #         # Optional: Add tight_layout to prevent overlap
    #         plt.tight_layout()
    #         plt.show()
    return results, [psfs, psfgs, psfhio, psfs2p], [phases, phigs, phihio, phis2p], pupil

def parse_metrics(output_string):
    parts = output_string.split(' +- ')
    mean = float(parts[0].split(': ')[-1])
    std = float(parts[1].split(',')[0])
    return mean, std

