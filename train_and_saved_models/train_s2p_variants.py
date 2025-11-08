import sys

sys.path.append(".")
import torch
from utils.fourier import ft2, ift2
import matplotlib.pyplot as plt
from utils.psf import psf
from utils.pupil import polymask
from utils.zernike import zern_abb
from IPython.display import clear_output
from methods.s2p_variants import S2P
from utils.dataset_prep import pupil_dataset
from utils.zernike import zernike_gram_schmidt
from utils.dataset_prep import s2p_alphas_betas


def train_s2p(N, pupil_shape, noiseless=True):
    tset = torch.load(f"datasets/y_phi_{pupil_shape}_{N}_train.pt", weights_only=False)
    vset = torch.load(f"datasets/y_phi_{pupil_shape}_{N}_val.pt", weights_only=False)
    pupil, scale = pupil_dataset(pupil_shape, N)
    pupil_flat = pupil.reshape(-1)
    # s2p = torch.load(f"datasets/s2p_{pupil_shape}_{N}_train.pt", weights_only=False)
    L = len(vset) + len(tset)
    xiZ = 21

    psf_list = []
    phase_list = []
    for psf_idx, phase_idx in tset:
        psf_list.append(psf_idx)
        phase_list.append(phase_idx)
    for psf_idx, phase_idx in vset:
        psf_list.append(psf_idx)
        phase_list.append(phase_idx)
    psfs = torch.stack(psf_list)
    phase = torch.stack(phase_list)

    xi, alphas, betas, psf_basis, psf_mu = s2p_alphas_betas(
        psfs, phase, pupil_shape, N, zoom=30, xi_count=xiZ
    )
    save_for_val = 500
    if noiseless:
        tset = torch.utils.data.TensorDataset(
            betas[: L - save_for_val],
            alphas[: L - save_for_val],
        )
        vset = torch.utils.data.TensorDataset(
            betas[L - save_for_val :],
            alphas[L - save_for_val :],
        )
    if not noiseless:
        tset = torch.utils.data.TensorDataset(
            psfs[: L - save_for_val],
            alphas[: L - save_for_val],
        )
        vset = torch.utils.data.TensorDataset(
            psfs[L - save_for_val :],
            alphas[L - save_for_val :],
        )

    batch_size = 128
    t_loader = torch.utils.data.DataLoader(tset, batch_size=batch_size, shuffle=True)
    v_loader = torch.utils.data.DataLoader(vset, batch_size=batch_size, shuffle=True)

    s2p = S2P(100, 400, 3, xiZ)
    opt = torch.optim.Adam(s2p.parameters(), 2e-5)
    criterion = torch.nn.MSELoss()

    for epoch in range(1000):
        train_loss = 0
        for xbatch, ybatch in t_loader:
            opt.zero_grad()
            if not noiseless:
                noise = torch.randn((xbatch.shape[0], N**2))
                noise = noise / torch.norm(noise, dim=(-1), keepdim=True)
                xbatch_noise = (
                    xbatch[
                        :, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30
                    ].reshape(xbatch.shape[0], -1)
                    - psf_mu.reshape(1, -1)
                    + noise[:, : (2 * 30) ** 2] * torch.rand((xbatch.shape[0], 1))
                )
                xbatch_betas = (xbatch_noise) @ torch.linalg.pinv(
                    psf_basis[:100].reshape(100, -1)
                )
                yout = s2p(xbatch_betas)
            else:
                yout = s2p(xbatch)
            loss = criterion(yout, ybatch)
            loss.backward()
            train_loss += loss.item()
            opt.step()

        valid_loss = 0
        with torch.no_grad():
            for xbatch, ybatch in v_loader:
                xbatch_betas = xbatch[
                    :, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30
                ].reshape(xbatch.shape[0], -1) @ torch.linalg.pinv(
                    psf_basis[:100].reshape(100, -1)
                )
                yout = s2p(xbatch_betas)
                loss = criterion(yout, ybatch)
                valid_loss += loss.item()
            print(
                f"epoch: {epoch}, tloss: {train_loss / len(t_loader)}, vloss: {valid_loss / len(v_loader)}"
            )
    if not noiseless:
        torch.save(s2p, f"datasets/s2p_noisy_{pupil_shape}_{N}_train.pt")
    else:
        torch.save(s2p, f"datasets/s2p_{pupil_shape}_{N}_train.pt")


def train_mlp1(N, pupil_shape, noiseless=True):
    tset = torch.load(f"datasets/y_phi_{pupil_shape}_{N}_train.pt", weights_only=False)
    vset = torch.load(f"datasets/y_phi_{pupil_shape}_{N}_val.pt", weights_only=False)
    pupil, scale = pupil_dataset(pupil_shape, N)
    pupil_flat = pupil.reshape(-1)
    # s2p = torch.load(f"datasets/s2p_{pupil_shape}_{N}_train.pt", weights_only=False)
    L = len(vset) + len(tset)
    xiZ = 21

    psf_list = []
    phase_list = []
    for psf_idx, phase_idx in tset:
        psf_list.append(psf_idx)
        phase_list.append(phase_idx)
    for psf_idx, phase_idx in vset:
        psf_list.append(psf_idx)
        phase_list.append(phase_idx)
    psfs = torch.stack(psf_list)
    phase = torch.stack(phase_list)

    xi, alphas, betas, psf_basis, psf_mu = s2p_alphas_betas(
        psfs, phase, pupil_shape, N, zoom=30, xi_count=xiZ
    )
    save_for_val = 500
    tset = torch.utils.data.TensorDataset(
        psfs[:, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30].reshape(L, -1)[
            : L - save_for_val
        ],
        alphas[: L - save_for_val],
    )
    vset = torch.utils.data.TensorDataset(
        psfs[:, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30].reshape(L, -1)[
            L - save_for_val :
        ],
        alphas[L - save_for_val :],
    )

    batch_size = 64
    t_loader = torch.utils.data.DataLoader(tset, batch_size=batch_size, shuffle=True)
    v_loader = torch.utils.data.DataLoader(vset, batch_size=batch_size, shuffle=True)

    mlp1 = S2P((2 * 30) ** 2, 400, 3, xiZ)
    opt = torch.optim.Adam(mlp1.parameters(), 2e-5)
    criterion = torch.nn.MSELoss()

    for epoch in range(1000):
        train_loss = 0
        for xbatch, ybatch in t_loader:
            opt.zero_grad()
            if not noiseless:
                noise = torch.randn(*xbatch.shape)

                yout = mlp1(
                    xbatch
                    + noise
                    / torch.norm(noise, dim=-1, keepdim=True)
                    * torch.rand((xbatch.shape[0], 1))
                )
            else:
                yout = mlp1(xbatch)
            loss = criterion(yout, ybatch)
            loss.backward()
            train_loss += loss.item()
            opt.step()

        valid_loss = 0
        with torch.no_grad():
            for xbatch, ybatch in v_loader:
                yout = mlp1(xbatch)
                loss = criterion(yout, ybatch)
                valid_loss += loss.item()
            print(
                f"epoch: {epoch}, tloss: {train_loss / len(t_loader)}, vloss: {valid_loss / len(v_loader)}"
            )
    if not noiseless:
        torch.save(mlp1, f"datasets/mlp1_noisy_{pupil_shape}_{N}_train.pt")
    else:
        torch.save(mlp1, f"datasets/mlp1_{pupil_shape}_{N}_train.pt")


def train_mlp2(N, pupil_shape, noiseless=True):
    tset = torch.load(f"datasets/y_phi_{pupil_shape}_{N}_train.pt", weights_only=False)
    vset = torch.load(f"datasets/y_phi_{pupil_shape}_{N}_val.pt", weights_only=False)
    pupil, scale = pupil_dataset(pupil_shape, N)
    pupil_flat = pupil.reshape(-1)
    # s2p = torch.load(f"datasets/s2p_{pupil_shape}_{N}_train.pt", weights_only=False)
    L = len(vset) + len(tset)
    xiZ = 21

    psf_list = []
    phase_list = []
    for psf_idx, phase_idx in tset:
        psf_list.append(psf_idx)
        phase_list.append(phase_idx)
    for psf_idx, phase_idx in vset:
        psf_list.append(psf_idx)
        phase_list.append(phase_idx)
    psfs = torch.stack(psf_list)
    phase = torch.stack(phase_list)
    phase_crop = phase.reshape(L, -1)[:, pupil_flat > 0.1]

    save_for_val = 500
    tset = torch.utils.data.TensorDataset(
        psfs[:, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30].reshape(L, -1)[
            : L - save_for_val
        ],
        phase_crop[: L - save_for_val],
    )
    vset = torch.utils.data.TensorDataset(
        psfs[:, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30].reshape(L, -1)[
            L - save_for_val :
        ],
        phase_crop[L - save_for_val :],
    )

    batch_size = 64
    t_loader = torch.utils.data.DataLoader(tset, batch_size=batch_size, shuffle=True)
    v_loader = torch.utils.data.DataLoader(vset, batch_size=batch_size, shuffle=True)

    mlp2 = S2P((2 * 30) ** 2, 400, 3, phase_crop.shape[-1])
    opt = torch.optim.Adam(mlp2.parameters(), 2e-5)
    criterion = torch.nn.MSELoss()

    for epoch in range(1000):
        train_loss = 0
        for xbatch, ybatch in t_loader:
            opt.zero_grad()
            if not noiseless:
                noise = torch.randn(*xbatch.shape)
                yout = mlp2(
                    xbatch
                    + noise
                    / torch.norm(noise, dim=-1, keepdim=True)
                    * torch.rand((xbatch.shape[0], 1))
                )
            else:
                yout = mlp2(xbatch)
            loss = criterion(yout, ybatch)
            loss.backward()
            train_loss += loss.item()
            opt.step()

        valid_loss = 0
        with torch.no_grad():
            for xbatch, ybatch in v_loader:
                yout = mlp2(xbatch)
                loss = criterion(yout, ybatch)
                valid_loss += loss.item()
            print(
                f"epoch: {epoch}, tloss: {train_loss / len(t_loader)}, vloss: {valid_loss / len(v_loader)}"
            )
    if not noiseless:
        torch.save(mlp2, f"datasets/mlp2_noisy_{pupil_shape}_{N}_train.pt")
    else:
        torch.save(mlp2, f"datasets/mlp2_{pupil_shape}_{N}_train.pt")


if __name__ == "__main__":
    N = 128
    # train_s2p(N, "triangle", noiseless=True) # completed (1/17)
    # train_s2p(N, "pentagon", noiseless=True) # completed (1/17)
    # train_s2p(N, "circle", noiseless=True) # completed (1/17)
    # train_mlp1(N, "triangle", noiseless=True) # completed (1/17)
    # train_mlp1(N, "pentagon", noiseless=True)  # completed (1/17)
    # train_mlp1(N, "circle", noiseless=True)  # completed (1/17)
    # train_mlp2(N, "triangle", noiseless=True) # completed (1/17)
    # train_mlp2(N, "pentagon", noiseless=True)  # completed (1/17)
    # train_mlp2(N, "circle", noiseless=True)  # completed (1/17)

    train_s2p(N, "triangle", noiseless=False)  #
    train_s2p(N, "pentagon", noiseless=False)  #
    train_s2p(N, "circle", noiseless=False)  #
    train_mlp1(N, "triangle", noiseless=False)  #
    train_mlp1(N, "pentagon", noiseless=False)  #
    train_mlp1(N, "circle", noiseless=False)  #
    train_mlp2(N, "triangle", noiseless=False)  #
    train_mlp2(N, "pentagon", noiseless=False)  #
    train_mlp2(N, "circle", noiseless=False)  #
