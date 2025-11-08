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
    psfs = psfs[:, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30]
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

    for epoch in range(300):
        train_loss = 0
        for xbatch, ybatch in t_loader:
            opt.zero_grad()
            xbatch_betas = (
                xbatch.reshape(xbatch.shape[0], -1) - psf_mu.reshape(1, -1)
            ) @ torch.linalg.pinv(psf_basis[:100].reshape(100, -1))
            yout = s2p(xbatch_betas)
            phase_out = torch.einsum("dB,Bxy->dxy", yout, xi)
            psf_out = psf(pupil, phase_out)[
                :, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30
            ]
            loss = criterion(psf_out, xbatch)
            loss.backward()
            train_loss += loss.item()
            opt.step()

        valid_loss = 0
        with torch.no_grad():
            for xbatch, ybatch in v_loader:
                xbatch_betas = (
                    xbatch.reshape(xbatch.shape[0], -1) - psf_mu.reshape(1, -1)
                ) @ torch.linalg.pinv(psf_basis[:100].reshape(100, -1))
                yout = s2p(xbatch_betas)
                phase_out = torch.einsum("dB,Bxy->dxy", yout, xi)
                psf_out = psf(pupil, phase_out)[
                    :, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30
                ]
                loss = criterion(psf_out, xbatch)
                valid_loss += loss.item()
            print(
                f"epoch: {epoch}, tloss: {train_loss / len(t_loader)}, vloss: {valid_loss / len(v_loader)}"
            )
    torch.save(s2p, f"datasets/s2p_selfsup_{pupil_shape}_{N}_train.pt")


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
            yout = mlp1(xbatch)
            phase_out = torch.einsum("dB,Bxy->dxy", yout, xi)
            psf_out = psf(pupil, phase_out)[
                :, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30
            ]
            loss = criterion(psf_out.reshape(xbatch.shape[0], -1), xbatch)
            loss.backward()
            train_loss += loss.item()
            opt.step()

        valid_loss = 0
        with torch.no_grad():
            for xbatch, ybatch in v_loader:
                yout = mlp1(xbatch)
                phase_out = torch.einsum("dB,Bxy->dxy", yout, xi)
                psf_out = psf(pupil, phase_out)[
                    :, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30
                ]
                loss = criterion(psf_out.reshape(xbatch.shape[0], -1), xbatch)
                valid_loss += loss.item()
            print(
                f"epoch: {epoch}, tloss: {train_loss / len(t_loader)}, vloss: {valid_loss / len(v_loader)}"
            )
    torch.save(mlp1, f"datasets/mlp1_selfsup_{pupil_shape}_{N}_train.pt")


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

    mlp2 = S2P((2 * 30) ** 2, 400, 4, phase_crop.shape[-1])
    opt = torch.optim.Adam(mlp2.parameters(), 2e-5)
    criterion = torch.nn.MSELoss()

    for epoch in range(1000):
        train_loss = 0
        for xbatch, ybatch in t_loader:
            opt.zero_grad()
            yout = torch.zeros((ybatch.shape[0], N**2))
            yout[:, pupil_flat > 0.1] = mlp2(xbatch)
            psfout = psf(pupil, yout.reshape(ybatch.shape[0], N, N))[
                :, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30
            ]
            loss = criterion(xbatch, psfout.reshape(ybatch.shape[0], -1))
            loss.backward()
            train_loss += loss.item()
            opt.step()

        valid_loss = 0
        with torch.no_grad():
            for xbatch, ybatch in v_loader:
                yout = torch.zeros((ybatch.shape[0], N**2))
                yout[:, pupil_flat > 0.1] = mlp2(xbatch)
                psfout = psf(pupil, yout.reshape(ybatch.shape[0], N, N))[
                    :, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30
                ]
                loss = criterion(xbatch, psfout.reshape(ybatch.shape[0], -1))
                valid_loss += loss.item()
            print(
                f"epoch: {epoch}, tloss: {train_loss / len(t_loader)}, vloss: {valid_loss / len(v_loader)}"
            )
    torch.save(mlp2, f"datasets/mlp2_selfsup_{pupil_shape}_{N}_train.pt")


if __name__ == "__main__":
    N = 128
    train_s2p(N, "triangle", noiseless=True)  # completed (300 epochs)
    train_s2p(N, "pentagon", noiseless=True)  # completed (300 epochs)
    train_s2p(N, "circle", noiseless=True)  # completed (300 epochs)
    # train_mlp1(N, "pentagon", noiseless=True)
    # train_mlp1(N, "circle", noiseless=True)
    # train_mlp1(N, "triangle", noiseless=True)
    # train_mlp2(N, "pentagon", noiseless=True)
    # train_mlp2(N, "triangle", noiseless=True)
    # train_mlp2(N, "circle", noiseless=True)
