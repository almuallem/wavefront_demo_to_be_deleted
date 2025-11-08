import sys

sys.path.append(".")
import torch
from utils.fourier import ft2, ift2
import matplotlib.pyplot as plt
from utils.psf import psf
from utils.pupil import polymask
from utils.zernike import zern_abb
from IPython.display import clear_output
from methods.unrolled_s2p import HIO_net, HIO_zern_net
from utils.dataset_prep import pupil_dataset


def train_HIO_zern_supervised(N, pupil_shape, noiseless=True):
    tset = torch.load(f"datasets/y_phi_{pupil_shape}_{N}_train.pt", weights_only=False)
    vset = torch.load(f"datasets/y_phi_{pupil_shape}_{N}_val.pt", weights_only=False)
    pupil, scale = pupil_dataset(pupil_shape, N)
    pupil_flat = pupil.reshape(-1)
    L = len(vset) + len(tset)

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
    save_for_val = 500
    tset = torch.utils.data.TensorDataset(
        psfs[:, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30].reshape(L, -1)[
            : L - save_for_val
        ],
        phase.reshape(L, -1)[: L - save_for_val],
    )
    vset = torch.utils.data.TensorDataset(
        psfs[:, N // 2 - 30 : N // 2 + 30, N // 2 - 30 : N // 2 + 30].reshape(L, -1)[
            L - save_for_val :
        ],
        phase.reshape(L, -1)[L - save_for_val :],
    )

    batch_size = 64
    t_loader = torch.utils.data.DataLoader(tset, batch_size=batch_size, shuffle=True)
    v_loader = torch.utils.data.DataLoader(vset, batch_size=batch_size, shuffle=True)

    hionet = HIO_zern_net(
        3,
        pupil,
        (2 * 30) ** 2,
        scale,
        hidden_size=400,
    )

    opt = torch.optim.Adam(hionet.parameters(), 2e-4)
    criterion = torch.nn.MSELoss()

    for epoch in range(100):
        train_loss = 0
        for xbatch, ybatch in t_loader:
            opt.zero_grad()
            if not noiseless:
                noise = torch.randn(*xbatch.shape)

                yout = hionet(
                    xbatch
                    + noise
                    / torch.norm(noise, dim=-1, keepdim=True)
                    * torch.rand((xbatch.shape[0], 1)),
                    zoom=30,
                )
            else:
                yout = hionet(torch.nn.functional.normalize(xbatch), zoom=30)
            yout_cropped = yout[:, pupil_flat > 0.1]
            loss = criterion(yout_cropped, ybatch[:, pupil_flat > 0.1])
            loss.backward()
            train_loss += loss.item()
            opt.step()

        valid_loss = 0
        with torch.no_grad():
            for xbatch, ybatch in v_loader:
                if not noiseless:
                    noise = torch.randn(*xbatch.shape)

                    yout = hionet(
                        xbatch
                        + noise
                        / torch.norm(noise, dim=-1, keepdim=True)
                        * torch.rand((xbatch.shape[0], 1)),
                        zoom=30,
                    )
                else:
                    yout = hionet(torch.nn.functional.normalize(xbatch), zoom=30)
                yout_cropped = yout[:, pupil_flat > 0.1]
                loss = criterion(yout_cropped, ybatch[:, pupil_flat > 0.1])
                valid_loss += loss.item()
            print(
                f"epoch: {epoch}, tloss: {train_loss / len(t_loader)}, vloss: {valid_loss / len(v_loader)}"
            )
    if noiseless:
        torch.save(hionet, f"datasets/unrolled_s2p_{pupil_shape}_{N}_train.pt")
    if not noiseless:
        torch.save(hionet, f"datasets/unrolled_noisy_s2p_{pupil_shape}_{N}_train.pt")


def train_HIO_zern_selfsupervised(pupil_shape, noiseless=True):
    N = 128
    tset = torch.load(f"datasets/{pupil_shape}_train_19500.pt", weights_only=False)
    vset = torch.load(f"datasets/{pupil_shape}_val_500.pt", weights_only=False)
    if pupil_shape == "circle":
        scale = 0.3
        pupil = polymask(N, -1, scale)
    if pupil_shape == "pentagon":
        scale = 0.345
        pupil = polymask(N, 5, scale)
    if pupil_shape == "triangle":
        scale = 0.4681
        pupil = polymask(N, 3, scale)
    pupil_flat = pupil.reshape(-1)
    hionet = HIO_zern_net(
        5,
        pupil,
        vset.__getitem__(0)[1].shape[0],
        vset.__getitem__(0)[0].shape[0],
        hidden_size=128,
    )

    batch_size = 64
    t_loader = torch.utils.data.DataLoader(tset, batch_size=batch_size, shuffle=True)
    v_loader = torch.utils.data.DataLoader(vset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(hionet.parameters(), 2e-4)
    criterion = torch.nn.MSELoss()

    for epoch in range(100):
        train_loss = 0
        for xbatch, ybatch in t_loader:
            opt.zero_grad()
            yout = hionet(torch.nn.functional.normalize(xbatch))
            batch_size = xbatch.shape[0]
            # phi_hat = torch.zeros((batch_size, pupil.shape[0] ** 2))
            # phi_hat[:, pupil_flat > 0.1] = yout * 1.0
            y0 = psf(pupil, yout).reshape(batch_size, -1)
            loss = criterion(
                torch.nn.functional.normalize(xbatch), torch.nn.functional.normalize(y0)
            )
            loss.backward()
            train_loss += loss.item()

            opt.step()

        valid_loss = 0
        with torch.no_grad():
            for xbatch, ybatch in v_loader:
                yout = hionet(torch.nn.functional.normalize(xbatch))
                batch_size = xbatch.shape[0]
                # phi_hat = torch.zeros((batch_size, pupil.shape[0] ** 2))
                # phi_hat[:, pupil_flat > 0.1] = yout * 1.0
                y0 = psf(pupil, yout).reshape(batch_size, -1)
                loss = criterion(
                    torch.nn.functional.normalize(xbatch),
                    torch.nn.functional.normalize(y0),
                )
                valid_loss += loss.item()
            print(
                f"epoch: {epoch}, tloss: {train_loss / len(t_loader)}, vloss: {valid_loss / len(v_loader)}"
            )


if __name__ == "__main__":
    N = 128
    train_HIO_zern_supervised(N, "triangle", noiseless=False)  # completed
    train_HIO_zern_supervised(N, "pentagon", noiseless=False)
    train_HIO_zern_supervised(N, "circle", noiseless=False)
    # train_HIO_zern_selfsupervised("circle", noiseless=True)
    # train_HIO_zern_selfsupervised("triangle", noiseless=True)
