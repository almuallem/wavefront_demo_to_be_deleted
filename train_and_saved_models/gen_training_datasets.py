import sys

sys.path.append("..")
import torch
from utils.fourier import ft2, ift2
import matplotlib.pyplot as plt
from utils.psf import psf
from utils.pupil import polymask
from utils.zernike import zern_abb
from IPython.display import clear_output
from utils.zernike import zernike_gram_schmidt
from utils.dataset_prep import pupil_dataset


def gen_training_dataset(pupil_shape):
    L = 500#25000
    N = 128
    pupil, scale = pupil_dataset(pupil_shape, N)
    alphas = 4 * (torch.rand(L, 21) - 0.5)
    alphas[:, :3] = 0
    phases = zern_abb(
        N, alphas, 0.4681
    )  # 0.4681 for fairness across pupil shapes (explained in supp)
    psfs = psf(pupil, phases)
    save_for_val = 500

    tset = torch.utils.data.TensorDataset(
        psfs[: L - save_for_val],
        phases[: L - save_for_val],
    )
    vset = torch.utils.data.TensorDataset(
        psfs[L - save_for_val :],
        phases[L - save_for_val :],
    )

    torch.save(tset, f"../datasets/y_phi_{pupil_shape}_{N}_train.pt")
    torch.save(vset, f"../datasets/y_phi_{pupil_shape}_{N}_val.pt")


if __name__ == "__main__":
    gen_training_dataset("circle")
    gen_training_dataset("triangle")
    gen_training_dataset("pentagon")

