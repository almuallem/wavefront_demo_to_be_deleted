import sys

sys.path.append(".")
import torch
from utils.fourier import ft2, ift2
from tqdm import tqdm

def gerchberg_saxton(sqrt_psf, pupil, max_iter, init_guess=None, device="cpu", return_predictions = False):
    phase_x = torch.rand(*sqrt_psf.shape, device=device) - 0.5
    amplitude_x = sqrt_psf.to(device) * 1.0
    phase_f = torch.rand(*sqrt_psf.shape, device=device) - 0.5
    amplitude_f = pupil.to(device) * 1.0

    signal_x = amplitude_x * torch.exp(1j * phase_x)
    
    if return_predictions:
        predictions = [] #To hold all the predicted phases

    for its in tqdm(range(max_iter)):
        signal_f = amplitude_f * ift2(signal_x, device=device)
        phase_f = torch.angle(signal_f)
        signal_f = amplitude_f * torch.exp(1j * phase_f)
        signal_x = ft2(signal_f, device=device)
        phase_x = torch.angle(signal_x)
        signal_x = amplitude_x * torch.exp(1j * phase_x)
        if return_predictions:
            predictions.append(phase_f * pupil)
    if return_predictions:

        return phase_f * pupil, predictions
    else:
        return phase_f * pupil


# def fienup_hio(sqrt_psf, pupil, max_iter, beta=0.5, init_guess=None, device="cpu"):
#     pupil = pupil.to(device) * 1.0
#     phase_f = torch.rand(*sqrt_psf.shape, device=device) - 0.5
#     sigf = pupil.to(device) * torch.exp(1j * phase_f)
#     sqrt_psf = sqrt_psf.to(device) * 1.0

#     sigx = sqrt_psf * torch.exp(1j * phase_f)
#     for its in range(max_iter):
#         sigf_ft = ft2(pupil * sigf, device=device)
#         sigx = sqrt_psf * torch.exp(1j * torch.angle(sigf_ft))
#         sigx_ift = ift2(sigx, device=device)
#         sigf = (
#             torch.where(
#                 ~torch.isclose(torch.abs(sigx_ift), pupil * 1.0),
#                 # sigf - beta * sigx_ift,
#                 # sigx_ift,
#                 sigx_ift,
#                 sigf - beta * sigx_ift,
#             )
#             * pupil
#         )
#     return torch.angle(sigf) * pupil


def fienup_hio(sqrt_psf, pupil, max_iter, beta=0.5, init_guess=None, device="cpu", return_predictions = False):
    # Ensure proper tensor setup
    pupil = pupil.to(device).float()
    sqrt_psf = sqrt_psf.to(device).float()

    # Initialize signal field
    if init_guess is None:
        phase = (
            torch.rand(*sqrt_psf.shape, device=device) * 2 * torch.pi - torch.pi
        )  # -π to π
        sigf = pupil * torch.exp(1j * phase)  # Constrain initial guess to support
    else:
        sigf = init_guess.to(device)

    # Main iteration loop

    predictions = []
    for _ in tqdm(range(max_iter)):
        # Fourier domain update
        sigf_ft = ft2(sigf, device=device)
        sigx = sqrt_psf * torch.exp(1j * torch.angle(sigf_ft))

        # Object domain update
        sigx_ift = ift2(sigx, device=device)

        # HIO core update rule
        sigf = torch.where(
            pupil.bool(),
            sigx_ift,  # Inside support: use direct update
            sigf - beta * sigx_ift,  # Outside support: apply feedback
        )
        if return_predictions:
            predictions.append(torch.angle(sigf) * pupil)

    if return_predictions:
        return torch.angle(sigf) * pupil, predictions
    else:
        return torch.angle(sigf) * pupil  # Return phase within support
