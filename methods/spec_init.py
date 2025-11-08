import sys

sys.path.append(".")
import torch
from matplotlib import pyplot as plt
from utils.fourier import ft2, ift2
import math
import numpy as np


def spectral_init(meas, its=10):
    x_i = torch.randn(*meas.shape)
    x_i = x_i / torch.norm(x_i)
    for i in range(its):
        x_i_last = x_i * 1.0
        x_i_temp = ift2((1 - 1 / meas) * ft2(x_i))
        x_i = x_i_temp / torch.norm(x_i_temp)
        if i % 100 == 0:
            print(i, torch.norm(x_i - x_i_last))
    return x_i


def wirt_flow(meas, w, its=10):
    # x_i = torch.randn(*meas.shape) + 1j * torch.randn(*meas.shape)
    # x_i = x_i / torch.norm(x_i)
    x_i = spectral_init(meas, its=50)
    x_i[:w] = 0
    x_i[-w:] = 0
    x_i[:, :w] = 0
    x_i[:, -w:] = 0
    for i in range(its):
        x_i_last = x_i * 1.0
        # dfx = ift2(
        #     (torch.abs(ft2(x_i)) ** 2 - meas) * (ft2(x_i)) / torch.abs(ft2(x_i)) ** 2
        # )
        dfx = ift2(
            (torch.abs(ft2(x_i)) ** 1 - meas) * (ft2(x_i)) / torch.abs(ft2(x_i)) ** 1
        )
        x_i = x_i - dfx * np.min([1 - math.exp(-i / 330), 0.2])
        x_i[:w] = 0
        x_i[-w:] = 0
        x_i[:, :w] = 0
        x_i[:, -w:] = 0
        if i % 100 == 0:
            print(i, torch.norm(dfx))
    return x_i


def wirtinger_flow_fft(y, image_shape, w, max_iter=1000, step_size=0.1):
    """
    Implements the Wirtinger Flow algorithm for phase retrieval using 2D FFT for image processing.

    Parameters:
    - y: Magnitude measurements (H, W)  (assuming m = n for DFT matrix)
    - image_shape: Tuple (H, W) representing the shape of the image
    - max_iter: Maximum number of iterations
    - step_size: Gradient descent step size

    Returns:
    - x_est: Estimated image (H, W)
    """
    # Convert inputs to PyTorch tensors
    y = torch.tensor(y, dtype=torch.float32)
    H, W = image_shape

    # Initialization via spectral method
    z0 = torch.randn(
        (H, W), dtype=torch.complex64
    )  # Random initialization in the complex domain
    # z0 = spectral_init(y, its=200)
    z0[:w] = 0
    z0[-w:] = 0
    z0[:, :w] = 0
    z0[:, -w:] = 0

    # Gradient Descent Loop
    z = z0.clone()
    for i in range(max_iter):
        # Compute 2D FFT and IFFT for the DFT matrix operations
        Az = ft2(z)  # Forward 2D FFT simulates multiplication by the DFT matrix
        grad = ift2(
            (torch.abs(Az) ** 2 - y) * Az / torch.abs(Az) ** 2
        )  # Gradient computation
        z -= step_size * grad  # Gradient descent update
        z[:w] = 0
        z[-w:] = 0
        z[:, :w] = 0
        z[:, -w:] = 0
        if i % 10 == 0:
            print(i, torch.norm(y - torch.abs(Az) ** 2))

    return z
