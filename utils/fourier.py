import torch


def ft2(signals, dim=(-1, -2), device="cpu"):
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(signals.to(device), dim=dim), dim=dim),
        dim=dim,
    )


def ift2(signals, dim=(-1, -2), device="cpu"):
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(signals.to(device), dim=dim), dim=dim),
        dim=dim,
    )
