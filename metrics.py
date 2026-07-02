"""
Signal quality metrics for EEG denoising evaluation.
SNR, MSE, 1-D SSIM, Pearson correlation.
"""

import torch
import numpy as np


def compute_snr(clean: torch.Tensor, denoised: torch.Tensor) -> float:
    """SNR in dB: 10 * log10(signal_power / noise_power), averaged over batch."""
    signal_power = (clean ** 2).mean(dim=-1)              # (B, C)
    noise_power  = ((clean - denoised) ** 2).mean(dim=-1) # (B, C)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    return snr.mean().item()


def compute_mse(clean: torch.Tensor, denoised: torch.Tensor) -> float:
    """Mean Squared Error."""
    return ((clean - denoised) ** 2).mean().item()


def compute_ssim_1d(
    clean: torch.Tensor,
    denoised: torch.Tensor,
    window_size: int = 11,
    C1: float = 1e-4,
    C2: float = 9e-4,
) -> float:
    """1-D SSIM adapted for EEG signals using a sliding window."""
    B, C, L = clean.shape
    pad = window_size // 2

    # reshape to (B*C, 1, L) for grouped conv
    x = clean.reshape(B * C, 1, L)
    y = denoised.reshape(B * C, 1, L)

    kernel = torch.ones(1, 1, window_size, device=clean.device) / window_size

    mu_x  = torch.nn.functional.conv1d(x, kernel, padding=pad)
    mu_y  = torch.nn.functional.conv1d(y, kernel, padding=pad)
    mu_xx = torch.nn.functional.conv1d(x * x, kernel, padding=pad)
    mu_yy = torch.nn.functional.conv1d(y * y, kernel, padding=pad)
    mu_xy = torch.nn.functional.conv1d(x * y, kernel, padding=pad)

    sigma_x  = mu_xx - mu_x ** 2
    sigma_y  = mu_yy - mu_y ** 2
    sigma_xy = mu_xy - mu_x * mu_y

    numerator   = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    ssim_map = numerator / (denominator + 1e-10)
    return ssim_map.mean().item()


def compute_pearson(clean: torch.Tensor, denoised: torch.Tensor) -> float:
    """Mean Pearson correlation across all channels and batch."""
    c_mean = clean.mean(dim=-1, keepdim=True)
    d_mean = denoised.mean(dim=-1, keepdim=True)
    c_c = clean    - c_mean
    d_c = denoised - d_mean

    num = (c_c * d_c).sum(dim=-1)
    den = (c_c.norm(dim=-1) * d_c.norm(dim=-1) + 1e-10)
    return (num / den).mean().item()


def evaluate_all(clean: torch.Tensor, denoised: torch.Tensor) -> dict:
    """Run all quality metrics and return as a dict."""
    return {
        "snr_db":  compute_snr(clean, denoised),
        "mse":     compute_mse(clean, denoised),
        "ssim":    compute_ssim_1d(clean, denoised),
        "pearson": compute_pearson(clean, denoised),
    }


if __name__ == "__main__":
    B, C, L = 4, 64, 256
    clean    = torch.randn(B, C, L)
    denoised = clean + 0.1 * torch.randn_like(clean)
    metrics  = evaluate_all(clean, denoised)
    print("Metrics on slight noise:")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")
