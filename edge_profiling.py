"""
Objective 3 — Full edge profile for CompactVAE-mini.
Reports latency, file size, parameter count, and peak GPU memory.
"""

import os
import time
import torch
import numpy as np

from edge_deployment import build_compact_vae
from metrics import evaluate_all
from eeg_dataset import EEGDenoisingDataset
from torch.utils.data import DataLoader


WEIGHTS_PATH = "results/compact_vae_mini_trained.pt"
N_CHANNELS   = 64
SIGNAL_LEN   = 256


def measure_latency(model, device, n_warmup=20, n_runs=100):
    model = model.to(device).eval()
    dummy = torch.randn(1, N_CHANNELS, SIGNAL_LEN, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model.denoise(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model.denoise(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return np.mean(times), np.std(times), np.percentile(times, 95)


def measure_peak_memory(model, device):
    if device != "cuda":
        return None
    model = model.to(device).eval()
    dummy = torch.randn(1, N_CHANNELS, SIGNAL_LEN, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        model.denoise(dummy)
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return peak_bytes / (1024 ** 2)  # MB


def run_profile():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_compact_vae("mini", N_CHANNELS, SIGNAL_LEN)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    model = model.to(device)

    # 1. Parameter count
    n_params = model.count_parameters()

    # 2. File size on disk
    file_size_mb = os.path.getsize(WEIGHTS_PATH) / (1024 ** 2)

    # 3. Latency
    mean_ms, std_ms, p95_ms = measure_latency(model, device)

    # 4. Peak GPU memory
    peak_mb = measure_peak_memory(model, device)

    # 5. Quality on held-out val set (confirm weights are good)
    val_ds = EEGDenoisingDataset(
        n_samples=80, n_channels=N_CHANNELS,
        signal_length=SIGNAL_LEN, artifact_prob=0.6, seed=999
    )
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)
    model.eval()
    m_all = {"snr_db": [], "mse": [], "ssim": [], "pearson": []}
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            out = model.denoise(noisy)
            m = evaluate_all(clean, out)
            for k, v in m.items():
                m_all[k].append(v)
    quality = {k: sum(v) / len(v) for k, v in m_all.items()}

    print("\n" + "=" * 50)
    print("  CompactVAE-mini — Edge Profile")
    print("=" * 50)
    print(f"  Parameters       : {n_params:,}")
    print(f"  Weights file     : {file_size_mb:.3f} MB  ({WEIGHTS_PATH})")
    print(f"  Latency (mean)   : {mean_ms:.2f} ms")
    print(f"  Latency (std)    : {std_ms:.2f} ms")
    print(f"  Latency (p95)    : {p95_ms:.2f} ms")
    if peak_mb is not None:
        print(f"  Peak GPU memory  : {peak_mb:.2f} MB")
    else:
        print(f"  Peak GPU memory  : N/A (CPU run)")
    print(f"  Device           : {device}")
    print()
    print("  Quality (val, artifact_prob=0.6):")
    print(f"    SNR    : {quality['snr_db']:.4f} dB")
    print(f"    SSIM   : {quality['ssim']:.4f}")
    print(f"    MSE    : {quality['mse']:.6f}")
    print(f"    Pearson: {quality['pearson']:.4f}")
    print("=" * 50)

    return {
        "n_params":     n_params,
        "file_size_mb": file_size_mb,
        "latency_mean_ms": mean_ms,
        "latency_std_ms":  std_ms,
        "latency_p95_ms":  p95_ms,
        "peak_gpu_mb":  peak_mb,
        **quality,
    }


if __name__ == "__main__":
    run_profile()
