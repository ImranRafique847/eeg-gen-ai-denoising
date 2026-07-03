"""
Energy consumption and CPU latency profiling for edge EEG denoising models.
Uses nvidia-smi for GPU power measurement (works without nvml.dll).
"""

import os
import time
import subprocess
import json
import torch
import numpy as np
from edge_deployment import build_compact_vae
from metrics import evaluate_all
from eeg_dataset import EEGDenoisingDataset
from torch.utils.data import DataLoader


N_CHANNELS  = 64
SIGNAL_LEN  = 256


def read_gpu_power_w():
    """Read current GPU power draw via nvidia-smi (milliwatts -> watts)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        return float(out)
    except Exception:
        return None


def measure_gpu_latency_and_energy(model, n_warmup=20, n_runs=200):
    """
    Measure GPU inference latency and estimate energy per inference.
    Power is sampled via nvidia-smi before and after the timed loop.
    Energy = average_power * total_time / n_runs
    """
    device = "cuda"
    model = model.to(device).eval()
    dummy = torch.randn(1, N_CHANNELS, SIGNAL_LEN, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model.denoise(dummy)
    torch.cuda.synchronize()

    # Sample idle power before
    power_before = read_gpu_power_w()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model.denoise(dummy)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    # Sample power during the run (approximate)
    power_during = read_gpu_power_w()

    mean_ms = float(np.mean(times))
    std_ms  = float(np.std(times))
    p95_ms  = float(np.percentile(times, 95))

    # Energy estimate per single inference
    avg_power_w = power_during if power_during else 0.0
    energy_mj = avg_power_w * (mean_ms / 1000.0) * 1000  # millijoules

    return {
        "mean_ms":   mean_ms,
        "std_ms":    std_ms,
        "p95_ms":    p95_ms,
        "power_w":   avg_power_w,
        "energy_mj": energy_mj,
    }


def measure_cpu_latency(model, n_warmup=10, n_runs=100):
    """CPU inference latency — no GPU involved."""
    model = model.to("cpu").eval()
    dummy = torch.randn(1, N_CHANNELS, SIGNAL_LEN)

    with torch.no_grad():
        for _ in range(n_warmup):
            model.denoise(dummy)

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model.denoise(dummy)
            times.append((time.perf_counter() - t0) * 1000)

    return {
        "mean_ms": float(np.mean(times)),
        "std_ms":  float(np.std(times)),
        "p95_ms":  float(np.percentile(times, 95)),
    }


def profile_all():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Shared val set for quality measurement
    val_ds = EEGDenoisingDataset(
        n_samples=80, n_channels=N_CHANNELS, signal_length=SIGNAL_LEN,
        artifact_prob=0.6, seed=999
    )
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    all_results = []

    print("\n" + "=" * 66)
    print("  Energy & CPU Profiling — CompactVAE Edge Models")
    print("=" * 66)

    for config in ["nano", "micro", "mini"]:
        weight_path = f"results/compact_vae_{config}_trained.pt"
        if not os.path.exists(weight_path):
            print(f"  Skipping {config} — weights not found")
            continue

        model = build_compact_vae(config, N_CHANNELS, SIGNAL_LEN)
        model.load_state_dict(torch.load(weight_path, map_location="cpu"))

        n_params    = model.count_parameters()
        file_mb     = os.path.getsize(weight_path) / (1024 ** 2)

        gpu_stats   = measure_gpu_latency_and_energy(model)
        cpu_stats   = measure_cpu_latency(model)

        # Quality on val set
        model = model.to(device).eval()
        m_all = {"snr_db": [], "mse": [], "ssim": [], "pearson": []}
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                out = model.denoise(noisy)
                m = evaluate_all(clean, out)
                for k, v in m.items():
                    m_all[k].append(v)
        quality = {k: sum(v) / len(v) for k, v in m_all.items()}

        # Battery estimate: 1 Hz inference, 2000 mAh @ 3.7 V = 7.4 Wh
        battery_wh = 3.7 * 2.0
        energy_per_hour_wh = (gpu_stats["energy_mj"] / 1000.0) * 3600  # 1 Hz → 3600/hr
        battery_hours = battery_wh / energy_per_hour_wh if energy_per_hour_wh > 0 else 0

        result = {
            "model":          f"CompactVAE-{config}",
            "params":         n_params,
            "file_mb":        round(file_mb, 3),
            "gpu_latency_mean_ms": round(gpu_stats["mean_ms"], 3),
            "gpu_latency_std_ms":  round(gpu_stats["std_ms"],  3),
            "gpu_latency_p95_ms":  round(gpu_stats["p95_ms"],  3),
            "gpu_power_w":         round(gpu_stats["power_w"], 2),
            "energy_per_inference_mj": round(gpu_stats["energy_mj"], 4),
            "battery_life_hours_1hz":  round(battery_hours, 1),
            "cpu_latency_mean_ms": round(cpu_stats["mean_ms"], 2),
            "cpu_latency_p95_ms":  round(cpu_stats["p95_ms"],  2),
            "snr_db":   round(quality["snr_db"],   4),
            "ssim":     round(quality["ssim"],     4),
            "mse":      round(quality["mse"],      6),
            "pearson":  round(quality["pearson"],  4),
        }
        all_results.append(result)

        print(f"\n  CompactVAE-{config}")
        print(f"    Parameters          : {n_params:,}")
        print(f"    Weights file        : {file_mb:.3f} MB")
        print(f"    GPU latency (mean)  : {gpu_stats['mean_ms']:.2f} ms")
        print(f"    GPU latency (p95)   : {gpu_stats['p95_ms']:.2f} ms")
        print(f"    GPU power draw      : {gpu_stats['power_w']:.2f} W")
        print(f"    Energy / inference  : {gpu_stats['energy_mj']:.4f} mJ")
        print(f"    Battery life @ 1 Hz : {battery_hours:.1f} h  (2000 mAh, 3.7V)")
        print(f"    CPU latency (mean)  : {cpu_stats['mean_ms']:.2f} ms")
        print(f"    CPU latency (p95)   : {cpu_stats['p95_ms']:.2f} ms")
        print(f"    SNR                 : {quality['snr_db']:.4f} dB")
        print(f"    SSIM                : {quality['ssim']:.4f}")

    print("\n" + "=" * 66)

    # Save JSON
    os.makedirs("results", exist_ok=True)
    out_path = "results/energy_cpu_profile.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {out_path}")

    # Print comparison table
    print("\n  Summary table")
    print(f"  {'Model':<20} {'Params':>8} {'GPU ms':>8} {'CPU ms':>8} {'Energy mJ':>10} {'Batt h':>8} {'SNR dB':>8}")
    print("  " + "-" * 78)
    for r in all_results:
        print(f"  {r['model']:<20} {r['params']:>8,} "
              f"{r['gpu_latency_mean_ms']:>8.2f} "
              f"{r['cpu_latency_mean_ms']:>8.2f} "
              f"{r['energy_per_inference_mj']:>10.4f} "
              f"{r['battery_life_hours_1hz']:>8.1f} "
              f"{r['snr_db']:>8.4f}")

    return all_results


if __name__ == "__main__":
    profile_all()
