"""
Objective 4 — Closed-loop BCI simulation.

Pipeline:
  noisy EEG -> CompactVAE-mini (denoise) -> EEGTransformerClassifier-tiny (classify)

Reports:
  - End-to-end latency (denoising + classification, single trial)
  - Classification accuracy on noisy input vs. denoised input
  - Shows whether the denoiser actually helps the classifier
"""

import os
import time
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from eeg_dataset import EEGDenoisingDataset
from edge_deployment import build_compact_vae
from eeg_transformer_classifier import build_classifier
from metrics import evaluate_all


N_CHANNELS  = 64
SIGNAL_LEN  = 256
N_CLASSES   = 4
N_TEST      = 200   # trials for the closed-loop evaluation


def load_models(device):
    denoiser = build_compact_vae("mini", N_CHANNELS, SIGNAL_LEN)
    denoiser.load_state_dict(
        torch.load("results/compact_vae_mini_trained.pt", map_location=device)
    )
    denoiser = denoiser.to(device).eval()

    classifier = build_classifier("tiny", N_CHANNELS, SIGNAL_LEN, N_CLASSES)
    classifier.load_state_dict(
        torch.load("results/classifier_tiny_trained.pt", map_location=device)
    )
    classifier = classifier.to(device).eval()

    return denoiser, classifier


def measure_e2e_latency(denoiser, classifier, device, n_warmup=20, n_runs=100):
    """Time the full pipeline: denoise + classify on a single trial."""
    dummy = torch.randn(1, N_CHANNELS, SIGNAL_LEN, device=device)

    # warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            cleaned = denoiser.denoise(dummy)
            classifier(cleaned)

    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            cleaned = denoiser.denoise(dummy)
            _ = classifier(cleaned)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

    return float(np.mean(times)), float(np.percentile(times, 95))


def evaluate_pipeline(denoiser, classifier, noisy, labels, device):
    """Return accuracy on noisy input and on denoised input."""
    noisy  = noisy.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        # classify directly from noisy signal
        logits_noisy = classifier(noisy)
        acc_noisy = (logits_noisy.argmax(1) == labels).float().mean().item()

        # denoise first, then classify
        cleaned = denoiser.denoise(noisy)
        logits_clean = classifier(cleaned)
        acc_clean = (logits_clean.argmax(1) == labels).float().mean().item()

        # signal quality improvement from denoising
        snr_before = evaluate_all(labels.float().unsqueeze(-1).expand_as(noisy),
                                   noisy)  # not meaningful for labels
        # use a separate EEG quality check instead
        quality_noisy  = evaluate_all(noisy, noisy)   # baseline (SNR = inf)
        quality_cleaned = None  # we don't have clean ground truth here

    return acc_noisy, acc_clean


def run_closed_loop():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    denoiser, classifier = load_models(device)

    # generate test trials (noisy EEG + labels)
    ds = EEGDenoisingDataset(
        n_samples=N_TEST, n_channels=N_CHANNELS,
        signal_length=SIGNAL_LEN, artifact_prob=0.6, seed=42
    )
    labels = torch.arange(N_TEST) % N_CLASSES
    loader = DataLoader(
        TensorDataset(ds.noisy, ds.clean, labels),
        batch_size=32, shuffle=False
    )

    # batch evaluation
    acc_noisy_list  = []
    acc_clean_list  = []
    snr_before_list = []
    snr_after_list  = []

    with torch.no_grad():
        for noisy_batch, clean_batch, lbl_batch in loader:
            noisy_batch  = noisy_batch.to(device)
            clean_batch  = clean_batch.to(device)
            lbl_batch    = lbl_batch.to(device)

            # classification on raw noisy
            logits_noisy = classifier(noisy_batch)
            acc_noisy_list.append(
                (logits_noisy.argmax(1) == lbl_batch).float().mean().item()
            )

            # denoise then classify
            cleaned = denoiser.denoise(noisy_batch)
            logits_clean = classifier(cleaned)
            acc_clean_list.append(
                (logits_clean.argmax(1) == lbl_batch).float().mean().item()
            )

            # signal quality (we have clean ground truth from the dataset)
            snr_before_list.append(evaluate_all(clean_batch, noisy_batch)["snr_db"])
            snr_after_list.append(evaluate_all(clean_batch, cleaned)["snr_db"])

    acc_noisy  = float(np.mean(acc_noisy_list))  * 100
    acc_clean  = float(np.mean(acc_clean_list))  * 100
    snr_before = float(np.mean(snr_before_list))
    snr_after  = float(np.mean(snr_after_list))

    # end-to-end latency (single trial)
    lat_mean, lat_p95 = measure_e2e_latency(denoiser, classifier, device)

    # individual step latencies
    dummy = torch.randn(1, N_CHANNELS, SIGNAL_LEN, device=device)

    # denoise only
    with torch.no_grad():
        for _ in range(10): denoiser.denoise(dummy)
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            denoiser.denoise(dummy)
            if device == "cuda": torch.cuda.synchronize()
    lat_denoise = (time.perf_counter() - t0) / 100 * 1000

    # classify only
    with torch.no_grad():
        for _ in range(10): classifier(dummy)
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            classifier(dummy)
            if device == "cuda": torch.cuda.synchronize()
    lat_classify = (time.perf_counter() - t0) / 100 * 1000

    # print report
    print("=" * 56)
    print("  Closed-Loop BCI Simulation — Results")
    print("=" * 56)
    print(f"  Pipeline: noisy EEG -> CompactVAE-mini -> Classifier-tiny")
    print()
    print("  Classification accuracy:")
    print(f"    Without denoising : {acc_noisy:.1f}%")
    print(f"    With denoising    : {acc_clean:.1f}%")
    print(f"    Improvement       : {acc_clean - acc_noisy:+.1f} pp")
    print()
    print("  Signal quality (vs clean ground truth):")
    print(f"    SNR before denoising : {snr_before:.2f} dB")
    print(f"    SNR after denoising  : {snr_after:.2f} dB")
    print(f"    SNR improvement      : {snr_after - snr_before:+.2f} dB")
    print()
    print("  End-to-end latency (single trial, batch=1):")
    print(f"    Denoising step    : {lat_denoise:.2f} ms")
    print(f"    Classification    : {lat_classify:.2f} ms")
    print(f"    Total (mean)      : {lat_mean:.2f} ms")
    print(f"    Total (p95)       : {lat_p95:.2f} ms")
    print(f"    Real-time budget  : {'PASS' if lat_mean < 20 else 'FAIL'} (<20ms)")
    print("=" * 56)

    # save to text file
    os.makedirs("results", exist_ok=True)
    out_path = "results/closed_loop_results.txt"
    with open(out_path, "w") as f:
        f.write("Closed-Loop BCI Simulation Results\n")
        f.write("=" * 40 + "\n\n")
        f.write("Pipeline\n")
        f.write("  noisy EEG -> CompactVAE-mini -> EEGTransformerClassifier-tiny\n\n")
        f.write("Classification accuracy\n")
        f.write(f"  without denoising : {acc_noisy:.2f}%\n")
        f.write(f"  with denoising    : {acc_clean:.2f}%\n")
        f.write(f"  improvement       : {acc_clean - acc_noisy:+.2f} pp\n\n")
        f.write("Signal quality vs clean ground truth\n")
        f.write(f"  SNR before : {snr_before:.4f} dB\n")
        f.write(f"  SNR after  : {snr_after:.4f} dB\n")
        f.write(f"  SNR gain   : {snr_after - snr_before:+.4f} dB\n\n")
        f.write("End-to-end latency (single trial, batch=1)\n")
        f.write(f"  denoising step : {lat_denoise:.4f} ms\n")
        f.write(f"  classification : {lat_classify:.4f} ms\n")
        f.write(f"  total mean     : {lat_mean:.4f} ms\n")
        f.write(f"  total p95      : {lat_p95:.4f} ms\n")
        f.write(f"  real-time pass : {'yes' if lat_mean < 20 else 'no'} (<20ms threshold)\n")
    print(f"\nSaved: {out_path}")

    return {
        "acc_noisy": acc_noisy,
        "acc_clean": acc_clean,
        "snr_before": snr_before,
        "snr_after":  snr_after,
        "lat_mean_ms": lat_mean,
        "lat_p95_ms":  lat_p95,
    }


if __name__ == "__main__":
    run_closed_loop()
