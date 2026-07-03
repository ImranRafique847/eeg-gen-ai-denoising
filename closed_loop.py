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

    denoiser, classifier_noisy = load_models(device)

    # also load the classifier trained on denoised data
    classifier_denoised = build_classifier("tiny", N_CHANNELS, SIGNAL_LEN, N_CLASSES)
    denoised_ckpt = "results/classifier_tiny_denoised_trained.pt"
    if os.path.exists(denoised_ckpt):
        classifier_denoised.load_state_dict(
            torch.load(denoised_ckpt, map_location=device)
        )
        classifier_denoised = classifier_denoised.to(device).eval()
        has_denoised_clf = True
    else:
        has_denoised_clf = False

    # generate test trials from properly labeled MI dataset
    from eeg_dataset import MIDatasetLabeled
    ds = MIDatasetLabeled(
        n_samples=N_TEST, n_channels=N_CHANNELS,
        signal_length=SIGNAL_LEN, n_classes=N_CLASSES,
        artifact_prob=0.6, seed=77
    )
    loader = DataLoader(
        torch.utils.data.TensorDataset(ds.data, ds.labels),
        batch_size=32, shuffle=False
    )

    acc_noisy_list      = []
    acc_clean_noisy_clf = []   # denoised input, classifier trained on noisy
    acc_clean_dn_clf    = []   # denoised input, classifier trained on denoised
    snr_before_list     = []
    snr_after_list      = []

    # we need clean signal for SNR — generate matched clean data
    from eeg_dataset import generate_mi_eeg
    clean_data, _ = generate_mi_eeg(
        n_samples=N_TEST, n_channels=N_CHANNELS, signal_length=SIGNAL_LEN,
        n_classes=N_CLASSES, snr_db=40.0, artifact_prob=0.0, seed=77
    )

    with torch.no_grad():
        clean_idx = 0
        for noisy_batch, lbl_batch in loader:
            B = noisy_batch.shape[0]
            noisy_batch = noisy_batch.to(device)
            lbl_batch   = lbl_batch.to(device)
            clean_batch = clean_data[clean_idx:clean_idx+B].to(device)
            clean_idx  += B

            logits_noisy = classifier_noisy(noisy_batch)
            acc_noisy_list.append(
                (logits_noisy.argmax(1) == lbl_batch).float().mean().item()
            )

            cleaned = denoiser.denoise(noisy_batch)

            logits_cn = classifier_noisy(cleaned)
            acc_clean_noisy_clf.append(
                (logits_cn.argmax(1) == lbl_batch).float().mean().item()
            )

            if has_denoised_clf:
                logits_cd = classifier_denoised(cleaned)
                acc_clean_dn_clf.append(
                    (logits_cd.argmax(1) == lbl_batch).float().mean().item()
                )

            snr_before_list.append(evaluate_all(clean_batch, noisy_batch)["snr_db"])
            snr_after_list.append(evaluate_all(clean_batch, cleaned)["snr_db"])

    acc_noisy  = float(np.mean(acc_noisy_list))      * 100
    acc_cn_clf = float(np.mean(acc_clean_noisy_clf)) * 100
    snr_before = float(np.mean(snr_before_list))
    snr_after  = float(np.mean(snr_after_list))

    lat_mean, lat_p95 = measure_e2e_latency(denoiser, classifier_noisy, device)
    dummy = torch.randn(1, N_CHANNELS, SIGNAL_LEN, device=device)
    with torch.no_grad():
        for _ in range(10): denoiser.denoise(dummy)
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            denoiser.denoise(dummy)
            if device == "cuda": torch.cuda.synchronize()
    lat_denoise = (time.perf_counter() - t0) / 100 * 1000

    with torch.no_grad():
        for _ in range(10): classifier_noisy(dummy)
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            classifier_noisy(dummy)
            if device == "cuda": torch.cuda.synchronize()
    lat_classify = (time.perf_counter() - t0) / 100 * 1000

    print("=" * 60)
    print("  Closed-Loop BCI Simulation — Results")
    print("=" * 60)
    print(f"  Pipeline: noisy EEG -> CompactVAE-mini -> Classifier-tiny")
    print()
    print("  Classification accuracy (classifier trained on NOISY data):")
    print(f"    Input = noisy                    : {acc_noisy:.1f}%")
    print(f"    Input = denoised                 : {acc_cn_clf:.1f}%")
    print(f"    Improvement                      : {acc_cn_clf - acc_noisy:+.1f} pp")
    if has_denoised_clf:
        acc_cd_clf = float(np.mean(acc_clean_dn_clf)) * 100
        print()
        print("  Classification accuracy (classifier trained on DENOISED data):")
        print(f"    Input = denoised (matched clf)   : {acc_cd_clf:.1f}%")
        print(f"    vs noisy input + noisy clf       : {acc_cd_clf - acc_noisy:+.1f} pp")
    print()
    print("  Signal quality:")
    print(f"    SNR before denoising : {snr_before:.2f} dB")
    print(f"    SNR after denoising  : {snr_after:.2f} dB")
    print(f"    SNR improvement      : {snr_after - snr_before:+.2f} dB")
    print()
    print("  End-to-end latency (single trial, batch=1):")
    print(f"    Denoising step : {lat_denoise:.2f} ms")
    print(f"    Classification : {lat_classify:.2f} ms")
    print(f"    Total (mean)   : {lat_mean:.2f} ms")
    print(f"    Total (p95)    : {lat_p95:.2f} ms")
    print(f"    Real-time pass : {'PASS' if lat_mean < 20 else 'FAIL'} (<20ms)")
    print("=" * 60)

    os.makedirs("results", exist_ok=True)
    out_path = "results/closed_loop_results.txt"
    with open(out_path, "w") as f:
        f.write("Closed-Loop BCI Simulation Results\n")
        f.write("=" * 40 + "\n\n")
        f.write("Pipeline\n")
        f.write("  noisy EEG -> CompactVAE-mini -> EEGTransformerClassifier-tiny\n\n")
        f.write("Classifier trained on NOISY data\n")
        f.write(f"  accuracy (noisy input)    : {acc_noisy:.2f}%\n")
        f.write(f"  accuracy (denoised input) : {acc_cn_clf:.2f}%\n")
        f.write(f"  improvement               : {acc_cn_clf - acc_noisy:+.2f} pp\n\n")
        if has_denoised_clf:
            acc_cd_clf = float(np.mean(acc_clean_dn_clf)) * 100
            f.write("Classifier trained on DENOISED data (matched)\n")
            f.write(f"  accuracy (denoised input) : {acc_cd_clf:.2f}%\n")
            f.write(f"  vs baseline               : {acc_cd_clf - acc_noisy:+.2f} pp\n\n")
        f.write("Signal quality vs clean ground truth\n")
        f.write(f"  SNR before : {snr_before:.4f} dB\n")
        f.write(f"  SNR after  : {snr_after:.4f} dB\n")
        f.write(f"  SNR gain   : {snr_after - snr_before:+.4f} dB\n\n")
        f.write("End-to-end latency (single trial, batch=1)\n")
        f.write(f"  denoising step : {lat_denoise:.4f} ms\n")
        f.write(f"  classification : {lat_classify:.4f} ms\n")
        f.write(f"  total mean     : {lat_mean:.4f} ms\n")
        f.write(f"  total p95      : {lat_p95:.4f} ms\n")
        f.write(f"  real-time pass : {'yes' if lat_mean < 20 else 'no'} (<20ms)\n")
    print(f"\nSaved: {out_path}")

    return {"acc_noisy": acc_noisy, "acc_denoised_noisy_clf": acc_cn_clf,
            "snr_before": snr_before, "snr_after": snr_after,
            "lat_mean_ms": lat_mean}


if __name__ == "__main__":
    run_closed_loop()
