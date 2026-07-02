"""
Pareto frontier plot: latency vs SNR for all trained models.
Produces results/plots/pareto_frontier.png
"""

import os, time
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from torch.utils.data import DataLoader

from eeg_dataset import EEGDenoisingDataset
from metrics import evaluate_all

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("results/plots", exist_ok=True)

val_ds = EEGDenoisingDataset(
    n_samples=80, n_channels=64, signal_length=256,
    artifact_prob=0.6, seed=999
)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

val_ds_art = EEGDenoisingDataset(
    n_samples=80, n_channels=64, signal_length=256,
    artifact_prob=1.0, seed=999
)
val_loader_art = DataLoader(val_ds_art, batch_size=16, shuffle=False)


def measure_latency_ms(model, n_warmup=20, n_runs=100):
    """Returns (mean_ms, p95_ms) for single-trial inference."""
    model = model.to(device).eval()
    dummy = torch.randn(1, 64, 256, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model.denoise(dummy) if hasattr(model, "denoise") else model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model.denoise(dummy) if hasattr(model, "denoise") else model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.percentile(times, 95))


def eval_snr(model, loader):
    model.eval()
    snrs = []
    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            out = model.denoise(noisy) if hasattr(model, "denoise") else model(noisy)
            if isinstance(out, tuple): out = out[0]
            snrs.append(evaluate_all(clean, out)["snr_db"])
    return float(np.mean(snrs))


print("Loading and evaluating all models...\n")

entries = []

# GAN original
try:
    from gan_artifact_removal import build_generator
    m = build_generator("small").to(device)
    m.load_state_dict(torch.load("results/gan_small_trained.pt", map_location=device))
    lat, p95 = measure_latency_ms(m)
    snr = eval_snr(m, val_loader_art)
    entries.append({"label": "GAN-small\n(original)", "latency": lat, "p95": p95,
                    "snr": snr, "color": "#e67e22", "marker": "o",
                    "task": "Artifact Removal", "params": sum(p.numel() for p in m.parameters())})
    print(f"  GAN-small          | {lat:.2f}ms | SNR={snr:.2f}dB")
except Exception as e:
    print(f"  GAN-small skipped: {e}")

# DHCT-GAN upgraded
try:
    from dhct_gan import build_dhct_generator
    m = build_dhct_generator("small").to(device)
    m.load_state_dict(torch.load("results/dhct_gan_small_trained.pt", map_location=device))
    lat, p95 = measure_latency_ms(m)
    snr = eval_snr(m, val_loader_art)
    entries.append({"label": "DHCT-GAN-small\n(upgraded)", "latency": lat, "p95": p95,
                    "snr": snr, "color": "#c0392b", "marker": "D",
                    "task": "Artifact Removal", "params": sum(p.numel() for p in m.parameters())})
    print(f"  DHCT-GAN-small     | {lat:.2f}ms | SNR={snr:.2f}dB")
except Exception as e:
    print(f"  DHCT-GAN skipped: {e}")

# RCVAE
try:
    from rcvae import build_rcvae
    m = build_rcvae("small").to(device)
    m.load_state_dict(torch.load("results/rcvae_small_trained.pt", map_location=device))
    lat, p95 = measure_latency_ms(m)
    snr = eval_snr(m, val_loader)
    entries.append({"label": "RCVAE-small\n(general denoise)", "latency": lat, "p95": p95,
                    "snr": snr, "color": "#2980b9", "marker": "s",
                    "task": "General Denoising", "params": sum(p.numel() for p in m.parameters())})
    print(f"  RCVAE-small        | {lat:.2f}ms | SNR={snr:.2f}dB")
except Exception as e:
    print(f"  RCVAE skipped: {e}")

# Denoiseformer
try:
    from denoiseformer import build_denoiseformer, random_segment_mask
    m = build_denoiseformer("small").to(device)
    m.load_state_dict(torch.load("results/denoiseformer_small_trained.pt", map_location=device))

    class WrappedDenoiseformer(nn.Module):
        def __init__(self, model): super().__init__(); self.model = model
        def denoise(self, x):
            masked, mask = random_segment_mask(x, mask_ratio=0.25)
            return self.model(masked, mask)

    wrapped = WrappedDenoiseformer(m)
    lat, p95 = measure_latency_ms(wrapped)
    m.eval(); snrs = []
    with torch.no_grad():
        for _, clean in val_loader:
            clean = clean.to(device)
            masked, mask = random_segment_mask(clean, mask_ratio=0.25)
            recon = m(masked, mask)
            snrs.append(evaluate_all(clean, recon)["snr_db"])
    snr = float(np.mean(snrs))
    entries.append({"label": "Denoiseformer-small\n(missing repair)", "latency": lat, "p95": p95,
                    "snr": snr, "color": "#8e44ad", "marker": "^",
                    "task": "Missing Repair", "params": sum(p.numel() for p in m.parameters())})
    print(f"  Denoiseformer-small| {lat:.2f}ms | SNR={snr:.2f}dB")
except Exception as e:
    print(f"  Denoiseformer skipped: {e}")

# CompactVAE variants
try:
    from edge_deployment import build_compact_vae
    for cfg_name in ["nano", "micro", "mini"]:
        path = f"results/compact_vae_{cfg_name}_trained.pt"
        if not os.path.exists(path): continue
        m = build_compact_vae(cfg_name).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        lat, p95 = measure_latency_ms(m)
        snr = eval_snr(m, val_loader)
        entries.append({"label": f"CompactVAE-{cfg_name}\n(edge)", "latency": lat, "p95": p95,
                        "snr": snr, "color": "#27ae60", "marker": "v",
                        "task": "Edge Deployment", "params": sum(p.numel() for p in m.parameters())})
        print(f"  CompactVAE-{cfg_name:5s}   | {lat:.2f}ms | SNR={snr:.2f}dB")
except Exception as e:
    print(f"  CompactVAE skipped: {e}")

# baseline
val_snrs = []
with torch.no_grad():
    for noisy, clean in val_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        val_snrs.append(evaluate_all(clean, noisy)["snr_db"])
baseline_snr = float(np.mean(val_snrs))
print(f"\n  Baseline (no denoise)          | SNR={baseline_snr:.2f}dB")


def compute_pareto(points):
    """Lower latency + higher SNR = Pareto-optimal."""
    sorted_pts = sorted(points, key=lambda p: p[0])
    pareto = []
    max_snr = -float("inf")
    for lat, snr, idx in sorted_pts:
        if snr >= max_snr:
            pareto.append((lat, snr, idx))
            max_snr = snr
    return pareto


points = [(e["latency"], e["snr"], i) for i, e in enumerate(entries)]
pareto = compute_pareto(points)
pareto_indices = {p[2] for p in pareto}

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor("#0f1117")
for ax in axes:
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.grid(True, alpha=0.2, color="#555", linestyle="--")

# left: latency vs SNR scatter
ax = axes[0]
for i, e in enumerate(entries):
    is_pareto = i in pareto_indices
    size      = 180 if is_pareto else 100
    alpha     = 1.0 if is_pareto else 0.55
    edgecolor = "white" if is_pareto else "#888"
    lw        = 2.0 if is_pareto else 0.8
    ax.scatter(e["latency"], e["snr"],
               color=e["color"], marker=e["marker"],
               s=size, alpha=alpha, edgecolors=edgecolor, linewidths=lw, zorder=5)
    ax.annotate(e["label"],
                xy=(e["latency"], e["snr"]),
                xytext=(6, 4), textcoords="offset points",
                fontsize=7.5, color="white", alpha=0.9)

if len(pareto) > 1:
    px = [p[0] for p in pareto]
    py = [p[1] for p in pareto]
    ax.plot(px, py, color="#f1c40f", linewidth=1.8,
            linestyle="--", alpha=0.8, label="Pareto frontier", zorder=4)

ax.axhline(baseline_snr, color="#aaa", linestyle=":", linewidth=1.2,
           alpha=0.7, label=f"Baseline SNR ({baseline_snr:.2f}dB)")

for budget, label in [(10, "10ms\n(real-time)"), (50, "50ms\n(near RT)")]:
    ax.axvline(budget, color="#e74c3c", linestyle="-.", linewidth=1, alpha=0.5)
    ax.text(budget + 0.3, ax.get_ylim()[0] if ax.get_ylim()[0] > -5 else -5,
            label, color="#e74c3c", fontsize=7.5, alpha=0.8, va="bottom")

ax.set_xlabel("Inference Latency — single trial (ms)", fontsize=11)
ax.set_ylabel("SNR (dB)", fontsize=11)
ax.set_title("Latency vs Quality — Pareto Frontier\nEEG Generative AI Denoising",
             fontsize=12, fontweight="bold")
ax.legend(facecolor="#1a1d27", edgecolor="#555", labelcolor="white",
          fontsize=8, loc="lower right")

# right: bar chart — SNR by model
ax2 = axes[1]
labels  = [e["label"].replace("\n", " ") for e in entries]
snrs    = [e["snr"] for e in entries]
colors  = [e["color"] for e in entries]
alphas  = [1.0 if i in pareto_indices else 0.55 for i in range(len(entries))]

bars = ax2.barh(range(len(entries)), snrs, color=colors,
                edgecolor="white", linewidth=0.5)
for bar, alpha in zip(bars, alphas):
    bar.set_alpha(alpha)

ax2.set_yticks(range(len(entries)))
ax2.set_yticklabels(labels, fontsize=8.5, color="white")
ax2.axvline(baseline_snr, color="#aaa", linestyle=":", linewidth=1.5,
            alpha=0.8, label=f"Baseline ({baseline_snr:.2f}dB)")
ax2.axvline(0, color="#555", linewidth=0.8)

for i, e in enumerate(entries):
    ax2.text(e["snr"] + 0.05, i, f"{e['snr']:.2f}dB  ({e['latency']:.1f}ms)",
             va="center", ha="left", fontsize=8, color="white", alpha=0.9)

ax2.set_xlabel("SNR (dB)", fontsize=11)
ax2.set_title("SNR by Model\n(opacity = Pareto-optimal)", fontsize=12, fontweight="bold")
ax2.legend(facecolor="#1a1d27", edgecolor="#555", labelcolor="white", fontsize=8)
ax2.invert_yaxis()

task_colors = {
    "Artifact Removal":  "#e67e22",
    "General Denoising": "#2980b9",
    "Missing Repair":    "#8e44ad",
    "Edge Deployment":   "#27ae60",
}
patches = [mpatches.Patch(color=c, label=t) for t, c in task_colors.items()]
fig.legend(handles=patches, loc="lower center", ncol=4,
           facecolor="#1a1d27", edgecolor="#555", labelcolor="white",
           fontsize=9, bbox_to_anchor=(0.5, -0.04))

plt.suptitle("EEG Generative AI Denoising — Objective 1 Complete\n"
             "Latency vs Quality Trade-off Across All Models",
             color="white", fontsize=13, fontweight="bold", y=1.01)

plt.tight_layout()
plt.savefig("results/plots/pareto_frontier.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print("\nSaved: results/plots/pareto_frontier.png")

print("\n" + "=" * 75)
print(f"{'MODEL':<28} {'TASK':<20} {'SNR':>7} {'LATENCY':>9} {'PARETO'}")
print("=" * 75)
for i, e in enumerate(entries):
    pareto_mark = "YES" if i in pareto_indices else ""
    print(f"  {e['label'].replace(chr(10),' '):26} {e['task']:<20} "
          f"{e['snr']:>6.2f}dB {e['latency']:>7.2f}ms  {pareto_mark}")
print("=" * 75)

print(f"\nLatency budgets:")
for budget in [5, 10, 20, 50]:
    eligible = [(e['snr'], e['label'].replace('\n',' '), e['latency'])
                for e in entries if e['latency'] <= budget]
    if eligible:
        best = max(eligible, key=lambda x: x[0])
        print(f"  <={budget:3d}ms budget -> best: {best[1]:30s} "
              f"SNR={best[0]:.2f}dB  latency={best[2]:.2f}ms")
    else:
        print(f"  <={budget:3d}ms budget -> no model fits")


def main():
    pass  # all work done at module level above
