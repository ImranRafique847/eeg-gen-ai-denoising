"""
Load every trained model and evaluate on the same held-out validation set.
Groups results by task for apples-to-apples comparison.
"""

import os
import torch
from torch.utils.data import DataLoader
from eeg_dataset import EEGDenoisingDataset
from metrics import evaluate_all

device = "cuda" if torch.cuda.is_available() else "cpu"

# shared validation set — same seed for all models
val_ds = EEGDenoisingDataset(
    n_samples=80, n_channels=64, signal_length=256,
    artifact_prob=0.6, seed=999
)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

# artifact-only set for GAN models
val_ds_art = EEGDenoisingDataset(
    n_samples=80, n_channels=64, signal_length=256,
    artifact_prob=1.0, seed=999
)
val_loader_art = DataLoader(val_ds_art, batch_size=16, shuffle=False)


def evaluate_model(model, loader):
    model.eval()
    m_all = {"snr_db": [], "mse": [], "ssim": [], "pearson": []}
    with torch.no_grad():
        for noisy, clean in loader:
            noisy, clean = noisy.to(device), clean.to(device)
            if hasattr(model, "denoise"):
                out = model.denoise(noisy)
            else:
                out = model(noisy)
                if isinstance(out, tuple):
                    out = out[0]
            m = evaluate_all(clean, out)
            for k, v in m.items():
                m_all[k].append(v)
    return {k: sum(v)/len(v) for k, v in m_all.items()}


results = []

# artifact removal
for size in ("tiny", "small"):
    weight = f"results/gan_{size}_trained.pt"
    try:
        from gan_artifact_removal import build_generator
        model = build_generator(size, n_channels=64).to(device)
        model.load_state_dict(torch.load(weight, map_location=device))
        m = evaluate_model(model, val_loader_art)
        results.append({"model": f"GAN-{size}", "group": "A. Artifact Removal",
                         "trained": "loaded", **m})
        print(f"loaded GAN-{size}")
    except Exception as e:
        print(f"failed GAN-{size}: {e}")

for size in ("tiny", "small"):
    weight = f"results/dhct_gan_{size}_trained.pt"
    try:
        from dhct_gan import build_dhct_generator
        model = build_dhct_generator(size, n_channels=64).to(device)
        model.load_state_dict(torch.load(weight, map_location=device))
        m = evaluate_model(model, val_loader_art)
        results.append({"model": f"DHCT-GAN-{size}", "group": "A. Artifact Removal",
                         "trained": "loaded", **m})
        print(f"loaded DHCT-GAN-{size}")
    except Exception as e:
        print(f"failed DHCT-GAN-{size}: {e}")

# general denoising
for size in ("tiny", "small"):
    weight = f"results/rcvae_{size}_trained.pt"
    try:
        from rcvae import build_rcvae
        model = build_rcvae(size, n_channels=64).to(device)
        model.load_state_dict(torch.load(weight, map_location=device))
        m = evaluate_model(model, val_loader)
        results.append({"model": f"RCVAE-{size}", "group": "B. General Denoising",
                         "trained": "loaded", **m})
        print(f"loaded RCVAE-{size}")
    except Exception as e:
        print(f"failed RCVAE-{size}: {e}")

for size in ("nano", "micro", "mini"):
    weight = f"results/compact_vae_{size}_trained.pt"
    try:
        from edge_deployment import build_compact_vae
        model = build_compact_vae(size, n_channels=64).to(device)
        model.load_state_dict(torch.load(weight, map_location=device))
        m = evaluate_model(model, val_loader)
        results.append({"model": f"CompactVAE-{size}", "group": "B. General Denoising",
                         "trained": "loaded", **m})
        print(f"loaded CompactVAE-{size}")
    except Exception as e:
        print(f"failed CompactVAE-{size}: {e}")

# signal inpainting
for size in ("tiny", "small"):
    weight = f"results/denoiseformer_{size}_trained.pt"
    try:
        from denoiseformer import build_denoiseformer, random_segment_mask
        model = build_denoiseformer(size, n_channels=64).to(device)
        model.load_state_dict(torch.load(weight, map_location=device))
        model.eval()
        m_all = {"snr_db": [], "mse": [], "ssim": [], "pearson": []}
        with torch.no_grad():
            for _, clean in val_loader:
                clean = clean.to(device)
                masked, mask = random_segment_mask(clean, mask_ratio=0.25)
                recon = model(masked, mask)
                m = evaluate_all(clean, recon)
                for k, v in m.items():
                    m_all[k].append(v)
        m = {k: sum(v)/len(v) for k, v in m_all.items()}
        results.append({"model": f"Denoiseformer-{size}", "group": "C. Signal Inpainting",
                         "trained": "loaded", **m})
        print(f"loaded Denoiseformer-{size}")
    except Exception as e:
        print(f"failed Denoiseformer-{size}: {e}")

# baseline: no denoising
m_all = {"snr_db": [], "mse": [], "ssim": [], "pearson": []}
with torch.no_grad():
    for noisy, clean in val_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        m = evaluate_all(clean, noisy)
        for k, v in m.items():
            m_all[k].append(v)
baseline = {k: sum(v)/len(v) for k, v in m_all.items()}
results.append({"model": "Baseline (no denoising)", "group": "—",
                 "trained": "—", **baseline})

print("\n")
print("=" * 95)
print(f"{'MODEL':<22} {'GROUP':<26} {'SNR':>7} {'SSIM':>7} {'MSE':>8} {'PEARSON':>8}  STATUS")
print("=" * 95)

trained = [r for r in results if r["group"] != "—"]
baseline_row = [r for r in results if r["group"] == "—"]

prev_group = None
for r in sorted(trained, key=lambda x: (x["group"], -x["snr_db"])):
    if r["group"] != prev_group:
        if prev_group is not None:
            print()
        prev_group = r["group"]
    print(
        f"{r['model']:<22} {r['group']:<26} "
        f"{r['snr_db']:>7.2f} {r['ssim']:>7.4f} "
        f"{r['mse']:>8.4f} {r['pearson']:>8.4f}  {r['trained']}"
    )

if baseline_row:
    b = baseline_row[0]
    print()
    print(
        f"{b['model']:<22} {b['group']:<26} "
        f"{b['snr_db']:>7.2f} {b['ssim']:>7.4f} "
        f"{b['mse']:>8.4f} {b['pearson']:>8.4f}  {b['trained']}"
    )
print("=" * 95)

print()
groups = sorted(set(r["group"] for r in trained))
for g in groups:
    g_results = [r for r in trained if r["group"] == g]
    if not g_results:
        continue
    best = max(g_results, key=lambda r: r["snr_db"])
    print(f"  {g:<26} best SNR -> {best['model']} ({best['snr_db']:.2f} dB)")
