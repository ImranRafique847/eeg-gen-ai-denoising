"""
Edge deployment: compact VAE for real-time EEG denoising plus latency benchmarking.
Targets <50ms inference on CPU, <5ms on GPU for a single 256-sample trial.
"""

import time
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eeg_dataset import EEGDenoisingDataset
from metrics import evaluate_all


class CompactVAE(nn.Module):
    """
    Minimal VAE for edge inference. Depthwise separable convolutions,
    no batch norm, small latent dim, residual noise prediction.
    """

    def __init__(
        self,
        n_channels: int = 64,
        signal_length: int = 256,
        latent_dim: int = 8,
        base_dim: int = 16,
    ):
        super().__init__()
        self.latent_dim   = latent_dim
        self.n_channels   = n_channels
        self.signal_length = signal_length

        # depthwise separable + strided conv encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(n_channels, n_channels, kernel_size=7,
                      padding=3, groups=n_channels, bias=False),
            nn.Conv1d(n_channels, base_dim, kernel_size=1, bias=False),
            nn.ELU(),
            nn.Conv1d(base_dim, base_dim * 2, kernel_size=5,
                      stride=2, padding=2, bias=False),
            nn.InstanceNorm1d(base_dim * 2, affine=True),
            nn.ELU(),
            nn.Conv1d(base_dim * 2, base_dim * 2, kernel_size=3,
                      padding=1, bias=False),
            nn.InstanceNorm1d(base_dim * 2, affine=True),
            nn.ELU(),
        )
        enc_ch = base_dim * 2

        # per-timestep latent projections (convolutional, NOT FC)
        self.conv_mean   = nn.Conv1d(enc_ch, latent_dim, kernel_size=1)
        self.conv_logvar = nn.Conv1d(enc_ch, latent_dim, kernel_size=1)
        self.conv_decode = nn.Conv1d(latent_dim, enc_ch, kernel_size=1)

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(enc_ch, base_dim, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.InstanceNorm1d(base_dim, affine=True),
            nn.ELU(),
            nn.Conv1d(base_dim, base_dim, kernel_size=3,
                      padding=1, bias=False),
            nn.ELU(),
            nn.Conv1d(base_dim, n_channels, kernel_size=1, bias=False),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.conv_mean(h), self.conv_logvar(h)

    def decode(self, z):
        h = self.conv_decode(z)
        out = self.decoder(h)
        if out.shape[-1] != self.signal_length:
            out = F.interpolate(out, size=self.signal_length,
                                mode="linear", align_corners=False)
        return out

    def forward(self, x):
        """Residual prediction: output = input - predicted_noise."""
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv.clamp(-4, 4)) \
            if self.training else mu
        noise_pred = self.decode(z)
        return x - noise_pred, mu, lv

    def denoise(self, x):
        """Single-pass deterministic denoising (inference mode)."""
        self.eval()
        with torch.no_grad():
            recon, _, _ = self.forward(x)
        return recon

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


EDGE_CONFIGS = {
    "nano":  {"latent_dim": 4,  "base_dim": 8},
    "micro": {"latent_dim": 8,  "base_dim": 16},
    "mini":  {"latent_dim": 16, "base_dim": 24},
}


def build_compact_vae(config_name: str, n_channels: int = 64,
                      signal_length: int = 256) -> CompactVAE:
    cfg = EDGE_CONFIGS[config_name]
    return CompactVAE(n_channels=n_channels, signal_length=signal_length,
                      **cfg)


def train_compact_vae(
    config_name: str = "micro",
    n_epochs: int = 20,
    batch_size: int = 32,
    lr: float = 2e-4,
    beta: float = 0.01,
    beta_warmup: int = 5,
    n_train: int = 400,
    n_val: int = 80,
    n_channels: int = 64,
    signal_length: int = 256,
    seed: int = 0,
    verbose: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_compact_vae(config_name, n_channels, signal_length).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.1
    )

    train_ds = EEGDenoisingDataset(
        n_samples=n_train, n_channels=n_channels,
        signal_length=signal_length, artifact_prob=0.6, seed=seed
    )
    val_ds = EEGDenoisingDataset(
        n_samples=n_val, n_channels=n_channels,
        signal_length=signal_length, artifact_prob=0.6, seed=seed + 1
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    free_bits = 0.1   # min KL per latent dim before penalising (matches RCVAE)

    history = []
    for epoch in range(1, n_epochs + 1):
        current_beta = beta * min(1.0, epoch / max(beta_warmup, 1))
        model.train()
        run_loss = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            recon, mu, lv = model(noisy)
            recon_loss = F.mse_loss(recon, clean)
            kl_per_dim = -0.5 * (1 + lv - mu.pow(2) - lv.exp())
            kl_loss = kl_per_dim.clamp(min=free_bits).mean()
            loss = recon_loss + current_beta * kl_loss
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            run_loss += loss.item()
        scheduler.step()

        model.eval()
        val_m = {"snr_db": [], "mse": [], "ssim": [], "pearson": []}
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                recon, _, _ = model(noisy)
                m = evaluate_all(clean, recon)
                for k, v in m.items():
                    val_m[k].append(v)
        val_avg = {k: sum(v)/len(v) for k, v in val_m.items()}

        record = {"epoch": epoch, "loss": run_loss/len(train_loader), **val_avg}
        history.append(record)

        if verbose:
            print(
                f"[CompactVAE-{config_name}] epoch {epoch:3d}/{n_epochs} "
                f"| loss={run_loss/len(train_loader):.4f} "
                f"| val SNR={val_avg['snr_db']:.2f}dB "
                f"SSIM={val_avg['ssim']:.4f}"
            )

    return model, history


def measure_latency(model: nn.Module, n_channels: int = 64,
                    signal_length: int = 256,
                    batch_size: int = 1,
                    n_warmup: int = 10, n_runs: int = 50,
                    device: str = "cuda") -> dict:
    """Measure single-trial inference latency with GPU sync for accurate timing."""
    model = model.to(device).eval()
    dummy = torch.randn(batch_size, n_channels, signal_length, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model.denoise(dummy) if hasattr(model, "denoise") else model(dummy)

    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            out = model.denoise(dummy) if hasattr(model, "denoise") else model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)  # ms

    times_t = torch.tensor(times)
    return {
        "mean_ms":   times_t.mean().item(),
        "std_ms":    times_t.std().item(),
        "min_ms":    times_t.min().item(),
        "p95_ms":    times_t.quantile(0.95).item(),
        "n_params":  sum(p.numel() for p in model.parameters()),
    }


def benchmark_all_models(n_channels: int = 64, signal_length: int = 256,
                          device: str = "cuda"):
    """Benchmark inference latency for every trained model in results/."""
    results = []

    model_registry = [
        ("CompactVAE-nano",  lambda: build_compact_vae("nano"),  "results/compact_vae_nano_trained.pt"),
        ("CompactVAE-micro", lambda: build_compact_vae("micro"), "results/compact_vae_micro_trained.pt"),
        ("CompactVAE-mini",  lambda: build_compact_vae("mini"),  "results/compact_vae_mini_trained.pt"),
    ]

    try:
        from rcvae import build_rcvae
        for sz in ["tiny", "small", "medium"]:
            p = f"results/rcvae_{sz}_trained.pt"
            model_registry.append((f"RCVAE-{sz}", lambda s=sz: build_rcvae(s), p))
    except ImportError:
        pass

    try:
        from gan_artifact_removal import build_generator
        for sz in ["tiny", "small"]:
            p = f"results/gan_{sz}_trained.pt"
            model_registry.append((f"GAN-{sz}", lambda s=sz: build_generator(s), p))
    except ImportError:
        pass

    val_ds = EEGDenoisingDataset(
        n_samples=50, n_channels=n_channels,
        signal_length=signal_length, artifact_prob=0.6, seed=999
    )
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    for label, build_fn, weights_path in model_registry:
        if not os.path.exists(weights_path):
            print(f"  {label:25s} — no weights found, skipping")
            continue

        try:
            model = build_fn()
            model.load_state_dict(
                torch.load(weights_path, map_location="cpu")
            )
            model = model.to(device).eval()

            lat = measure_latency(model, n_channels, signal_length,
                                   batch_size=1, device=device)

            snrs = []
            with torch.no_grad():
                for noisy, clean in val_loader:
                    noisy, clean = noisy.to(device), clean.to(device)
                    if hasattr(model, "denoise"):
                        recon = model.denoise(noisy)
                    else:
                        out = model(noisy)
                        recon = out[0] if isinstance(out, tuple) else out
                    snrs.append(evaluate_all(clean, recon)["snr_db"])

            r = {
                "model":       label,
                "latency_ms":  lat["mean_ms"],
                "latency_p95": lat["p95_ms"],
                "snr_db":      sum(snrs) / len(snrs),
                "n_params":    lat["n_params"],
            }
            results.append(r)

            print(
                f"  {label:25s} | "
                f"latency={lat['mean_ms']:6.2f}ms (p95={lat['p95_ms']:6.2f}ms) | "
                f"SNR={sum(snrs)/len(snrs):6.2f}dB | "
                f"params={lat['n_params']:,}"
            )
        except Exception as e:
            print(f"  {label:25s} — ERROR: {e}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Edge deployment: compact VAE training + latency benchmark."
    )
    parser.add_argument("--train",          action="store_true")
    parser.add_argument("--benchmark_all",  action="store_true")
    parser.add_argument("--model",          default="micro",
                        choices=list(EDGE_CONFIGS.keys()))
    parser.add_argument("--epochs",         type=int, default=20)
    parser.add_argument("--quick",          action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    os.makedirs("results", exist_ok=True)

    if args.train or args.quick:
        configs_to_train = (["nano"] if args.quick
                            else list(EDGE_CONFIGS.keys()))
        epochs = 2 if args.quick else args.epochs
        n_train = 40 if args.quick else 400
        n_val   = 16 if args.quick else 80

        for cfg_name in configs_to_train:
            model = build_compact_vae(cfg_name)
            print(f"CompactVAE-{cfg_name}: {model.count_parameters():,} params")

            trained, history = train_compact_vae(
                config_name=cfg_name, n_epochs=epochs,
                n_train=n_train, n_val=n_val,
            )
            save_path = f"results/compact_vae_{cfg_name}_trained.pt"
            torch.save(trained.state_dict(), save_path)
            print(f"  Saved to {save_path}")

            lat = measure_latency(trained, device=device)
            print(f"  Inference latency: {lat['mean_ms']:.2f}ms "
                  f"(p95={lat['p95_ms']:.2f}ms)\n")

    if args.benchmark_all:
        print("\n" + "=" * 70)
        print("LATENCY vs QUALITY BENCHMARK — All Trained Models")
        print("=" * 70)
        results = benchmark_all_models(device=device)

        if results:
            print("\n--- Real-time budget analysis ---")
            for budget in [5, 10, 20, 50]:
                eligible = [r for r in results if r["latency_ms"] <= budget]
                if eligible:
                    best = max(eligible, key=lambda r: r["snr_db"])
                    print(
                        f"  Budget {budget:3d}ms: best={best['model']} "
                        f"SNR={best['snr_db']:.2f}dB "
                        f"latency={best['latency_ms']:.2f}ms"
                    )
                else:
                    print(f"  Budget {budget:3d}ms: no model fits")

    if not args.train and not args.benchmark_all and not args.quick:
        print("Demo: train nano CompactVAE (2 epochs) + latency test")
        model = build_compact_vae("nano")
        print(f"CompactVAE-nano: {model.count_parameters():,} params")
        trained, _ = train_compact_vae("nano", n_epochs=2, n_train=40, n_val=16)
        lat = measure_latency(trained, device=device)
        print(f"Inference latency: {lat['mean_ms']:.3f}ms "
              f"(p95={lat['p95_ms']:.3f}ms)")
        print("\nFor full benchmark: python edge_deployment.py --benchmark_all")
