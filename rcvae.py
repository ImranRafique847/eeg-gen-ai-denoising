"""
Residual Convolutional VAE (RCVAE) for general EEG denoising.
Fully convolutional latent space; predicts noise residual to avoid posterior collapse.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eeg_dataset import EEGDenoisingDataset
from metrics import evaluate_all


class ResConvBlock1D(nn.Module):
    """Residual 1-D conv block: preserves high-frequency EEG oscillations."""

    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.InstanceNorm1d(channels, affine=True),
            nn.ELU(),
            nn.Conv1d(channels, channels, kernel_size, padding=pad),
            nn.InstanceNorm1d(channels, affine=True),
        )
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class DownBlock(nn.Module):
    """Strided conv to halve temporal resolution."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(out_ch, affine=True),
            nn.ELU(),
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    """Transposed conv to double temporal resolution."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose1d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(out_ch, affine=True),
            nn.ELU(),
        )

    def forward(self, x):
        return self.conv(x)


class RCVAE(nn.Module):
    """
    Residual Convolutional VAE — fully convolutional, no global pooling.

    Latent space is a feature map (B, latent_dim, L'), not a flat vector.
    Predicts noise to subtract (residual learning):
        output = input - predicted_noise
    """

    def __init__(
        self,
        n_channels: int = 64,
        signal_length: int = 256,
        base_dim: int = 32,
        latent_dim: int = 16,
        depth: int = 2,
        beta: float = 0.01,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.signal_length = signal_length
        self.latent_dim = latent_dim
        self.beta = beta

        enc = [nn.Conv1d(n_channels, base_dim, 3, padding=1), nn.ELU()]
        ch = base_dim
        for _ in range(depth):
            out_ch = min(ch * 2, 256)
            enc.append(ResConvBlock1D(ch))
            enc.append(DownBlock(ch, out_ch))
            ch = out_ch
        enc.append(ResConvBlock1D(ch))
        self.encoder = nn.Sequential(*enc)
        self._enc_ch = ch

        # per-timestep latent projections (convolutional, NOT FC)
        self.conv_mean   = nn.Conv1d(ch, latent_dim, 1)
        self.conv_logvar = nn.Conv1d(ch, latent_dim, 1)
        self.conv_decode = nn.Conv1d(latent_dim, ch, 1)

        dec = [ResConvBlock1D(ch)]
        for _ in range(depth):
            out_ch = max(ch // 2, base_dim)
            dec.append(UpBlock(ch, out_ch))
            dec.append(ResConvBlock1D(out_ch))
            ch = out_ch
        dec.append(nn.Conv1d(ch, n_channels, 3, padding=1))
        self.decoder = nn.Sequential(*dec)

    def encode(self, x):
        h = self.encoder(x)
        return self.conv_mean(h), self.conv_logvar(h)

    def reparameterise(self, mean, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar.clamp(-4, 4))
            return mean + std * torch.randn_like(std)
        return mean

    def decode(self, z):
        h = self.conv_decode(z)
        out = self.decoder(h)
        if out.shape[-1] != self.signal_length:
            out = F.interpolate(out, size=self.signal_length,
                                mode="linear", align_corners=False)
        return out

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterise(mean, logvar)
        noise_pred = self.decode(z)          # predict what to remove
        return x - noise_pred, mean, logvar  # residual: clean = noisy - noise

    def denoise(self, x):
        self.eval()
        with torch.no_grad():
            recon, _, _ = self.forward(x)
        return recon

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def rcvae_loss(recon, target, mean, logvar,
               beta=0.01, free_bits=0.1):
    """
    ELBO = MSE reconstruction + beta * KL with free bits per dim.
    free_bits prevents all latent dims collapsing to prior simultaneously.
    """
    recon_loss = F.mse_loss(recon, target)
    kl_per_dim = -0.5 * (1 + logvar - mean.pow(2) - logvar.exp())
    kl_loss    = kl_per_dim.clamp(min=free_bits).mean()
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


RCVAE_CONFIGS = {
    "tiny":   {"base_dim": 32, "latent_dim": 8,  "depth": 2},
    "small":  {"base_dim": 48, "latent_dim": 16, "depth": 2},
    "medium": {"base_dim": 64, "latent_dim": 32, "depth": 3},
    "large":  {"base_dim": 96, "latent_dim": 64, "depth": 3},
}


def build_rcvae(config_name, n_channels=64, signal_length=256,
                beta=0.01):
    cfg = RCVAE_CONFIGS[config_name]
    return RCVAE(n_channels=n_channels, signal_length=signal_length,
                 beta=beta, **cfg)


def train_rcvae(
    config_name="small",
    n_epochs=20,
    batch_size=16,
    lr=2e-4,
    beta=0.01,
    beta_warmup_epochs=5,
    free_bits=0.1,
    n_train=400,
    n_val=80,
    n_channels=64,
    signal_length=256,
    seed=0,
    verbose=True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_rcvae(config_name, n_channels, signal_length, beta).to(device)
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

    history = []
    for epoch in range(1, n_epochs + 1):
        current_beta = beta * min(1.0, epoch / max(beta_warmup_epochs, 1))

        model.train()
        run_loss = run_recon = run_kl = 0.0
        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            recon, mean, logvar = model(noisy)
            loss, recon_l, kl_l = rcvae_loss(
                recon, clean, mean, logvar, current_beta, free_bits
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            run_loss  += loss.item()
            run_recon += recon_l.item()
            run_kl    += kl_l.item()
        scheduler.step()
        n = len(train_loader)

        model.eval()
        val_m = {"snr_db": [], "mse": [], "ssim": [], "pearson": []}
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                recon, _, _ = model(noisy)
                m = evaluate_all(clean, recon)
                for k, v in m.items():
                    val_m[k].append(v)
        val_avg = {k: sum(v) / len(v) for k, v in val_m.items()}

        record = {
            "epoch": epoch,
            "loss": run_loss / n,
            "recon_loss": run_recon / n,
            "kl_loss": run_kl / n,
            **val_avg,
        }
        history.append(record)

        if verbose:
            print(
                f"[RCVAE-{config_name}] epoch {epoch:3d}/{n_epochs} "
                f"| beta={current_beta:.4f} "
                f"recon={run_recon/n:.4f} kl={run_kl/n:.4f} "
                f"| val SNR={val_avg['snr_db']:6.2f}dB "
                f"SSIM={val_avg['ssim']:.4f}"
            )

    return model, history


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(description="Train RCVAE for EEG denoising.")
    parser.add_argument("--model",  default="small",
                        choices=list(RCVAE_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--beta",   type=float, default=0.01)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--quick",  action="store_true")
    args = parser.parse_args()

    if args.quick:
        config, epochs, n_train, n_val = "tiny", 2, 40, 16
        print("Running in --quick mode\n")
    else:
        config, epochs, n_train, n_val = args.model, args.epochs, 400, 80

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Model  : RCVAE-{config}  epochs={epochs}  "
          f"beta={args.beta}  warmup={args.warmup}\n")

    model = build_rcvae(config)
    print(f"Parameters: {model.count_parameters():,}\n")

    trained, history = train_rcvae(
        config_name=config, n_epochs=epochs,
        beta=args.beta, beta_warmup_epochs=args.warmup,
        n_train=n_train, n_val=n_val,
    )

    print("\nFinal validation metrics:")
    final = history[-1]
    for k in ("snr_db", "mse", "ssim", "pearson"):
        print(f"  {k:10s}: {final[k]:.4f}")

    os.makedirs("results", exist_ok=True)
    save_path = f"results/rcvae_{config}_trained.pt"
    torch.save(trained.state_dict(), save_path)
    print(f"\nSaved to: {save_path}")
