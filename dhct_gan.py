"""
DHCT-GAN — Dual-branch CNN-Transformer hybrid GAN for EEG artifact removal.
Improves on the pure-CNN GAN by adding a Transformer branch in each generator arm.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eeg_dataset import EEGDenoisingDataset
from gan_artifact_removal import DualDiscriminator, gradient_penalty
from metrics import evaluate_all


class ConvBlock1D(nn.Module):
    """Conv1D + InstanceNorm + LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size // 2),
            nn.InstanceNorm1d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TransformerBlock1D(nn.Module):
    """
    Lightweight Transformer block over the temporal dimension.
    Captures long-range dependencies that Conv1D misses.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d_model, L) — switch to (B, L, d_model) for attention
        x_t = x.permute(0, 2, 1)
        attn_out, _ = self.attn(self.norm1(x_t), self.norm1(x_t),
                                self.norm1(x_t))
        x_t = x_t + attn_out
        x_t = x_t + self.ff(self.norm2(x_t))
        return x_t.permute(0, 2, 1)   # back to (B, d_model, L)


class HybridBranch(nn.Module):
    """
    Parallel CNN + Transformer streams fused by a learned sigmoid gate.
    CNN handles local patterns; Transformer handles global context.
    """

    def __init__(self, in_ch: int, base_dim: int, depth: int,
                 n_heads: int = 4):
        super().__init__()

        conv_layers = [ConvBlock1D(in_ch, base_dim)]
        for _ in range(depth - 1):
            conv_layers.append(ConvBlock1D(base_dim, base_dim))
        self.conv_stream = nn.Sequential(*conv_layers)

        self.transformer_proj = nn.Conv1d(in_ch, base_dim, 1)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock1D(base_dim, n_heads)
            for _ in range(max(depth // 2, 1))
        ])

        self.gate = nn.Sequential(
            nn.Conv1d(base_dim * 2, base_dim, 1),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Conv1d(base_dim, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_feat = self.conv_stream(x)           # (B, base_dim, L)

        trans_feat = self.transformer_proj(x)     # (B, base_dim, L)
        for blk in self.transformer_blocks:
            trans_feat = blk(trans_feat)          # (B, base_dim, L)

        gate = self.gate(torch.cat([conv_feat, trans_feat], dim=1))
        fused = gate * conv_feat + (1 - gate) * trans_feat
        return self.out_proj(fused)               # (B, in_ch, L)


class DHCTGenerator(nn.Module):
    """
    Dual-branch Hybrid CNN-Transformer Generator.
    Same divide-and-fuse structure as DivideAndFuseGenerator but each
    branch uses HybridBranch (CNN + Transformer) instead of pure CNN.
    """

    def __init__(self, n_channels: int = 64, base_dim: int = 32,
                 depth: int = 3, n_heads: int = 4):
        super().__init__()

        self.clean_branch    = HybridBranch(n_channels, base_dim, depth, n_heads)
        self.artifact_branch = HybridBranch(n_channels, base_dim, depth, n_heads)

        self.fusion_gate = nn.Sequential(
            nn.Conv1d(n_channels * 2, base_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_dim, n_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        clean_est    = self.clean_branch(x)
        artifact_est = self.artifact_branch(x)

        gate = self.fusion_gate(torch.cat([clean_est, artifact_est], dim=1))
        return gate * clean_est + (1 - gate) * (x - artifact_est)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


DHCT_CONFIGS = {
    "tiny":   {"base_dim": 8,  "depth": 2, "n_heads": 2},
    "small":  {"base_dim": 16, "depth": 3, "n_heads": 4},
    "medium": {"base_dim": 32, "depth": 3, "n_heads": 4},
    "large":  {"base_dim": 64, "depth": 4, "n_heads": 8},
}


def build_dhct_generator(config_name: str,
                         n_channels: int = 64) -> DHCTGenerator:
    cfg = DHCT_CONFIGS[config_name]
    return DHCTGenerator(n_channels=n_channels, **cfg)


LAMBDA_L1 = 10.0
LAMBDA_GP  = 10.0
N_CRITIC   = 2
LR_G = 2e-4
LR_D = 2e-4
N_CHANNELS    = 64
SIGNAL_LENGTH = 256
BATCH_SIZE    = 16


def train_dhct(
    config_name: str = "small",
    n_epochs: int = 20,
    n_train: int = 400,
    n_val: int = 80,
    seed: int = 0,
    verbose: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    generator     = build_dhct_generator(config_name, N_CHANNELS).to(device)
    from gan_artifact_removal import build_discriminator
    discriminator = build_discriminator(config_name, N_CHANNELS).to(device)

    opt_g = torch.optim.Adam(generator.parameters(),     lr=LR_G, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=LR_D, betas=(0.5, 0.9))

    train_ds = EEGDenoisingDataset(
        n_samples=n_train, n_channels=N_CHANNELS,
        signal_length=SIGNAL_LENGTH, artifact_prob=1.0, seed=seed
    )
    val_ds = EEGDenoisingDataset(
        n_samples=n_val, n_channels=N_CHANNELS,
        signal_length=SIGNAL_LENGTH, artifact_prob=1.0, seed=seed + 1
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    l1_fn = nn.L1Loss()
    history = []

    for epoch in range(1, n_epochs + 1):
        generator.train(); discriminator.train()
        run_g = run_d = 0.0
        n_batches = 0

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)

            for _ in range(N_CRITIC):
                with torch.no_grad():
                    fake = generator(noisy)
                d_real_l, d_real_g = discriminator(clean)
                d_fake_l, d_fake_g = discriminator(fake)
                gp = gradient_penalty(discriminator, clean, fake, device)
                d_loss = (
                    -(d_real_l.mean() - d_fake_l.mean())
                    - (d_real_g.mean() - d_fake_g.mean())
                    + LAMBDA_GP * gp
                )
                opt_d.zero_grad(); d_loss.backward(); opt_d.step()

            fake = generator(noisy)
            d_fake_l, d_fake_g = discriminator(fake)
            adv  = -(d_fake_l.mean() + d_fake_g.mean())
            l1   = l1_fn(fake, clean)
            g_loss = adv + LAMBDA_L1 * l1
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()

            run_g += g_loss.item(); run_d += d_loss.item()
            n_batches += 1

        generator.eval()
        val_m = {"snr_db": [], "mse": [], "ssim": [], "pearson": []}
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                denoised = generator(noisy)
                m = evaluate_all(clean, denoised)
                for k, v in m.items():
                    val_m[k].append(v)
        val_avg = {k: sum(v) / len(v) for k, v in val_m.items()}

        record = {
            "epoch": epoch,
            "g_loss": run_g / n_batches,
            "d_loss": run_d / n_batches,
            **val_avg,
        }
        history.append(record)

        if verbose:
            print(
                f"[DHCT-{config_name}] epoch {epoch:3d}/{n_epochs} "
                f"| g={run_g/n_batches:.3f} d={run_d/n_batches:.3f} "
                f"| val SNR={val_avg['snr_db']:6.2f}dB "
                f"SSIM={val_avg['ssim']:.4f}"
            )

    return generator, history


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(
        description="Train DHCT-GAN for EEG artifact removal."
    )
    parser.add_argument("--model",  default="small",
                        choices=list(DHCT_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--quick",  action="store_true")
    args = parser.parse_args()

    if args.quick:
        config, epochs, n_train, n_val = "tiny", 2, 40, 16
        print("Running in --quick mode\n")
    else:
        config, epochs, n_train, n_val = args.model, args.epochs, 400, 80

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Model  : DHCT-GAN-{config}  epochs={epochs}\n")

    gen = build_dhct_generator(config)
    print(f"Generator parameters: {gen.count_parameters():,}\n")

    trained, history = train_dhct(
        config_name=config, n_epochs=epochs,
        n_train=n_train, n_val=n_val,
    )

    print("\nFinal validation metrics:")
    final = history[-1]
    for k in ("snr_db", "mse", "ssim", "pearson"):
        print(f"  {k:10s}: {final[k]:.4f}")

    os.makedirs("results", exist_ok=True)
    save_path = f"results/dhct_gan_{config}_trained.pt"
    torch.save(trained.state_dict(), save_path)
    print(f"\nSaved to: {save_path}")
