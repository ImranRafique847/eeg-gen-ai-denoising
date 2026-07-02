"""
Denoiseformer — Transformer model for EEG missing-segment repair.
Multi-scale CNN front-end + slice-pattern attention for gap inpainting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from eeg_dataset import EEGDenoisingDataset
from metrics import evaluate_all


def random_segment_mask(
    x: torch.Tensor,
    mask_ratio: float = 0.25,
    n_segments: int = 2,
) -> tuple:
    """Zero out random contiguous segments; returns (masked_x, binary_mask)."""
    B, C, L = x.shape
    mask = torch.ones(B, 1, L, device=x.device)
    seg_len = int(L * mask_ratio / n_segments)
    seg_len = max(seg_len, 1)

    for b in range(B):
        for _ in range(n_segments):
            start = torch.randint(0, max(L - seg_len, 1), (1,)).item()
            mask[b, 0, start: start + seg_len] = 0.0

    masked = x * mask
    return masked, mask


class MultiScaleEncoder(nn.Module):
    """
    Three parallel conv streams at different kernel sizes (3, 7, 15)
    capture EEG patterns at short, medium, and long temporal scales.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        branch_ch = out_ch // 3
        extra = out_ch - branch_ch * 3

        def branch(k):
            return nn.Sequential(
                nn.Conv1d(in_ch, branch_ch, kernel_size=k, padding=k // 2),
                nn.InstanceNorm1d(branch_ch, affine=True),
                nn.GELU(),
            )

        self.branch_short  = branch(3)
        self.branch_medium = branch(7)
        self.branch_long   = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch + extra, kernel_size=15, padding=7),
            nn.InstanceNorm1d(branch_ch + extra, affine=True),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([
            self.branch_short(x),
            self.branch_medium(x),
            self.branch_long(x),
        ], dim=1)


class SlicePatternAttention(nn.Module):
    """
    Treats each temporal position as a token and runs multi-head
    self-attention across the time axis, letting masked positions attend
    to valid context on both sides.
    """

    def __init__(self, d_model: int, n_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class Denoiseformer(nn.Module):
    """
    EEG missing-segment repair model.
    Input:  (B, C+1, L) — masked EEG + binary mask channel
    Output: (B, C, L)   — full reconstructed signal
    """

    def __init__(
        self,
        n_channels: int = 64,
        signal_length: int = 256,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        base_dim: int = 64,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.signal_length = signal_length

        # input: masked EEG (C channels) + mask (1 channel) = C+1
        self.multiscale = MultiScaleEncoder(n_channels + 1, base_dim)
        self.input_proj = nn.Conv1d(base_dim, d_model, kernel_size=1)

        self.pos_embed = nn.Parameter(
            torch.zeros(1, signal_length, d_model)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.transformer = nn.ModuleList([
            SlicePatternAttention(d_model, n_heads)
            for _ in range(n_layers)
        ])
        self.transformer_norm = nn.LayerNorm(d_model)

        self.decoder = nn.Sequential(
            nn.Conv1d(d_model, base_dim, kernel_size=3, padding=1),
            nn.InstanceNorm1d(base_dim, affine=True),
            nn.GELU(),
            nn.Conv1d(base_dim, base_dim // 2, kernel_size=3, padding=1),
            nn.InstanceNorm1d(base_dim // 2, affine=True),
            nn.GELU(),
            nn.Conv1d(base_dim // 2, n_channels, kernel_size=3, padding=1),
        )

    def forward(self, masked_eeg: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        x = torch.cat([masked_eeg, mask], dim=1)         # (B, C+1, L)

        x = self.multiscale(x)                            # (B, base_dim, L)
        x = self.input_proj(x)                            # (B, d_model, L)

        x = x.permute(0, 2, 1) + self.pos_embed          # (B, L, d_model)

        # mask missing positions so they don't contaminate keys/values
        attn_mask = (mask.squeeze(1) == 0)                # (B, L)

        for layer in self.transformer:
            x = layer(x, key_padding_mask=attn_mask)
        x = self.transformer_norm(x)

        x = x.permute(0, 2, 1)
        return self.decoder(x)                            # (B, C, L)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


DENOISEFORMER_CONFIGS = {
    "tiny":   {"d_model": 64,  "n_heads": 2, "n_layers": 2, "base_dim": 32},
    "small":  {"d_model": 128, "n_heads": 4, "n_layers": 4, "base_dim": 64},
    "medium": {"d_model": 256, "n_heads": 8, "n_layers": 6, "base_dim": 128},
    "large":  {"d_model": 512, "n_heads": 8, "n_layers": 8, "base_dim": 128},
}


def build_denoiseformer(config_name: str, n_channels: int = 64,
                        signal_length: int = 256) -> Denoiseformer:
    cfg = DENOISEFORMER_CONFIGS[config_name]
    return Denoiseformer(n_channels=n_channels,
                         signal_length=signal_length, **cfg)


def train_denoiseformer(
    config_name: str = "small",
    n_epochs: int = 20,
    batch_size: int = 8,
    lr: float = 5e-4,
    mask_ratio: float = 0.25,
    n_segments: int = 2,
    n_train: int = 400,
    n_val: int = 80,
    n_channels: int = 64,
    signal_length: int = 256,
    seed: int = 0,
    verbose: bool = True,
):
    """Train Denoiseformer on the missing-segment repair task."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_denoiseformer(config_name, n_channels, signal_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.05
    )

    # train on clean data with synthetic masking — learns EEG structure
    train_ds = EEGDenoisingDataset(
        n_samples=n_train, n_channels=n_channels,
        signal_length=signal_length, artifact_prob=0.0, seed=seed
    )
    val_ds = EEGDenoisingDataset(
        n_samples=n_val, n_channels=n_channels,
        signal_length=signal_length, artifact_prob=0.0, seed=seed + 1
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    history = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        run_loss = 0.0
        for _, clean in train_loader:
            clean = clean.to(device)
            masked, mask = random_segment_mask(
                clean, mask_ratio=mask_ratio, n_segments=n_segments
            )
            recon = model(masked, mask)

            # loss only on masked positions — that's what we're learning
            masked_loss = F.mse_loss(
                recon * (1 - mask), clean * (1 - mask)
            )
            # small auxiliary loss on observed positions keeps output coherent
            obs_loss = F.mse_loss(recon * mask, clean * mask) * 0.1
            loss = masked_loss + obs_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            run_loss += loss.item()
        scheduler.step()
        n = len(train_loader)

        model.eval()
        val_metrics = {"snr_db": [], "mse": [], "ssim": [], "pearson": []}
        with torch.no_grad():
            for _, clean in val_loader:
                clean = clean.to(device)
                masked, mask = random_segment_mask(
                    clean, mask_ratio=mask_ratio, n_segments=n_segments
                )
                recon = model(masked, mask)
                m = evaluate_all(clean, recon)
                for k, v in m.items():
                    val_metrics[k].append(v)
        val_avg = {k: sum(v) / len(v) for k, v in val_metrics.items()}

        record = {"epoch": epoch, "loss": run_loss / n, **val_avg}
        history.append(record)

        if verbose:
            print(
                f"[Denoiseformer-{config_name}] epoch {epoch:3d}/{n_epochs} "
                f"| loss={run_loss/n:.4f} "
                f"| val SNR={val_avg['snr_db']:6.2f}dB "
                f"SSIM={val_avg['ssim']:.4f}"
            )

    return model, history


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(
        description="Train Denoiseformer for EEG missing-segment repair."
    )
    parser.add_argument("--model",  default="small",
                        choices=list(DENOISEFORMER_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--mask_ratio", type=float, default=0.25)
    parser.add_argument("--quick",  action="store_true")
    args = parser.parse_args()

    if args.quick:
        config, epochs, n_train, n_val = "tiny", 2, 40, 16
        print("Running in --quick mode\n")
    else:
        config, epochs, n_train, n_val = args.model, args.epochs, 400, 80

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Model  : Denoiseformer-{config}  epochs={epochs}  "
          f"mask_ratio={args.mask_ratio}\n")

    model = build_denoiseformer(config)
    print(f"Parameters: {model.count_parameters():,}\n")

    trained, history = train_denoiseformer(
        config_name=config, n_epochs=epochs,
        mask_ratio=args.mask_ratio,
        n_train=n_train, n_val=n_val,
    )

    print("\nFinal validation metrics:")
    final = history[-1]
    for k in ("snr_db", "mse", "ssim", "pearson"):
        print(f"  {k:10s}: {final[k]:.4f}")

    os.makedirs("results", exist_ok=True)
    save_path = f"results/denoiseformer_{config}_trained.pt"
    torch.save(trained.state_dict(), save_path)
    print(f"\nSaved to: {save_path}")
