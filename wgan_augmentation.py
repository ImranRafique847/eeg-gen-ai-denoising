"""
Conditional WGAN-GP for motor imagery EEG data augmentation.
Generates class-conditioned synthetic EEG trials to supplement real training data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from eeg_dataset import EEGDenoisingDataset
from metrics import evaluate_all


class MIDataset(Dataset):
    """Synthetic Motor Imagery dataset with round-robin class labels."""

    def __init__(self, n_samples: int = 400, n_channels: int = 64,
                 signal_length: int = 256, n_classes: int = 4, seed: int = 0):
        import numpy as np
        from eeg_dataset import generate_synthetic_eeg

        self.n_classes = n_classes
        _, noisy = generate_synthetic_eeg(
            n_samples=n_samples, n_channels=n_channels,
            signal_length=signal_length, artifact_prob=0.0, seed=seed
        )
        labels = torch.arange(n_samples) % n_classes
        self.data   = noisy
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class ConditionalGenerator(nn.Module):
    """
    Noise + class label embedding -> synthetic EEG trial.
    Class embedding is concatenated with the noise vector before decoding.
    """

    def __init__(self, n_channels: int = 64, signal_length: int = 256,
                 latent_dim: int = 128, n_classes: int = 4,
                 base_dim: int = 64):
        super().__init__()
        self.n_channels    = n_channels
        self.signal_length = signal_length
        self.latent_dim    = latent_dim

        self.class_embed = nn.Embedding(n_classes, latent_dim)

        self._init_len = signal_length // 8
        self._init_ch  = base_dim * 4

        self.fc = nn.Linear(latent_dim * 2, self._init_ch * self._init_len)

        self.net = nn.Sequential(
            nn.ConvTranspose1d(self._init_ch, base_dim * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(base_dim * 2, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(base_dim * 2, base_dim,
                               kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(base_dim, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose1d(base_dim, base_dim // 2,
                               kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm1d(base_dim // 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(base_dim // 2, n_channels, kernel_size=3, padding=1),
            nn.Tanh(),
        )

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cls_emb = self.class_embed(labels)          # (B, latent_dim)
        inp = torch.cat([z, cls_emb], dim=1)        # (B, latent_dim*2)
        h = self.fc(inp).view(-1, self._init_ch, self._init_len)
        out = self.net(h)
        if out.shape[-1] != self.signal_length:
            out = F.interpolate(out, size=self.signal_length, mode="linear",
                                align_corners=False)
        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ConditionalDiscriminator(nn.Module):
    """
    Scores EEG trial conditioned on class label.
    Label is embedded as an extra spatial channel (projection discriminator trick).
    """

    def __init__(self, n_channels: int = 64, signal_length: int = 256,
                 n_classes: int = 4, base_dim: int = 64):
        super().__init__()
        self.n_channels    = n_channels
        self.signal_length = signal_length

        self.class_embed = nn.Embedding(n_classes, signal_length)

        self.net = nn.Sequential(
            nn.Conv1d(n_channels + 1, base_dim,
                      kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_dim, base_dim * 2,
                      kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(base_dim * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_dim * 2, base_dim * 4,
                      kernel_size=5, stride=2, padding=2),
            nn.InstanceNorm1d(base_dim * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(base_dim * 4, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cls_channel = self.class_embed(labels).unsqueeze(1)  # (B,1,L)
        x_cond = torch.cat([x, cls_channel], dim=1)          # (B,C+1,L)
        return self.net(x_cond).squeeze(-1)                   # (B,1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def cond_gradient_penalty(discriminator, real, fake,
                           labels, device) -> torch.Tensor:
    """WGAN-GP gradient penalty for the conditional discriminator."""
    B = real.shape[0]
    eps = torch.rand(B, 1, 1, device=device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    score = discriminator(interp, labels)
    grads = torch.autograd.grad(
        outputs=score, inputs=interp,
        grad_outputs=torch.ones_like(score),
        create_graph=True, retain_graph=True,
    )[0]
    grads = grads.reshape(B, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


WGAN_AUG_CONFIGS = {
    "tiny":   {"latent_dim": 64,  "base_dim": 32},
    "small":  {"latent_dim": 128, "base_dim": 64},
    "medium": {"latent_dim": 256, "base_dim": 128},
}


def build_wgan_generator(config_name: str, n_channels: int = 64,
                          signal_length: int = 256,
                          n_classes: int = 4) -> ConditionalGenerator:
    cfg = WGAN_AUG_CONFIGS[config_name]
    return ConditionalGenerator(n_channels=n_channels,
                                signal_length=signal_length,
                                n_classes=n_classes, **cfg)


def build_wgan_discriminator(config_name: str, n_channels: int = 64,
                              signal_length: int = 256,
                              n_classes: int = 4) -> ConditionalDiscriminator:
    cfg = WGAN_AUG_CONFIGS[config_name]
    return ConditionalDiscriminator(n_channels=n_channels,
                                    signal_length=signal_length,
                                    n_classes=n_classes,
                                    base_dim=cfg["base_dim"])


def train_wgan_aug(
    config_name: str = "small",
    n_epochs: int = 50,
    n_classes: int = 4,
    batch_size: int = 16,
    n_critic: int = 5,
    lambda_gp: float = 10.0,
    lr: float = 1e-4,
    n_train: int = 400,
    n_channels: int = 64,
    signal_length: int = 256,
    seed: int = 0,
    verbose: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = WGAN_AUG_CONFIGS[config_name]

    G = build_wgan_generator(config_name, n_channels, signal_length,
                              n_classes).to(device)
    D = build_wgan_discriminator(config_name, n_channels, signal_length,
                                  n_classes).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.0, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.0, 0.9))

    ds = MIDataset(n_samples=n_train, n_channels=n_channels,
                   signal_length=signal_length, n_classes=n_classes, seed=seed)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    history = []
    for epoch in range(1, n_epochs + 1):
        G.train(); D.train()
        run_g = run_d = 0.0; n_batches = 0

        for real, labels in loader:
            real, labels = real.to(device), labels.to(device)
            B = real.shape[0]

            for _ in range(n_critic):
                z = torch.randn(B, cfg["latent_dim"], device=device)
                with torch.no_grad():
                    fake = G(z, labels)
                d_real = D(real, labels)
                d_fake = D(fake.detach(), labels)
                gp = cond_gradient_penalty(D, real, fake, labels, device)
                d_loss = -(d_real.mean() - d_fake.mean()) + lambda_gp * gp
                opt_d.zero_grad(); d_loss.backward(); opt_d.step()

            z = torch.randn(B, cfg["latent_dim"], device=device)
            fake = G(z, labels)
            g_loss = -D(fake, labels).mean()
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()

            run_g += g_loss.item(); run_d += d_loss.item()
            n_batches += 1

        record = {
            "epoch": epoch,
            "g_loss": run_g / n_batches,
            "d_loss": run_d / n_batches,
        }
        history.append(record)

        if verbose and (epoch % 5 == 0 or epoch <= 3):
            print(
                f"[cWGAN-{config_name}] epoch {epoch:3d}/{n_epochs} "
                f"| g_loss={run_g/n_batches:.3f} "
                f"d_loss={run_d/n_batches:.3f}"
            )

    return G, history


@torch.no_grad()
def spectral_eeg_quality(
    generator: "ConditionalGenerator",
    real_data: torch.Tensor,
    n_classes: int,
    device: str = "cpu",
    n_synthetic: int = 100,
) -> dict:
    """
    Assess generated EEG quality by comparing power spectral densities.
    Returns spectral_corr, band_power_error, and amplitude_ratio.
    """
    import numpy as np

    generator.eval()
    latent_dim = generator.latent_dim
    B_real = real_data.shape[0]
    n_per_cls = n_synthetic // n_classes

    syn_list = []
    for cls in range(n_classes):
        z = torch.randn(n_per_cls, latent_dim, device=device)
        lbl = torch.full((n_per_cls,), cls, dtype=torch.long, device=device)
        syn_list.append(generator(z, lbl).cpu())
    synthetic = torch.cat(syn_list)  # (n_synthetic, C, L)

    real = real_data[:B_real].cpu()

    def mean_psd(x: torch.Tensor) -> np.ndarray:
        x_np = x.numpy().reshape(-1, x.shape[-1])  # (N*C, L)
        psds = np.abs(np.fft.rfft(x_np, axis=-1)) ** 2
        return psds.mean(axis=0)

    psd_real = mean_psd(real)
    psd_syn  = mean_psd(synthetic)

    psd_r_c = psd_real - psd_real.mean()
    psd_s_c = psd_syn  - psd_syn.mean()
    spectral_corr = float(
        (psd_r_c * psd_s_c).sum() /
        (np.linalg.norm(psd_r_c) * np.linalg.norm(psd_s_c) + 1e-10)
    )

    L = real.shape[-1]
    fs = 256
    freqs = np.fft.rfftfreq(L, d=1.0 / fs)
    bands = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13),
             "beta": (13, 30), "gamma": (30, 50)}
    band_errors = []
    for _, (lo, hi) in bands.items():
        idx = np.where((freqs >= lo) & (freqs < hi))[0]
        if len(idx) == 0:
            continue
        p_r = psd_real[idx].mean()
        p_s = psd_syn[idx].mean()
        band_errors.append(abs(p_r - p_s) / (p_r + 1e-10))
    band_power_error = float(np.mean(band_errors))

    rms_real = float(real.pow(2).mean().sqrt())
    rms_syn  = float(synthetic.pow(2).mean().sqrt())
    amplitude_ratio = rms_syn / (rms_real + 1e-10)

    return {
        "spectral_corr":    spectral_corr,
        "band_power_error": band_power_error,
        "amplitude_ratio":  amplitude_ratio,
    }


@torch.no_grad()
def generate_synthetic_trials(
    generator: ConditionalGenerator,
    n_per_class: int,
    n_classes: int,
    device: str = "cuda",
) -> tuple:
    """Generate n_per_class synthetic trials per class. Returns (data, labels)."""
    generator.eval()
    cfg_latent = generator.latent_dim
    all_data, all_labels = [], []
    for cls in range(n_classes):
        z = torch.randn(n_per_class, cfg_latent, device=device)
        lbl = torch.full((n_per_class,), cls, dtype=torch.long, device=device)
        synthetic = generator(z, lbl)
        all_data.append(synthetic.cpu())
        all_labels.append(lbl.cpu())
    return torch.cat(all_data), torch.cat(all_labels)


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(
        description="Train conditional WGAN-GP for MI EEG augmentation."
    )
    parser.add_argument("--model",     default="small",
                        choices=list(WGAN_AUG_CONFIGS.keys()))
    parser.add_argument("--epochs",    type=int, default=50)
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--quick",     action="store_true")
    args = parser.parse_args()

    if args.quick:
        config, epochs, n_train = "tiny", 5, 40
        print("Running in --quick mode\n")
    else:
        config, epochs, n_train = args.model, args.epochs, 400

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: cWGAN-{config}  epochs={epochs}  n_classes={args.n_classes}\n")

    G = build_wgan_generator(config, n_classes=args.n_classes)
    print(f"Generator parameters: {G.count_parameters():,}\n")

    from wgan_augmentation import MIDataset
    ds = MIDataset(n_samples=n_train, n_classes=args.n_classes)

    trained_G, history = train_wgan_aug(
        config_name=config, n_epochs=epochs,
        n_classes=args.n_classes, n_train=n_train,
    )

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    syn_data, syn_labels = generate_synthetic_trials(
        trained_G, n_per_class=10, n_classes=args.n_classes, device=device_str
    )
    print(f"\nGenerated synthetic data shape: {syn_data.shape}")
    print(f"Labels: {syn_labels[:12].tolist()}...")

    trained_G.to(device_str)
    quality = spectral_eeg_quality(
        trained_G, ds.data, n_classes=args.n_classes, device=device_str
    )
    print("\nSpectral quality metrics (synthetic vs real):")
    print(f"  spectral_corr    : {quality['spectral_corr']:.4f}  (ideal -> 1.0)")
    print(f"  band_power_error : {quality['band_power_error']:.4f}  (ideal -> 0.0)")
    print(f"  amplitude_ratio  : {quality['amplitude_ratio']:.4f}  (ideal -> 1.0)")

    os.makedirs("results", exist_ok=True)
    save_path = f"results/wgan_aug_{config}_trained.pt"
    torch.save(trained_G.state_dict(), save_path)
    print(f"\nSaved to: {save_path}")
