"""
GAN-based EEG artifact removal targeting EOG (eye-blink) and EMG (muscle) artifacts.
Dual-branch generator with WGAN-GP training for stable time-series adversarial training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    """Basic Conv1D + InstanceNorm + LeakyReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 5,
                 stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride=stride,
                      padding=padding),
            nn.InstanceNorm1d(out_ch, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DivideAndFuseGenerator(nn.Module):
    """
    Dual-branch generator: one branch for clean signal, one for artifact,
    combined via an adaptive fusion gate.
    """

    def __init__(self, n_channels: int = 64, base_dim: int = 32,
                 depth: int = 3):
        super().__init__()

        # branch 1: clean signal estimator
        clean_layers = [ConvBlock1D(n_channels, base_dim)]
        for _ in range(depth - 1):
            clean_layers.append(ConvBlock1D(base_dim, base_dim))
        clean_layers.append(nn.Conv1d(base_dim, n_channels, 3, padding=1))
        self.clean_branch = nn.Sequential(*clean_layers)

        # branch 2: artifact estimator
        artifact_layers = [ConvBlock1D(n_channels, base_dim)]
        for _ in range(depth - 1):
            artifact_layers.append(ConvBlock1D(base_dim, base_dim))
        artifact_layers.append(nn.Conv1d(base_dim, n_channels, 3, padding=1))
        self.artifact_branch = nn.Sequential(*artifact_layers)

        # learned gate: how much to trust clean estimate vs subtract artifact
        self.fusion_gate = nn.Sequential(
            nn.Conv1d(n_channels * 2, base_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_dim, n_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        clean_est = self.clean_branch(x)
        artifact_est = self.artifact_branch(x)

        gate_input = torch.cat([clean_est, artifact_est], dim=1)
        gate = self.fusion_gate(gate_input)

        fused = gate * clean_est + (1 - gate) * (x - artifact_est)
        return fused

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PatchDiscriminator1D(nn.Module):
    """
    1-D patch discriminator: outputs a score per local temporal patch.
    Encourages the generator to fix local structured artifacts rather than
    just matching global statistics.
    """

    def __init__(self, n_channels: int = 64, base_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock1D(n_channels, base_dim, kernel_size=7, stride=2),
            ConvBlock1D(base_dim, base_dim * 2, kernel_size=5, stride=2),
            ConvBlock1D(base_dim * 2, base_dim * 2, kernel_size=5, stride=1),
            nn.Conv1d(base_dim * 2, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, 1, L') per-patch realness score


class DualDiscriminator(nn.Module):
    """
    Two discriminators at different receptive fields.
    d_local catches sharp local artifacts; d_global catches broader statistical mismatches.
    """

    def __init__(self, n_channels: int = 64, base_dim: int = 32):
        super().__init__()
        self.d_local = PatchDiscriminator1D(n_channels, base_dim)
        self.d_global = nn.Sequential(
            ConvBlock1D(n_channels, base_dim, kernel_size=9, stride=4),
            ConvBlock1D(base_dim, base_dim * 2, kernel_size=7, stride=4),
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(base_dim * 2, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        local_score = self.d_local(x)
        global_score = self.d_global(x)
        return local_score, global_score

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def gradient_penalty(discriminator: DualDiscriminator,
                      real: torch.Tensor, fake: torch.Tensor,
                      device: str) -> torch.Tensor:
    """Standard WGAN-GP gradient penalty applied to the local discriminator branch."""
    B = real.shape[0]
    eps = torch.rand(B, 1, 1, device=device)
    interpolated = (eps * real + (1 - eps) * fake).requires_grad_(True)

    d_interpolated, _ = discriminator(interpolated)
    grad_outputs = torch.ones_like(d_interpolated, device=device)

    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.reshape(B, -1)
    grad_norm = gradients.norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()


GAN_MODEL_CONFIGS = {
    "tiny":   {"base_dim": 8,  "depth": 2},
    "small":  {"base_dim": 16, "depth": 3},
    "medium": {"base_dim": 32, "depth": 3},
    "large":  {"base_dim": 64, "depth": 4},
}


def build_generator(config_name: str, n_channels: int = 64) -> DivideAndFuseGenerator:
    cfg = GAN_MODEL_CONFIGS[config_name]
    return DivideAndFuseGenerator(n_channels=n_channels, **cfg)


def build_discriminator(config_name: str, n_channels: int = 64) -> DualDiscriminator:
    cfg = GAN_MODEL_CONFIGS[config_name]
    return DualDiscriminator(n_channels=n_channels, base_dim=cfg["base_dim"])


if __name__ == "__main__":
    for name in GAN_MODEL_CONFIGS:
        gen = build_generator(name)
        disc = build_discriminator(name)
        print(f"{name:8s} — Generator params: {gen.count_parameters():,}  "
              f"Discriminator params: {disc.count_parameters():,}")

    x = torch.randn(2, 64, 256)
    gen = build_generator("small")
    out = gen(x)
    print(f"\nGenerator I/O check — input: {x.shape}, output: {out.shape}")
    assert out.shape == x.shape, "Generator must preserve input shape!"

    disc = build_discriminator("small")
    local_score, global_score = disc(out)
    print(f"Discriminator scores — local: {local_score.shape}, "
          f"global: {global_score.shape}")
