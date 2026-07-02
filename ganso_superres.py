"""
GANSO-style graph GAN for EEG spatial super-resolution.
Upsamples from low-density (few electrodes) to high-density layouts using GCN + GAN.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from eeg_dataset import EEGDenoisingDataset
from metrics import evaluate_all


def build_adjacency(n_channels: int, n_neighbours: int = 4) -> torch.Tensor:
    """
    Banded adjacency matrix connecting channel i to i±n_neighbours.
    Proxy for spatial proximity — use real 3-D electrode positions for production.
    """
    A = torch.zeros(n_channels, n_channels)
    for i in range(n_channels):
        for offset in range(1, n_neighbours + 1):
            if i + offset < n_channels:
                A[i, i + offset] = 1.0
                A[i + offset, i] = 1.0
        A[i, i] = 1.0  # self-loop

    # symmetric normalisation: D^{-1/2} A D^{-1/2}
    deg = A.sum(dim=1, keepdim=True).clamp(min=1)
    A = A / deg.sqrt() / deg.sqrt().T
    return A


class GraphConv1D(nn.Module):
    """
    Graph convolution over the spatial (channel) dimension.
    Aggregates neighbour features using the adjacency matrix.
    """

    def __init__(self, in_features: int, out_features: int,
                 adjacency: torch.Tensor):
        super().__init__()
        self.register_buffer("A", adjacency)
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.norm   = nn.LayerNorm(out_features)
        self.act    = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, n_channels, in_features) -> (B, n_channels, out_features)"""
        x_agg = torch.einsum("nc,bct->bnt", self.A, x)
        x_proj = self.linear(x_agg)
        return self.act(self.norm(x_proj))


class GraphSuperResGenerator(nn.Module):
    """
    Upsamples low-density EEG to high-density via temporal feature extraction,
    graph convolution over the channel graph, and linear channel upsampling.
    """

    def __init__(self, n_low: int, n_high: int, signal_length: int,
                 feat_dim: int, n_gcn_layers: int):
        super().__init__()
        self.n_low  = n_low
        self.n_high = n_high

        A_low = build_adjacency(n_low, n_neighbours=min(3, n_low - 1))

        self.temporal_enc = nn.Sequential(
            nn.Conv1d(1, feat_dim, kernel_size=7, padding=3),
            nn.InstanceNorm1d(feat_dim, affine=True), nn.ELU(),
            nn.Conv1d(feat_dim, feat_dim, kernel_size=5, padding=2),
            nn.InstanceNorm1d(feat_dim, affine=True), nn.ELU(),
        )

        self.gcn_in_dim = feat_dim

        self.gcn_layers = nn.ModuleList([
            GraphConv1D(feat_dim if i == 0 else feat_dim,
                        feat_dim, A_low)
            for i in range(n_gcn_layers)
        ])

        self.channel_upsample = nn.Linear(n_low, n_high)

        A_high = build_adjacency(n_high, n_neighbours=4)
        self.gcn_refine = GraphConv1D(feat_dim, feat_dim, A_high)

        self.temporal_dec = nn.Sequential(
            nn.Conv1d(feat_dim, feat_dim // 2, kernel_size=5, padding=2),
            nn.InstanceNorm1d(feat_dim // 2, affine=True), nn.ELU(),
            nn.Conv1d(feat_dim // 2, 1, kernel_size=3, padding=1),
        )

    def forward(self, x_low: torch.Tensor) -> torch.Tensor:
        """x_low: (B, n_low, L) -> (B, n_high, L)"""
        B, C, L = x_low.shape

        x_flat = x_low.reshape(B * C, 1, L)
        feat   = self.temporal_enc(x_flat)   # (B*C, feat_dim, L)
        feat   = feat.reshape(B, C, -1, L)   # (B, C, feat_dim, L)

        node_feat = feat.mean(dim=-1)         # (B, C, feat_dim)
        for gcn in self.gcn_layers:
            node_feat = gcn(node_feat)        # (B, C, feat_dim)

        node_feat_t = node_feat.permute(0, 2, 1)          # (B, feat_dim, n_low)
        node_feat_t = self.channel_upsample(node_feat_t)  # (B, feat_dim, n_high)
        node_feat   = node_feat_t.permute(0, 2, 1)        # (B, n_high, feat_dim)

        node_feat = self.gcn_refine(node_feat)             # (B, n_high, feat_dim)

        # bilinear upsample of low-density signal as residual base
        x_base = F.interpolate(
            x_low.unsqueeze(1), size=(self.n_high, L), mode="bilinear",
            align_corners=False
        ).squeeze(1)                                       # (B, n_high, L)

        node_feat_bc = node_feat.unsqueeze(-1).expand(B, self.n_high,
                                                       -1, L)
        n_high = self.n_high
        feat_dim = node_feat.shape[-1]
        dec_input = node_feat_bc.reshape(B * n_high, feat_dim, L)
        delta = self.temporal_dec(dec_input)               # (B*n_high, 1, L)
        delta = delta.reshape(B, n_high, L)

        return x_base + delta

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SuperResDiscriminator(nn.Module):
    """PatchGAN discriminator for high-density EEG signals."""

    def __init__(self, n_high: int, base_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_high, base_dim, 7, stride=2, padding=3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_dim, base_dim * 2, 5, stride=2, padding=2),
            nn.InstanceNorm1d(base_dim * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_dim * 2, base_dim * 4, 5, stride=2, padding=2),
            nn.InstanceNorm1d(base_dim * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_dim * 4, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)


GANSO_CONFIGS = {
    "tiny":  {"feat_dim": 16, "n_gcn_layers": 2, "base_dim": 16},
    "small": {"feat_dim": 32, "n_gcn_layers": 3, "base_dim": 32},
    "medium":{"feat_dim": 64, "n_gcn_layers": 4, "base_dim": 64},
}


def build_ganso(config_name: str, n_low: int = 16, n_high: int = 64,
                signal_length: int = 256) -> GraphSuperResGenerator:
    cfg = GANSO_CONFIGS[config_name]
    return GraphSuperResGenerator(
        n_low=n_low, n_high=n_high, signal_length=signal_length,
        feat_dim=cfg["feat_dim"], n_gcn_layers=cfg["n_gcn_layers"]
    )


def train_ganso(
    config_name: str = "small",
    n_epochs: int = 30,
    batch_size: int = 8,
    lr: float = 2e-4,
    lambda_l1: float = 10.0,
    lambda_gp: float = 10.0,
    n_low: int = 16,
    n_high: int = 64,
    signal_length: int = 256,
    n_train: int = 400,
    n_val: int = 80,
    seed: int = 0,
    verbose: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = GANSO_CONFIGS[config_name]

    G = build_ganso(config_name, n_low, n_high, signal_length).to(device)
    D = SuperResDiscriminator(n_high, cfg["base_dim"]).to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.9))

    train_ds = EEGDenoisingDataset(
        n_samples=n_train, n_channels=n_high, signal_length=signal_length,
        artifact_prob=0.0, seed=seed
    )
    val_ds = EEGDenoisingDataset(
        n_samples=n_val, n_channels=n_high, signal_length=signal_length,
        artifact_prob=0.0, seed=seed + 1
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    l1_fn = nn.L1Loss()
    history = []

    for epoch in range(1, n_epochs + 1):
        G.train(); D.train()
        run_g = run_d = 0.0; n_b = 0

        for _, high in train_loader:
            high = high.to(device)
            low  = high[:, :n_low, :]    # subsample channels as "low density"

            fake = G(low).detach()
            d_real = D(high); d_fake = D(fake)

            B = high.shape[0]
            eps = torch.rand(B, 1, 1, device=device)
            interp = (eps * high + (1 - eps) * fake).requires_grad_(True)
            d_interp = D(interp)
            grads = torch.autograd.grad(
                d_interp, interp,
                grad_outputs=torch.ones_like(d_interp),
                create_graph=True
            )[0].reshape(B, -1)
            gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()

            d_loss = -(d_real.mean() - d_fake.mean()) + lambda_gp * gp
            opt_d.zero_grad(); d_loss.backward(); opt_d.step()

            fake = G(low)
            g_adv = -D(fake).mean()
            g_l1  = l1_fn(fake, high)
            g_loss = g_adv + lambda_l1 * g_l1
            opt_g.zero_grad(); g_loss.backward(); opt_g.step()

            run_g += g_loss.item(); run_d += d_loss.item()
            n_b += 1

        G.eval()
        val_snr = val_ssim = 0.0
        with torch.no_grad():
            for _, high in val_loader:
                high = high.to(device)
                low  = high[:, :n_low, :]
                fake = G(low)
                m = evaluate_all(high, fake)
                val_snr  += m["snr_db"]
                val_ssim += m["ssim"]
        nv = len(val_loader)

        record = {"epoch": epoch, "g_loss": run_g/n_b, "d_loss": run_d/n_b,
                  "val_snr": val_snr/nv, "val_ssim": val_ssim/nv}
        history.append(record)

        if verbose:
            print(
                f"[GANSO-{config_name}] epoch {epoch:3d}/{n_epochs} "
                f"| g={run_g/n_b:.3f} d={run_d/n_b:.3f} "
                f"| val SNR={val_snr/nv:.2f}dB SSIM={val_ssim/nv:.4f}"
            )

    return G, history


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(
        description="Train graph GAN for EEG spatial super-resolution."
    )
    parser.add_argument("--model",   default="small",
                        choices=list(GANSO_CONFIGS.keys()))
    parser.add_argument("--epochs",  type=int, default=30)
    parser.add_argument("--n_low",   type=int, default=16)
    parser.add_argument("--n_high",  type=int, default=64)
    parser.add_argument("--quick",   action="store_true")
    args = parser.parse_args()

    if args.quick:
        config, epochs, n_train, n_val = "tiny", 2, 40, 16
        print("Running in --quick mode\n")
    else:
        config, epochs, n_train, n_val = args.model, args.epochs, 400, 80

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: GANSO-{config}  epochs={epochs}  "
          f"{args.n_low}ch -> {args.n_high}ch\n")

    G = build_ganso(config, args.n_low, args.n_high)
    print(f"Generator parameters: {G.count_parameters():,}\n")

    trained, history = train_ganso(
        config_name=config, n_epochs=epochs,
        n_low=args.n_low, n_high=args.n_high,
        n_train=n_train, n_val=n_val,
    )

    print("\nFinal metrics:")
    final = history[-1]
    print(f"  Val SNR : {final['val_snr']:.2f}dB")
    print(f"  Val SSIM: {final['val_ssim']:.4f}")

    os.makedirs("results", exist_ok=True)
    save_path = f"results/ganso_{config}_trained.pt"
    torch.save(trained.state_dict(), save_path)
    print(f"\nSaved to: {save_path}")
