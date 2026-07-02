"""
VAE-GAN with dual encoders for EEG emotion recognition.
Disentangles content and emotion latents; GAN discriminator improves reconstruction quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from eeg_dataset import EEGDenoisingDataset
from metrics import evaluate_all


def make_emotion_dataset(n_samples: int, n_emotions: int,
                          n_channels: int, signal_length: int,
                          seed: int = 0):
    """Synthetic emotion dataset built on EEGDenoisingDataset with round-robin labels."""
    ds = EEGDenoisingDataset(
        n_samples=n_samples, n_channels=n_channels,
        signal_length=signal_length, artifact_prob=0.3, seed=seed
    )
    labels = torch.arange(n_samples) % n_emotions
    return TensorDataset(ds.noisy, labels)


class ConvEncoder(nn.Module):
    """Shared Conv1D backbone mapping EEG to latent mu/logvar."""

    def __init__(self, n_channels: int, base_dim: int, depth: int,
                 latent_dim: int):
        super().__init__()
        layers = [nn.Conv1d(n_channels, base_dim, 7, padding=3),
                  nn.InstanceNorm1d(base_dim, affine=True), nn.ELU()]
        ch = base_dim
        for _ in range(depth):
            out_ch = min(ch * 2, 512)
            layers += [
                nn.Conv1d(ch, out_ch, 4, stride=2, padding=1),
                nn.InstanceNorm1d(out_ch, affine=True),
                nn.ELU(),
            ]
            ch = out_ch
        self.net = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_ch = ch
        self.fc_mu  = nn.Linear(ch, latent_dim)
        self.fc_lv  = nn.Linear(ch, latent_dim)

    def forward(self, x):
        h = self.pool(self.net(x)).squeeze(-1)
        return self.fc_mu(h), self.fc_lv(h)


class EmotionEncoder(nn.Module):
    """
    Content encoder conditioned on the emotion label.
    Learns to disentangle the emotion-relevant part of the EEG signal.
    """

    def __init__(self, n_channels: int, base_dim: int, depth: int,
                 latent_dim: int, n_emotions: int):
        super().__init__()
        self.conv_enc   = ConvEncoder(n_channels, base_dim, depth, latent_dim)
        self.label_emb  = nn.Embedding(n_emotions, latent_dim)
        self.fc_mu = nn.Linear(latent_dim * 2, latent_dim)
        self.fc_lv = nn.Linear(latent_dim * 2, latent_dim)

    def forward(self, x, labels):
        mu_c, lv_c = self.conv_enc(x)
        lbl = self.label_emb(labels)
        combined = torch.cat([mu_c, lbl], dim=1)
        return self.fc_mu(combined), self.fc_lv(combined)


class Decoder(nn.Module):
    """Decodes z_content + z_emotion -> EEG signal."""

    def __init__(self, n_channels: int, signal_length: int,
                 latent_dim: int, base_dim: int, depth: int):
        super().__init__()
        self.signal_length = signal_length
        self.init_len  = signal_length // (2 ** depth)
        self.init_ch   = base_dim * (2 ** depth)

        self.fc = nn.Linear(latent_dim * 2, self.init_ch * self.init_len)

        layers = []
        ch = self.init_ch
        for _ in range(depth):
            out_ch = ch // 2
            layers += [
                nn.ConvTranspose1d(ch, out_ch, 4, stride=2, padding=1),
                nn.InstanceNorm1d(out_ch, affine=True),
                nn.ELU(),
            ]
            ch = out_ch
        layers.append(nn.Conv1d(ch, n_channels, 7, padding=3))
        self.net = nn.Sequential(*layers)

    def forward(self, z_content, z_emotion):
        z = torch.cat([z_content, z_emotion], dim=1)
        h = self.fc(z).view(-1, self.init_ch, self.init_len)
        out = self.net(h)
        if out.shape[-1] != self.signal_length:
            out = F.interpolate(out, size=self.signal_length,
                                mode="linear", align_corners=False)
        return out


class PatchDiscriminator(nn.Module):
    """Patch-level realness scoring for EEG signal quality."""

    def __init__(self, n_channels: int, base_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, base_dim, 7, stride=2, padding=3),
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
        return self.net(x)   # (B, 1, L') patch scores


class EmotionClassifier(nn.Module):
    def __init__(self, latent_dim: int, n_emotions: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim // 2, n_emotions),
        )

    def forward(self, z):
        return self.net(z)


class VAEGANEmotion(nn.Module):
    """
    Dual-encoder VAE-GAN for EEG emotion recognition.
    Combines disentangled VAE latents with GAN discriminator and emotion classifier.
    """

    def __init__(
        self,
        n_channels: int = 64,
        signal_length: int = 256,
        n_emotions: int = 4,
        latent_dim: int = 64,
        base_dim: int = 32,
        depth: int = 3,
    ):
        super().__init__()
        self.content_encoder = ConvEncoder(n_channels, base_dim, depth,
                                           latent_dim)
        self.emotion_encoder  = EmotionEncoder(n_channels, base_dim, depth,
                                               latent_dim, n_emotions)
        self.decoder = Decoder(n_channels, signal_length, latent_dim,
                               base_dim, depth)
        self.discriminator = PatchDiscriminator(n_channels, base_dim)
        self.classifier    = EmotionClassifier(latent_dim, n_emotions)

    def reparameterise(self, mu, lv):
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return mu

    def encode(self, x, labels):
        mu_c, lv_c = self.content_encoder(x)
        mu_e, lv_e = self.emotion_encoder(x, labels)
        return mu_c, lv_c, mu_e, lv_e

    def forward(self, x, labels):
        mu_c, lv_c, mu_e, lv_e = self.encode(x, labels)
        z_c = self.reparameterise(mu_c, lv_c)
        z_e = self.reparameterise(mu_e, lv_e)
        recon  = self.decoder(z_c, z_e)
        logits = self.classifier(z_e)
        return recon, mu_c, lv_c, mu_e, lv_e, logits

    def classify(self, x, labels):
        """Return emotion class probabilities."""
        self.eval()
        with torch.no_grad():
            _, _, _, mu_e, _, logits = self.forward(x, labels)
        return F.softmax(logits, dim=-1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


VAEGAN_CONFIGS = {
    "tiny":   {"latent_dim": 32,  "base_dim": 16, "depth": 2},
    "small":  {"latent_dim": 64,  "base_dim": 32, "depth": 3},
    "medium": {"latent_dim": 128, "base_dim": 64, "depth": 3},
}


def build_vae_gan(config_name: str, n_channels: int = 64,
                  signal_length: int = 256,
                  n_emotions: int = 4) -> VAEGANEmotion:
    cfg = VAEGAN_CONFIGS[config_name]
    return VAEGANEmotion(n_channels=n_channels, signal_length=signal_length,
                         n_emotions=n_emotions, **cfg)


def train_vae_gan(
    config_name: str = "small",
    n_epochs: int = 30,
    n_emotions: int = 4,
    batch_size: int = 16,
    lr: float = 2e-4,
    beta_kl: float = 0.5,
    lambda_adv: float = 1.0,
    lambda_cls: float = 5.0,
    n_train: int = 400,
    n_val: int = 80,
    n_channels: int = 64,
    signal_length: int = 256,
    seed: int = 0,
    verbose: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_vae_gan(config_name, n_channels, signal_length,
                           n_emotions).to(device)

    vae_params = (list(model.content_encoder.parameters()) +
                  list(model.emotion_encoder.parameters()) +
                  list(model.decoder.parameters()) +
                  list(model.classifier.parameters()))
    opt_vae  = torch.optim.Adam(vae_params,               lr=lr, betas=(0.5, 0.9))
    opt_disc = torch.optim.Adam(model.discriminator.parameters(),
                                lr=lr, betas=(0.5, 0.9))

    train_ds = make_emotion_dataset(n_train, n_emotions, n_channels,
                                     signal_length, seed=seed)
    val_ds   = make_emotion_dataset(n_val,   n_emotions, n_channels,
                                     signal_length, seed=seed + 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    cls_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    history = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        run_vae = run_disc = run_cls_acc = 0.0
        n_batches = 0

        for eeg, labels in train_loader:
            eeg, labels = eeg.to(device), labels.to(device)

            recon, mu_c, lv_c, mu_e, lv_e, logits = model(eeg, labels)

            d_real = model.discriminator(eeg)
            d_fake = model.discriminator(recon.detach())
            disc_loss = (F.relu(1 - d_real).mean() +
                         F.relu(1 + d_fake).mean())
            opt_disc.zero_grad(); disc_loss.backward(); opt_disc.step()

            recon_loss = F.mse_loss(recon, eeg)
            kl_c = -0.5 * (1 + lv_c - mu_c.pow(2) - lv_c.exp()).mean()
            kl_e = -0.5 * (1 + lv_e - mu_e.pow(2) - lv_e.exp()).mean()
            adv_loss = -model.discriminator(recon).mean()
            cls_loss = cls_criterion(logits, labels)

            vae_loss = (recon_loss
                        + beta_kl * (kl_c + kl_e)
                        + lambda_adv * adv_loss
                        + lambda_cls * cls_loss)

            opt_vae.zero_grad(); vae_loss.backward(); opt_vae.step()

            run_vae  += vae_loss.item()
            run_disc += disc_loss.item()
            run_cls_acc += (logits.argmax(1) == labels).float().mean().item()
            n_batches += 1

        model.eval()
        val_acc = val_recon_snr = 0.0
        with torch.no_grad():
            for eeg, labels in val_loader:
                eeg, labels = eeg.to(device), labels.to(device)
                recon, mu_c, lv_c, mu_e, lv_e, logits = model(eeg, labels)
                val_acc += (logits.argmax(1) == labels).float().mean().item()
                m = evaluate_all(eeg, recon)
                val_recon_snr += m["snr_db"]
        n_v = len(val_loader)

        record = {
            "epoch": epoch,
            "vae_loss":  run_vae  / n_batches,
            "disc_loss": run_disc / n_batches,
            "train_cls_acc": run_cls_acc / n_batches * 100,
            "val_cls_acc":   val_acc / n_v * 100,
            "val_recon_snr": val_recon_snr / n_v,
        }
        history.append(record)

        if verbose:
            print(
                f"[VAEGAN-{config_name}] epoch {epoch:3d}/{n_epochs} "
                f"| vae={run_vae/n_batches:.3f} disc={run_disc/n_batches:.3f} "
                f"| train_acc={run_cls_acc/n_batches*100:5.1f}% "
                f"val_acc={val_acc/n_v*100:5.1f}% "
                f"recon_SNR={val_recon_snr/n_v:.2f}dB"
            )

    return model, history


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(
        description="Train VAE-GAN for EEG emotion recognition."
    )
    parser.add_argument("--model",      default="small",
                        choices=list(VAEGAN_CONFIGS.keys()))
    parser.add_argument("--epochs",     type=int, default=30)
    parser.add_argument("--n_emotions", type=int, default=4)
    parser.add_argument("--quick",      action="store_true")
    args = parser.parse_args()

    if args.quick:
        config, epochs, n_train, n_val = "tiny", 3, 60, 20
        print("Running in --quick mode\n")
    else:
        config, epochs, n_train, n_val = args.model, args.epochs, 400, 80

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: VAE-GAN-{config}  epochs={epochs}  "
          f"n_emotions={args.n_emotions}\n")

    model = build_vae_gan(config, n_emotions=args.n_emotions)
    print(f"Parameters: {model.count_parameters():,}\n")

    trained, history = train_vae_gan(
        config_name=config, n_epochs=epochs,
        n_emotions=args.n_emotions,
        n_train=n_train, n_val=n_val,
    )

    print("\nFinal metrics:")
    final = history[-1]
    print(f"  Val emotion accuracy : {final['val_cls_acc']:.1f}%")
    print(f"  Val recon SNR        : {final['val_recon_snr']:.2f}dB")

    os.makedirs("results", exist_ok=True)
    save_path = f"results/vaegan_emotion_{config}_trained.pt"
    torch.save(trained.state_dict(), save_path)
    print(f"\nSaved to: {save_path}")
