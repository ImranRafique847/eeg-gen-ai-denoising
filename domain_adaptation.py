"""
WGAN domain adaptation + MMDA-VAE for cross-subject EEG transfer.
WGAN-DA: supervised (needs a few target labels). MMDA-VAE: zero target labels needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from eeg_dataset import EEGDenoisingDataset


def make_subject_datasets(n_subjects: int = 2, n_samples_each: int = 200,
                           n_channels: int = 64, signal_length: int = 256,
                           n_classes: int = 4, seed: int = 0):
    """One TensorDataset per subject, different seeds simulate inter-subject variability."""
    datasets = []
    for s in range(n_subjects):
        ds = EEGDenoisingDataset(
            n_samples=n_samples_each, n_channels=n_channels,
            signal_length=signal_length, artifact_prob=0.4,
            seed=seed + s * 100
        )
        labels = torch.arange(n_samples_each) % n_classes
        datasets.append(TensorDataset(ds.noisy, labels))
    return datasets


class EEGFeatureExtractor(nn.Module):
    """Shared CNN backbone mapping EEG trials to feature vectors."""

    def __init__(self, n_channels: int, feature_dim: int, base_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_channels, base_dim, 7, padding=3),
            nn.InstanceNorm1d(base_dim, affine=True), nn.ELU(),
            nn.Conv1d(base_dim, base_dim * 2, 5, stride=2, padding=2),
            nn.InstanceNorm1d(base_dim * 2, affine=True), nn.ELU(),
            nn.Conv1d(base_dim * 2, base_dim * 4, 5, stride=2, padding=2),
            nn.InstanceNorm1d(base_dim * 4, affine=True), nn.ELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Linear(base_dim * 4, feature_dim)

    def forward(self, x):
        return self.proj(self.net(x).squeeze(-1))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# supervised domain adaptation

class DomainDiscriminator(nn.Module):
    """Discriminates which subject a feature came from; extractor is trained adversarially."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(feature_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x)


class TaskClassifier(nn.Module):
    """Classifies MI/emotion labels from domain-invariant features."""

    def __init__(self, feature_dim: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class WGANDA(nn.Module):
    """
    Wasserstein GAN Domain Adaptation.
    Feature extractor is trained to maximise task accuracy while minimising
    Wasserstein distance between source and target feature distributions.
    """

    def __init__(self, n_channels: int, feature_dim: int,
                 n_classes: int, base_dim: int = 32):
        super().__init__()
        self.extractor   = EEGFeatureExtractor(n_channels, feature_dim, base_dim)
        self.discriminator = DomainDiscriminator(feature_dim)
        self.classifier    = TaskClassifier(feature_dim, n_classes)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def train_wgan_da(
    config: dict,
    n_epochs: int = 30,
    n_classes: int = 4,
    batch_size: int = 16,
    lr_feat: float = 1e-4,
    lr_disc: float = 1e-4,
    lambda_adv: float = 1.0,
    n_critic: int = 3,
    n_channels: int = 64,
    signal_length: int = 256,
    n_train: int = 200,
    seed: int = 0,
    verbose: bool = True,
):
    """Train WGAN-DA with source (subject 0) and target (subject 1), both labelled."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feature_dim = config["feature_dim"]
    base_dim    = config["base_dim"]

    model = WGANDA(n_channels, feature_dim, n_classes, base_dim).to(device)

    opt_feat = torch.optim.Adam(
        list(model.extractor.parameters()) +
        list(model.classifier.parameters()),
        lr=lr_feat, betas=(0.5, 0.9)
    )
    opt_disc = torch.optim.Adam(model.discriminator.parameters(),
                                lr=lr_disc, betas=(0.5, 0.9))

    subjects = make_subject_datasets(2, n_train, n_channels, signal_length,
                                      n_classes, seed)
    src_loader = DataLoader(subjects[0], batch_size=batch_size,
                            shuffle=True, drop_last=True)
    tgt_loader = DataLoader(subjects[1], batch_size=batch_size,
                            shuffle=True, drop_last=True)

    cls_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    history = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        run_cls = run_disc = run_adv = 0.0; n_b = 0

        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)

        for _ in range(min(len(src_loader), len(tgt_loader))):
            try:
                src_eeg, src_lbl = next(src_iter)
                tgt_eeg, tgt_lbl = next(tgt_iter)
            except StopIteration:
                break

            src_eeg, src_lbl = src_eeg.to(device), src_lbl.to(device)
            tgt_eeg, tgt_lbl = tgt_eeg.to(device), tgt_lbl.to(device)

            src_feat = model.extractor(src_eeg)
            tgt_feat = model.extractor(tgt_eeg)

            for _ in range(n_critic):
                src_feat_d = model.extractor(src_eeg).detach()
                tgt_feat_d = model.extractor(tgt_eeg).detach()
                d_src = model.discriminator(src_feat_d)
                d_tgt = model.discriminator(tgt_feat_d)
                eps = torch.rand(src_feat_d.shape[0], 1, device=device)
                interp = (eps * src_feat_d + (1 - eps) * tgt_feat_d).requires_grad_(True)
                d_interp = model.discriminator(interp)
                grads = torch.autograd.grad(
                    outputs=d_interp, inputs=interp,
                    grad_outputs=torch.ones_like(d_interp),
                    create_graph=True, retain_graph=True,
                )[0]
                gp = ((grads.norm(2, dim=1) - 1) ** 2).mean()
                disc_loss = -(d_src.mean() - d_tgt.mean()) + 10.0 * gp
                opt_disc.zero_grad(); disc_loss.backward(); opt_disc.step()

            src_feat2 = model.extractor(src_eeg)
            tgt_feat2 = model.extractor(tgt_eeg)

            src_logits = model.classifier(src_feat2)
            tgt_logits = model.classifier(tgt_feat2)

            cls_loss  = (cls_criterion(src_logits, src_lbl) +
                         cls_criterion(tgt_logits, tgt_lbl)) * 0.5
            adv_loss  = -(model.discriminator(src_feat2).mean() -
                          model.discriminator(tgt_feat2).mean())
            # negate adv — extractor wants to fool the discriminator
            feat_loss = cls_loss - lambda_adv * adv_loss

            opt_feat.zero_grad(); feat_loss.backward(); opt_feat.step()

            run_cls  += cls_loss.item()
            run_disc += disc_loss.item()
            run_adv  += adv_loss.item()
            n_b += 1

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for eeg, lbl in DataLoader(subjects[1], batch_size=batch_size):
                eeg, lbl = eeg.to(device), lbl.to(device)
                logits = model.classifier(model.extractor(eeg))
                val_correct += (logits.argmax(1) == lbl).sum().item()
                val_total += lbl.size(0)

        val_acc = val_correct / val_total * 100
        record = {"epoch": epoch, "cls": run_cls/n_b,
                  "disc": run_disc/n_b, "val_acc": val_acc}
        history.append(record)

        if verbose:
            print(
                f"[WGAN-DA] epoch {epoch:3d}/{n_epochs} "
                f"| cls={run_cls/n_b:.4f} disc={run_disc/n_b:.4f} "
                f"| target val acc={val_acc:.1f}%"
            )

    return model, history


# unsupervised domain adaptation (zero target labels)

class MMDAVAE(nn.Module):
    """
    Multi-Modal Domain Adaptive VAE.
    Aligns source and target latent distributions via MMD without needing target labels.
    A classifier trained on source latents then transfers to target.
    """

    def __init__(self, n_channels: int, signal_length: int,
                 latent_dim: int, base_dim: int, depth: int):
        super().__init__()

        enc = [nn.Conv1d(n_channels, base_dim, 7, padding=3),
               nn.InstanceNorm1d(base_dim, affine=True), nn.ELU()]
        ch = base_dim
        for _ in range(depth):
            enc += [nn.Conv1d(ch, ch * 2, 4, stride=2, padding=1),
                    nn.InstanceNorm1d(ch * 2, affine=True), nn.ELU()]
            ch = ch * 2
        enc.append(nn.AdaptiveAvgPool1d(1))
        self.encoder   = nn.Sequential(*enc)
        self.fc_mu     = nn.Linear(ch, latent_dim)
        self.fc_lv     = nn.Linear(ch, latent_dim)
        self.bottleneck_ch = ch

        self.init_len = signal_length // (2 ** depth)
        self.fc_dec = nn.Linear(latent_dim, ch * self.init_len)
        dec = []
        for _ in range(depth):
            dec += [nn.ConvTranspose1d(ch, ch // 2, 4, stride=2, padding=1),
                    nn.InstanceNorm1d(ch // 2, affine=True), nn.ELU()]
            ch = ch // 2
        dec.append(nn.Conv1d(ch, n_channels, 7, padding=3))
        self.decoder = nn.Sequential(*dec)
        self._dec_init_ch = self.bottleneck_ch

        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2), nn.GELU(),
            nn.Linear(latent_dim // 2, 4),
        )

    def encode(self, x):
        h = self.encoder(x).squeeze(-1)
        return self.fc_mu(h), self.fc_lv(h)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, self._dec_init_ch, self.init_len)
        out = self.decoder(h)
        return out

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if self.training else mu
        recon = self.decode(z)
        return recon, mu, lv, z

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def mmd_loss(z_src: torch.Tensor, z_tgt: torch.Tensor,
             kernel_bw: float = 1.0) -> torch.Tensor:
    """
    Maximum Mean Discrepancy (Gaussian kernel).
    Minimising this aligns source and target distributions without target labels.
    """
    def rbf(a, b):
        diff = a.unsqueeze(1) - b.unsqueeze(0)  # (N, M, D)
        return torch.exp(-diff.pow(2).sum(-1) / (2 * kernel_bw ** 2))

    return (rbf(z_src, z_src).mean() - 2 * rbf(z_src, z_tgt).mean()
            + rbf(z_tgt, z_tgt).mean())


def train_mmda_vae(
    config: dict,
    n_epochs: int = 30,
    n_classes: int = 4,
    batch_size: int = 16,
    lr: float = 1e-3,
    beta_kl: float = 0.1,
    lambda_mmd: float = 1.0,
    lambda_cls: float = 5.0,
    n_channels: int = 64,
    signal_length: int = 256,
    n_train: int = 200,
    seed: int = 0,
    verbose: bool = True,
):
    """Train MMDA-VAE: source has labels, target has none (unsupervised)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MMDAVAE(n_channels, signal_length,
                     config["latent_dim"], config["base_dim"],
                     config["depth"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    subjects = make_subject_datasets(2, n_train, n_channels, signal_length,
                                      n_classes, seed)
    src_loader = DataLoader(subjects[0], batch_size=batch_size,
                            shuffle=True, drop_last=True)
    tgt_loader = DataLoader(subjects[1], batch_size=batch_size,
                            shuffle=True, drop_last=True)

    cls_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    history = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        run_total = run_cls_acc = 0.0; n_b = 0

        src_iter = iter(src_loader)
        tgt_iter = iter(tgt_loader)

        for _ in range(min(len(src_loader), len(tgt_loader))):
            try:
                src_eeg, src_lbl = next(src_iter)
                tgt_eeg, _       = next(tgt_iter)  # no target labels used
            except StopIteration:
                break

            src_eeg, src_lbl = src_eeg.to(device), src_lbl.to(device)
            tgt_eeg = tgt_eeg.to(device)

            src_recon, src_mu, src_lv, src_z = model(src_eeg)
            tgt_recon, tgt_mu, tgt_lv, tgt_z = model(tgt_eeg)

            recon_loss = (F.mse_loss(src_recon, src_eeg) +
                          F.mse_loss(tgt_recon, tgt_eeg)) * 0.5

            kl_loss = ((-0.5 * (1 + src_lv - src_mu.pow(2) - src_lv.exp())).mean() +
                       (-0.5 * (1 + tgt_lv - tgt_mu.pow(2) - tgt_lv.exp())).mean()) * 0.5

            mmd = mmd_loss(src_z, tgt_z)

            src_logits = model.classifier(src_z)
            cls_loss   = cls_criterion(src_logits, src_lbl)

            total = (recon_loss + beta_kl * kl_loss +
                     lambda_mmd * mmd + lambda_cls * cls_loss)

            optimizer.zero_grad(); total.backward(); optimizer.step()

            run_total    += total.item()
            run_cls_acc  += (src_logits.argmax(1) == src_lbl).float().mean().item()
            n_b += 1

        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for eeg, lbl in DataLoader(subjects[1], batch_size=batch_size):
                eeg, lbl = eeg.to(device), lbl.to(device)
                _, mu, _, _ = model(eeg)
                logits = model.classifier(mu)
                val_correct += (logits.argmax(1) == lbl).sum().item()
                val_total += lbl.size(0)

        val_acc = val_correct / val_total * 100
        record = {"epoch": epoch, "loss": run_total/n_b,
                  "src_acc": run_cls_acc/n_b*100, "target_acc_zeroshot": val_acc}
        history.append(record)

        if verbose:
            print(
                f"[MMDA-VAE] epoch {epoch:3d}/{n_epochs} "
                f"| loss={run_total/n_b:.4f} "
                f"| src acc={run_cls_acc/n_b*100:.1f}% "
                f"| target (0-shot)={val_acc:.1f}%"
            )

    return model, history


DA_CONFIGS = {
    "tiny":  {"feature_dim": 64,  "latent_dim": 64,  "base_dim": 16, "depth": 2},
    "small": {"feature_dim": 128, "latent_dim": 128, "base_dim": 32, "depth": 3},
    "medium":{"feature_dim": 256, "latent_dim": 256, "base_dim": 64, "depth": 3},
}


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(
        description="Train domain adaptation model for cross-subject EEG transfer."
    )
    parser.add_argument("--method", default="wgan",
                        choices=["wgan", "mmda"])
    parser.add_argument("--model",  default="small",
                        choices=list(DA_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--quick",  action="store_true")
    args = parser.parse_args()

    if args.quick:
        config_name, epochs, n_train = "tiny", 3, 80
        print("Running in --quick mode\n")
    else:
        config_name, epochs, n_train = args.model, args.epochs, 200

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DA_CONFIGS[config_name]
    print(f"Device: {device}")
    print(f"Method: {args.method.upper()}  Model: {config_name}  "
          f"epochs={epochs}\n")

    os.makedirs("results", exist_ok=True)

    if args.method == "wgan":
        model, history = train_wgan_da(
            config=cfg, n_epochs=epochs, n_train=n_train
        )
        save_path = f"results/wgan_da_{config_name}_trained.pt"
    else:
        model, history = train_mmda_vae(
            config=cfg, n_epochs=epochs, n_train=n_train
        )
        save_path = f"results/mmda_vae_{config_name}_trained.pt"

    final = history[-1]
    print("\nFinal metrics:")
    for k, v in final.items():
        if k != "epoch":
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    torch.save(model.state_dict(), save_path)
    print(f"\nSaved to: {save_path}")
