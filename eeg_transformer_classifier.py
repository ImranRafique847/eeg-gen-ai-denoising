"""
EEG Transformer classifier for motor imagery decoding.
Spatial filter + multi-scale temporal CNN + Transformer encoder + classification head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset

from eeg_dataset import EEGDenoisingDataset


class SpatialFilter(nn.Module):
    """Learnable spatial projection over EEG channels (like CSP but learned)."""

    def __init__(self, n_channels: int, n_filters: int):
        super().__init__()
        self.conv = nn.Conv1d(n_channels, n_filters, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm1d(n_filters)

    def forward(self, x):
        return F.elu(self.norm(self.conv(x)))


class TemporalCNN(nn.Module):
    """Multi-scale temporal feature extractor."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        branch_ch = out_ch // 3
        extra = out_ch - branch_ch * 3

        def branch(k):
            return nn.Sequential(
                nn.Conv1d(in_ch, branch_ch, kernel_size=k,
                          padding=k // 2, bias=False),
                nn.BatchNorm1d(branch_ch),
                nn.ELU(),
            )

        self.b3  = branch(3)
        self.b7  = branch(7)
        self.b15 = nn.Sequential(
            nn.Conv1d(in_ch, branch_ch + extra, kernel_size=15,
                      padding=7, bias=False),
            nn.BatchNorm1d(branch_ch + extra),
            nn.ELU(),
        )
        self.pool = nn.AvgPool1d(kernel_size=4, stride=4)

    def forward(self, x):
        x = torch.cat([self.b3(x), self.b7(x), self.b15(x)], dim=1)
        return self.pool(x)


class TransformerEncoderBlock(nn.Module):
    """Standard pre-norm Transformer encoder block."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class EEGTransformerClassifier(nn.Module):
    """
    Full EEG motor imagery classifier.
    Input: (B, C, L). Output: (B, n_classes) logits.
    """

    def __init__(
        self,
        n_channels: int = 64,
        signal_length: int = 256,
        n_classes: int = 4,
        n_spatial: int = 32,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.25,
    ):
        super().__init__()

        self.spatial = SpatialFilter(n_channels, n_spatial)
        self.temporal = TemporalCNN(n_spatial, d_model)

        seq_len = signal_length // 4
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.transformer = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(d_model // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial(x)                     # (B, n_spatial, L)
        x = self.temporal(x)                    # (B, d_model, L//4)
        x = x.permute(0, 2, 1) + self.pos_embed # (B, L//4, d_model)
        for blk in self.transformer:
            x = blk(x)
        x = self.norm(x)
        x = x.mean(dim=1)                       # global average pool over time
        return self.head(x)                     # (B, n_classes)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


CLASSIFIER_CONFIGS = {
    "tiny":   {"n_spatial": 16, "d_model": 64,  "n_heads": 2, "n_layers": 2},
    "small":  {"n_spatial": 32, "d_model": 128, "n_heads": 4, "n_layers": 3},
    "medium": {"n_spatial": 64, "d_model": 256, "n_heads": 8, "n_layers": 4},
}


def build_classifier(config_name: str, n_channels: int = 64,
                     signal_length: int = 256,
                     n_classes: int = 4) -> EEGTransformerClassifier:
    cfg = CLASSIFIER_CONFIGS[config_name]
    return EEGTransformerClassifier(
        n_channels=n_channels, signal_length=signal_length,
        n_classes=n_classes, **cfg
    )


def make_mi_dataset(n_samples: int, n_classes: int,
                    n_channels: int, signal_length: int, seed: int):
    """Wraps EEGDenoisingDataset with MI class labels."""
    ds = EEGDenoisingDataset(
        n_samples=n_samples, n_channels=n_channels,
        signal_length=signal_length, artifact_prob=0.3, seed=seed
    )
    labels = torch.arange(n_samples) % n_classes
    return TensorDataset(ds.noisy, labels)


def train_classifier(
    config_name: str = "small",
    n_epochs: int = 30,
    n_classes: int = 4,
    batch_size: int = 32,
    lr: float = 5e-4,
    n_train: int = 400,
    n_val: int = 80,
    n_channels: int = 64,
    signal_length: int = 256,
    augment: bool = False,
    seed: int = 0,
    verbose: bool = True,
):
    """
    Train the Transformer classifier. augment=True mixes in WGAN synthetic trials
    (requires wgan_augmentation.py weights).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_classifier(config_name, n_channels, signal_length,
                              n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01
    )

    train_ds = make_mi_dataset(n_train, n_classes, n_channels,
                               signal_length, seed=seed)
    val_ds   = make_mi_dataset(n_val,   n_classes, n_channels,
                               signal_length, seed=seed + 1)

    if augment:
        try:
            import os
            from wgan_augmentation import build_wgan_generator, \
                WGAN_AUG_CONFIGS, generate_synthetic_trials

            wgan_path = f"results/wgan_aug_small_trained.pt"
            if os.path.exists(wgan_path):
                G = build_wgan_generator("small", n_channels,
                                          signal_length, n_classes)
                G.load_state_dict(torch.load(wgan_path, map_location="cpu"))
                G = G.to(device)
                syn_data, syn_labels = generate_synthetic_trials(
                    G, n_per_class=n_train // n_classes,
                    n_classes=n_classes, device=device
                )
                syn_ds = TensorDataset(syn_data, syn_labels)
                train_ds = ConcatDataset([train_ds, syn_ds])
                print(f"  Augmented training set: {len(train_ds)} trials "
                      f"(real + synthetic)")
            else:
                print("  No WGAN weights found — training without augmentation.")
        except Exception as e:
            print(f"  Augmentation skipped: {e}")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        run_loss = run_correct = run_total = 0
        for batch in train_loader:
            eeg, labels = batch
            eeg, labels = eeg.to(device), labels.to(device)
            logits = model(eeg)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            run_loss += loss.item()
            run_correct += (logits.argmax(1) == labels).sum().item()
            run_total += labels.size(0)
        scheduler.step()

        model.eval()
        val_loss = val_correct = val_total = 0
        with torch.no_grad():
            for batch in val_loader:
                eeg, labels = batch
                eeg, labels = eeg.to(device), labels.to(device)
                logits = model(eeg)
                val_loss += criterion(logits, labels).item()
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        train_acc = run_correct / run_total * 100
        val_acc   = val_correct / val_total * 100

        record = {
            "epoch": epoch,
            "train_loss": run_loss / len(train_loader),
            "train_acc":  train_acc,
            "val_loss":   val_loss / len(val_loader),
            "val_acc":    val_acc,
        }
        history.append(record)

        if verbose:
            print(
                f"[Classifier-{config_name}] epoch {epoch:3d}/{n_epochs} "
                f"| train acc={train_acc:5.1f}%  loss={run_loss/len(train_loader):.4f} "
                f"| val acc={val_acc:5.1f}%  loss={val_loss/len(val_loader):.4f}"
            )

    return model, history


if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser(
        description="Train EEG Transformer classifier for motor imagery."
    )
    parser.add_argument("--model",     default="small",
                        choices=list(CLASSIFIER_CONFIGS.keys()))
    parser.add_argument("--epochs",    type=int, default=30)
    parser.add_argument("--n_classes", type=int, default=4)
    parser.add_argument("--augment",   action="store_true")
    parser.add_argument("--quick",     action="store_true")
    args = parser.parse_args()

    if args.quick:
        config, epochs, n_train, n_val = "tiny", 3, 60, 20
        print("Running in --quick mode\n")
    else:
        config, epochs, n_train, n_val = args.model, args.epochs, 400, 80

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: EEGTransformer-{config}  epochs={epochs}  "
          f"n_classes={args.n_classes}  augment={args.augment}\n")

    model = build_classifier(config, n_classes=args.n_classes)
    print(f"Parameters: {model.count_parameters():,}\n")

    trained, history = train_classifier(
        config_name=config, n_epochs=epochs,
        n_classes=args.n_classes, augment=args.augment,
        n_train=n_train, n_val=n_val,
    )

    print("\nFinal metrics:")
    final = history[-1]
    print(f"  Train accuracy: {final['train_acc']:.1f}%")
    print(f"  Val   accuracy: {final['val_acc']:.1f}%")

    os.makedirs("results", exist_ok=True)
    save_path = f"results/classifier_{config}_trained.pt"
    torch.save(trained.state_dict(), save_path)
    print(f"\nSaved to: {save_path}")
