"""
Training loop for GAN-based EEG artifact removal.
Uses WGAN-GP + L1 reconstruction loss on paired (noisy, clean) trials.
"""

import argparse
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from eeg_dataset import EEGDenoisingDataset
from gan_artifact_removal import (
    build_generator,
    build_discriminator,
    gradient_penalty,
    GAN_MODEL_CONFIGS,
)
from metrics import evaluate_all


N_CHANNELS = 64
SIGNAL_LENGTH = 256
BATCH_SIZE = 16
N_TRAIN_SAMPLES = 400
N_VAL_SAMPLES = 80

LAMBDA_L1 = 10.0       # weight on the reconstruction term
LAMBDA_GP = 10.0       # standard WGAN-GP penalty weight
N_CRITIC_STEPS = 2     # discriminator updates per generator update
LR_G = 2e-4
LR_D = 2e-4


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_dataloaders(seed: int = 0, n_train: int = N_TRAIN_SAMPLES,
                       n_val: int = N_VAL_SAMPLES):
    """Builds train/val loaders using artifact-only trials (artifact_prob=1.0)."""
    train_ds = EEGDenoisingDataset(
        n_samples=n_train,
        n_channels=N_CHANNELS,
        signal_length=SIGNAL_LENGTH,
        artifact_prob=1.0,
        seed=seed,
    )
    val_ds = EEGDenoisingDataset(
        n_samples=n_val,
        n_channels=N_CHANNELS,
        signal_length=SIGNAL_LENGTH,
        artifact_prob=1.0,
        seed=seed + 1,
    )
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader


def train_one_epoch(generator, discriminator, loader, opt_g, opt_d, device):
    """One full pass: N_CRITIC_STEPS discriminator updates then one generator update."""
    generator.train()
    discriminator.train()
    l1_loss_fn = nn.L1Loss()

    running_g_loss, running_d_loss = 0.0, 0.0
    n_batches = 0

    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)

        for _ in range(N_CRITIC_STEPS):
            with torch.no_grad():
                fake = generator(noisy)

            d_real_local, d_real_global = discriminator(clean)
            d_fake_local, d_fake_global = discriminator(fake)

            gp = gradient_penalty(discriminator, clean, fake, device)

            d_loss = (
                -(d_real_local.mean() - d_fake_local.mean())
                - (d_real_global.mean() - d_fake_global.mean())
                + LAMBDA_GP * gp
            )

            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

        fake = generator(noisy)
        d_fake_local, d_fake_global = discriminator(fake)

        adv_loss = -(d_fake_local.mean() + d_fake_global.mean())
        l1_loss = l1_loss_fn(fake, clean)
        g_loss = adv_loss + LAMBDA_L1 * l1_loss

        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

        running_g_loss += g_loss.item()
        running_d_loss += d_loss.item()
        n_batches += 1

    return running_g_loss / n_batches, running_d_loss / n_batches


@torch.no_grad()
def validate(generator, loader, device):
    """Evaluate generator on held-out validation set."""
    generator.eval()
    all_metrics = {"snr_db": [], "mse": [], "ssim": [], "pearson": []}

    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        denoised = generator(noisy)
        m = evaluate_all(clean, denoised)
        for k, v in m.items():
            all_metrics[k].append(v)

    return {k: sum(v) / len(v) for k, v in all_metrics.items()}


def train(config_name: str, n_epochs: int, seed: int = 0, verbose: bool = True,
          n_train: int = N_TRAIN_SAMPLES, n_val: int = N_VAL_SAMPLES):
    """Full training run for one model size. Returns trained generator and history."""
    device = get_device()
    generator = build_generator(config_name, n_channels=N_CHANNELS).to(device)
    discriminator = build_discriminator(config_name, n_channels=N_CHANNELS).to(device)

    opt_g = torch.optim.Adam(generator.parameters(), lr=LR_G, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=LR_D, betas=(0.5, 0.9))

    train_loader, val_loader = build_dataloaders(seed=seed, n_train=n_train, n_val=n_val)

    history = []
    for epoch in range(1, n_epochs + 1):
        t0 = time.perf_counter()
        g_loss, d_loss = train_one_epoch(
            generator, discriminator, train_loader, opt_g, opt_d, device
        )
        val_metrics = validate(generator, val_loader, device)
        elapsed = time.perf_counter() - t0

        record = {
            "epoch": epoch,
            "g_loss": g_loss,
            "d_loss": d_loss,
            "elapsed_sec": elapsed,
            **val_metrics,
        }
        history.append(record)

        if verbose:
            print(
                f"[{config_name}] epoch {epoch:3d}/{n_epochs} "
                f"| g_loss={g_loss:7.3f} d_loss={d_loss:7.3f} "
                f"| val SNR={val_metrics['snr_db']:6.2f}dB "
                f"SSIM={val_metrics['ssim']:.4f} "
                f"| {elapsed:.1f}s"
            )

    return generator, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the GAN-based EEG artifact-removal generator."
    )
    parser.add_argument(
        "--model", default="small", choices=list(GAN_MODEL_CONFIGS.keys()),
    )
    parser.add_argument(
        "--epochs", type=int, default=20,
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Fast sanity-check run: 2 epochs, tiny model, small dataset.",
    )
    args = parser.parse_args()

    if args.quick:
        print("Running in --quick mode (sanity check, not a real training run)\n")
        model_name = "tiny"
        epochs = 2
        n_train, n_val = 40, 16
    else:
        model_name = args.model
        epochs = args.epochs
        n_train, n_val = N_TRAIN_SAMPLES, N_VAL_SAMPLES

    print(f"Device: {get_device()}")
    print(f"Training '{model_name}' generator for {epochs} epoch(s) "
          f"on artifact-only EEG trials...\n")

    trained_generator, history = train(
        model_name, epochs, n_train=n_train, n_val=n_val
    )

    print("\nFinal validation metrics:")
    final = history[-1]
    for k in ("snr_db", "mse", "ssim", "pearson"):
        print(f"  {k:10s}: {final[k]:.4f}")

    save_path = f"results/gan_{model_name}_trained.pt"
    import os
    os.makedirs("results", exist_ok=True)
    torch.save(trained_generator.state_dict(), save_path)
    print(f"\nSaved trained generator weights to: {save_path}")
