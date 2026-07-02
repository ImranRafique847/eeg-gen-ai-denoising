"""Quick diagnostic for RCVAE output quality."""
import torch
from rcvae import build_rcvae, train_rcvae
from eeg_dataset import EEGDenoisingDataset

# Train a tiny model for 5 epochs and inspect outputs directly
print("Training RCVAE-tiny for 5 epochs...")
model, history = train_rcvae(
    config_name="tiny", n_epochs=5, n_train=100, n_val=20, verbose=False
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device).eval()

ds = EEGDenoisingDataset(n_samples=4, n_channels=64, signal_length=256,
                          artifact_prob=0.6, seed=99)
noisy, clean = ds.noisy.to(device), ds.clean.to(device)

with torch.no_grad():
    recon, mu, lv = model(noisy)

print(f"\nInput  (noisy) - mean: {noisy.mean():.4f}  std: {noisy.std():.4f}  range: [{noisy.min():.3f}, {noisy.max():.3f}]")
print(f"Target (clean) - mean: {clean.mean():.4f}  std: {clean.std():.4f}  range: [{clean.min():.3f}, {clean.max():.3f}]")
print(f"Output (recon) - mean: {recon.mean():.4f}  std: {recon.std():.4f}  range: [{recon.min():.3f}, {recon.max():.3f}]")
print(f"\nLatent mu  - mean: {mu.mean():.4f}  std: {mu.std():.4f}")
print(f"Latent lv  - mean: {lv.mean():.4f}  std: {lv.std():.4f}")

mse_noisy = ((clean - noisy)**2).mean().item()
mse_recon = ((clean - recon)**2).mean().item()
print(f"\nMSE(clean, noisy): {mse_noisy:.4f}")
print(f"MSE(clean, recon): {mse_recon:.4f}")
print(f"Improvement:       {(1 - mse_recon/mse_noisy)*100:.1f}%")

# Manual SNR
sig_power  = (clean**2).mean().item()
noise_before = ((clean - noisy)**2).mean().item()
noise_after  = ((clean - recon)**2).mean().item()
import math
snr_before = 10 * math.log10(sig_power / (noise_before + 1e-10))
snr_after  = 10 * math.log10(sig_power / (noise_after  + 1e-10))
print(f"\nSNR before denoising: {snr_before:.2f} dB")
print(f"SNR after  denoising: {snr_after:.2f} dB")
print(f"SNR gain:             {snr_after - snr_before:.2f} dB")
