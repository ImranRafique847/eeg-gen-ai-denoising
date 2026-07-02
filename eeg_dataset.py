"""
Synthetic EEG dataset with realistic noise, artifacts, and spatial mixing.
Generates paired (noisy, clean) trials for denoising model training.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, filtfilt


# EEG frequency bands (Hz) — standard clinical convention
EEG_BANDS = {
    "delta": (0.5, 4.0,  0.40),   # (low, high, relative power weight)
    "theta": (4.0, 8.0,  0.25),
    "alpha": (8.0, 13.0, 0.20),
    "beta":  (13.0, 30.0, 0.10),
    "gamma": (30.0, 50.0, 0.05),
}


def _bandpass(signal: np.ndarray, low: float, high: float, fs: int,
              order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter."""
    nyq = fs / 2.0
    low_n = max(low / nyq, 1e-4)
    high_n = min(high / nyq, 0.999)
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, signal)


def _pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """1/f pink noise via spectral shaping of white noise."""
    white = rng.normal(0, 1, n_samples)
    fft_vals = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = freqs[1]                       # avoid div-by-zero at DC
    scale = 1.0 / np.sqrt(freqs)
    fft_vals *= scale
    pink = np.fft.irfft(fft_vals, n=n_samples)
    return pink / (np.std(pink) + 1e-8)


def _nonstationary_envelope(n_samples: int, fs: int,
                             rng: np.random.Generator,
                             drift_strength: float = 0.3) -> np.ndarray:
    """Slow amplitude modulation simulating fatigue/attention drift."""
    n_segments = 6
    control_pts = 1.0 + drift_strength * rng.normal(0, 1, n_segments + 1)
    control_pts = np.clip(control_pts, 0.4, 1.8)
    t_ctrl = np.linspace(0, n_samples, n_segments + 1)
    t_full = np.arange(n_samples)
    envelope = np.interp(t_full, t_ctrl, control_pts)
    return envelope


def _eog_blink_artifact(n_samples: int, fs: int,
                         rng: np.random.Generator,
                         n_blinks: int = 2) -> np.ndarray:
    """Simulate eye-blink (EOG) artifacts: large slow exponentially-decaying deflections."""
    artifact = np.zeros(n_samples)
    for _ in range(n_blinks):
        center = rng.integers(int(0.1 * n_samples), int(0.9 * n_samples))
        width = rng.integers(int(0.05 * fs), int(0.15 * fs))
        amplitude = rng.uniform(3.0, 6.0) * rng.choice([-1, 1])
        t = np.arange(n_samples) - center
        blink = amplitude * np.exp(-(t ** 2) / (2 * (width ** 2)))
        artifact += blink
    return artifact


def _emg_burst_artifact(n_samples: int, fs: int,
                         rng: np.random.Generator,
                         n_bursts: int = 1) -> np.ndarray:
    """Simulate muscle (EMG) artifacts: short bursts of high-frequency noise."""
    artifact = np.zeros(n_samples)
    min_burst_len = 40   # filtfilt needs > ~3*(filter order+1); keep margin
    for _ in range(n_bursts):
        start = rng.integers(0, max(n_samples - int(0.2 * fs), 1))
        dur = rng.integers(int(0.05 * fs), int(0.2 * fs))
        dur = max(dur, min_burst_len)
        end = min(start + dur, n_samples)
        if end - start < min_burst_len:
            start = max(0, end - min_burst_len)
        burst = rng.normal(0, 1, end - start)
        if end - start > min_burst_len:
            burst = _bandpass(burst, 30, min(100, fs / 2 - 1), fs, order=2)
        window = np.hanning(end - start)
        artifact[start:end] += 2.5 * burst * window
    return artifact


def _spatial_mixing_matrix(n_channels: int,
                            rng: np.random.Generator) -> np.ndarray:
    """
    Banded mixing matrix as a crude proxy for volume conduction.
    Channel i correlates with i±1 and i±2.
    """
    mix = np.eye(n_channels)
    for offset, weight in [(1, 0.35), (2, 0.15)]:
        for i in range(n_channels):
            j_minus = i - offset
            j_plus = i + offset
            if j_minus >= 0:
                mix[i, j_minus] += weight
            if j_plus < n_channels:
                mix[i, j_plus] += weight
    # row-normalise so overall channel power stays comparable
    mix /= mix.sum(axis=1, keepdims=True)
    return mix


def generate_synthetic_eeg(
    n_samples: int = 500,
    n_channels: int = 64,
    signal_length: int = 256,
    fs: int = 256,
    snr_db: float = 5.0,
    artifact_prob: float = 0.6,
    nonstationary: bool = True,
    seed: int = 42,
):
    """
    Generate synthetic EEG trials with band structure, 1/f noise,
    spatial mixing, amplitude drift, and probabilistic EOG/EMG artifacts.

    Returns clean and noisy tensors of shape (n_samples, n_channels, signal_length).
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, signal_length / fs, signal_length, endpoint=False)
    mix_matrix = _spatial_mixing_matrix(n_channels, rng)

    clean = np.zeros((n_samples, n_channels, signal_length), dtype=np.float64)
    noisy = np.zeros((n_samples, n_channels, signal_length), dtype=np.float64)

    for s in range(n_samples):
        envelope = (
            _nonstationary_envelope(signal_length, fs, rng)
            if nonstationary else np.ones(signal_length)
        )
        has_artifact = rng.random() < artifact_prob

        trial_clean = np.zeros((n_channels, signal_length))
        trial_noise = np.zeros((n_channels, signal_length))

        for c in range(n_channels):
            sig = np.zeros(signal_length)
            for _, (low, high, weight) in EEG_BANDS.items():
                freq = rng.uniform(low, high)
                phase = rng.uniform(0, 2 * np.pi)
                amp_jitter = rng.uniform(0.7, 1.3) * weight
                sig += amp_jitter * np.sin(2 * np.pi * freq * t + phase)
            sig *= envelope
            sig /= (np.std(sig) + 1e-8)
            trial_clean[c] = sig

            pink = _pink_noise(signal_length, rng)
            signal_power = np.mean(sig ** 2)
            snr_linear = 10 ** (snr_db / 10.0)
            noise_power = signal_power / snr_linear
            pink = pink * np.sqrt(noise_power)
            trial_noise[c] = pink

        trial_clean = mix_matrix @ trial_clean
        trial_noise = mix_matrix @ trial_noise

        trial_noisy = trial_clean + trial_noise

        # add physiological artifacts to the noisy version only
        if has_artifact:
            frontal = slice(0, max(n_channels // 3, 1))
            blink = _eog_blink_artifact(signal_length, fs, rng,
                                        n_blinks=rng.integers(1, 3))
            trial_noisy[frontal] += blink[None, :] * rng.uniform(
                0.6, 1.0, size=(trial_noisy[frontal].shape[0], 1)
            )

            if rng.random() < 0.5:
                emg = _emg_burst_artifact(signal_length, fs, rng)
                affected = rng.choice(
                    n_channels, size=max(n_channels // 4, 1), replace=False
                )
                trial_noisy[affected] += emg[None, :]

        clean[s] = trial_clean
        noisy[s] = trial_noisy

    return (
        torch.from_numpy(clean).float(),
        torch.from_numpy(noisy).float(),
    )


class EEGDenoisingDataset(Dataset):
    """PyTorch Dataset wrapping synthetic EEG clean/noisy pairs."""

    def __init__(
        self,
        n_samples: int = 500,
        n_channels: int = 64,
        signal_length: int = 256,
        snr_db: float = 5.0,
        artifact_prob: float = 0.6,
        nonstationary: bool = True,
        seed: int = 42,
    ):
        self.clean, self.noisy = generate_synthetic_eeg(
            n_samples=n_samples,
            n_channels=n_channels,
            signal_length=signal_length,
            snr_db=snr_db,
            artifact_prob=artifact_prob,
            nonstationary=nonstationary,
            seed=seed,
        )

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, idx):
        return self.noisy[idx], self.clean[idx]


if __name__ == "__main__":
    ds = EEGDenoisingDataset(n_samples=10)
    noisy, clean = ds[0]
    print(f"Sample shape  — noisy: {noisy.shape}, clean: {clean.shape}")
    print(f"Noisy range   : [{noisy.min():.3f}, {noisy.max():.3f}]")
    print(f"Clean range   : [{clean.min():.3f}, {clean.max():.3f}]")
    measured_snr = 10 * np.log10(
        (clean ** 2).mean().item() / ((clean - noisy) ** 2).mean().item()
    )
    print(f"Measured SNR  : {measured_snr:.2f} dB")
