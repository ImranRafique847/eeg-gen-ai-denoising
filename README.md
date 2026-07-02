# EEG Generative-AI Denoising Benchmark
### Bitsolera Internship — "Ultra-Fast Distilled Generative Models for Real-Time Edge-Deployed BCIs"

Based on: **Han, Feng & Li (2026)**, *"Advancing brain-computer interfaces with generative AI: A review of state-of-the-art and future outlook"*, The Innovation Life 4(1):100198

---

## 1. What problem is this solving?

A Brain-Computer Interface (BCI) reads electrical brain activity (EEG) and decodes it into commands. Raw EEG is extremely noisy — low SNR, eye-blink and muscle artifacts, non-stationarity (fatigue, attention drift), and strong inter-subject variability.

**This project is NOT building a full BCI.** It solves one specific pipeline stage:

```
[EEG headset] → [THIS PROJECT: clean the signal] → [decode intent] → [action]
                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
```

Specifically, the proposal's **Objective 1** asks: *how does denoising quality trade off against inference latency?* That trade-off is measured, visualised, and analysed across 10 sub-problems and 10 model architectures.

---

## 2. What's built — 10 sub-problems, all implemented

The paper maps each EEG challenge to the generative model the literature says is best for it (Figure 5A). This project follows that mapping directly.

| # | Sub-problem | Paper's recommendation | File | Status |
|---|---|---|---|---|
| 1 | **Artifact Removal** (EOG/EMG) | GAN (primary) | `gan_artifact_removal.py` + `train_gan.py` | ✅ Trained |
| 1b | Artifact Removal (upgraded) | CNN-Transformer hybrid GAN | `dhct_gan.py` | ✅ Trained |
| 2 | **General Denoising** | Residual Conv VAE | `rcvae.py` | ✅ Trained |
| 3 | **Missing Signal Repair** | Transformer (mask-based) | `denoiseformer.py` | ✅ Trained |
| 4 | **Spatial Super-Resolution** | Graph GAN | `ganso_superres.py` | ✅ Trained |
| 5 | **Motor Imagery Classification** | Transformer classifier | `eeg_transformer_classifier.py` | ✅ Trained |
| 6 | **Emotion Recognition** | VAE-GAN dual-encoder | `vae_gan_emotion.py` | ✅ Trained |
| 7 | **Data Augmentation** (MI) | Conditional WGAN-GP | `wgan_augmentation.py` | ✅ Trained |
| 8a | **Domain Adaptation** (supervised) | WGAN-DA | `domain_adaptation.py` | ✅ Trained |
| 8b | **Domain Adaptation** (zero-shot) | MMDA-VAE | `domain_adaptation.py` | ✅ Trained |
| 9 | **Edge Deployment** | CompactVAE (depthwise separable) | `edge_deployment.py` | ✅ Trained |

---

## 3. Trained model weights (`results/`)

| File | Model | Config |
|---|---|---|
| `gan_tiny_trained.pt` | GAN Artifact Removal | tiny |
| `gan_small_trained.pt` | GAN Artifact Removal | small |
| `dhct_gan_tiny_trained.pt` | DHCT-GAN (CNN-Transformer) | tiny |
| `dhct_gan_small_trained.pt` | DHCT-GAN (CNN-Transformer) | small |
| `rcvae_tiny_trained.pt` | RCVAE General Denoising | tiny |
| `rcvae_small_trained.pt` | RCVAE General Denoising | small |
| `denoiseformer_tiny_trained.pt` | Denoiseformer Inpainting | tiny |
| `denoiseformer_small_trained.pt` | Denoiseformer Inpainting | small |
| `ganso_tiny_trained.pt` | GANSO Spatial Super-Res | tiny |
| `wgan_aug_tiny_trained.pt` | cWGAN-GP MI Augmentation | tiny |
| `wgan_da_tiny_trained.pt` | WGAN-DA Domain Adaptation | tiny |
| `mmda_vae_tiny_trained.pt` | MMDA-VAE Domain Adaptation | tiny |
| `vaegan_emotion_tiny_trained.pt` | VAE-GAN Emotion Recognition | tiny |
| `compact_vae_nano_trained.pt` | CompactVAE Edge | nano |
| `compact_vae_micro_trained.pt` | CompactVAE Edge | micro |
| `compact_vae_mini_trained.pt` | CompactVAE Edge | mini |
| `classifier_tiny_trained.pt` | EEG Transformer Classifier | tiny |

---

## 4. Project structure

```
eeg-gen-ai-denoising/
│
├── eeg_dataset.py              # Realistic synthetic EEG (bands, pink noise, artifacts, drift)
├── metrics.py                  # SNR, MSE, 1D-SSIM, Pearson correlation
│
├── gan_artifact_removal.py     # #1  Dual-branch GAN + dual discriminator
├── train_gan.py                # #1  WGAN-GP + L1 training loop for GAN
├── dhct_gan.py                 # #1b CNN-Transformer hybrid GAN
├── rcvae.py                    # #2  Residual Conv VAE — general denoising
├── denoiseformer.py            # #3  Transformer for missing segment repair
├── ganso_superres.py           # #4  Graph GAN — 16→64 channel super-resolution
├── eeg_transformer_classifier.py # #5 Motor imagery transformer classifier
├── vae_gan_emotion.py          # #6  Dual-encoder VAE-GAN for emotion recognition
├── wgan_augmentation.py        # #7  Conditional WGAN-GP for MI data augmentation
├── domain_adaptation.py        # #8  WGAN-DA + MMDA-VAE cross-subject transfer
├── edge_deployment.py          # #9  CompactVAE with depthwise-separable convs
│
├── compare_results.py          # Load all trained weights, evaluate side-by-side
├── pareto_plot.py              # Latency vs SNR Pareto frontier plot
├── main.py                     # Master runner — train / eval / pareto
│
├── results/                    # Trained .pt weights + plots/
│   └── plots/pareto_frontier.png
│
├── check_gpu.py                # Dev utility — verify CUDA is accessible
└── debug_rcvae.py              # Dev utility — RCVAE shape debugging
```

---

## 5. How to run

### Setup
```bash
pip install torch numpy scipy matplotlib tqdm
```

### Quick sanity-check (all models, tiny configs, 3 epochs)
```bash
python main.py --quick
```

### Train everything (small configs, 20 epochs each)
```bash
python main.py --stage train
```

### Train a single model
```bash
python main.py --stage train --model gan
python main.py --stage train --model rcvae
# Available: gan, dhct, rcvae, denoiseformer, wgan_aug, vaegan, ganso, da_wgan, da_mmda, compact
```

### Evaluate all trained models side-by-side
```bash
python compare_results.py
```
Outputs a table grouped by task (artifact removal, general denoising, inpainting) so models solving the same problem are compared directly.

### Regenerate Pareto frontier plot
```bash
python pareto_plot.py
```
Outputs `results/plots/pareto_frontier.png` — latency vs. SNR with 10 ms / 50 ms BCI budget lines.

### Train individual models directly
```bash
python train_gan.py --quick
python train_gan.py --model small --epochs 20

python rcvae.py --quick
python rcvae.py --model small --epochs 20

python denoiseformer.py --quick
python dhct_gan.py --quick
python edge_deployment.py --quick
python wgan_augmentation.py --quick
python vae_gan_emotion.py --quick
python domain_adaptation.py --method wgan --quick
python domain_adaptation.py --method mmda --quick
python ganso_superres.py --quick
python eeg_transformer_classifier.py --quick
```

---

## 6. Metrics

All denoising models are evaluated on four signal-quality metrics — not classification accuracy, which is not the right unit for a denoising task.

| Metric | What it measures | Ideal |
|---|---|---|
| **SNR (dB)** | Power of clean signal vs. residual noise | Higher |
| **SSIM** | Structural similarity (local waveform fidelity) | → 1.0 |
| **MSE** | Mean squared reconstruction error | → 0.0 |
| **Pearson r** | Linear correlation between clean and denoised | → 1.0 |

For the classifier (`eeg_transformer_classifier.py`) and domain adaptation models, **classification accuracy** is the reported metric instead.

---

## 7. Data

Everything trains on **realistic synthetic EEG** generated by `eeg_dataset.py`. The simulator produces:
- 5-band neural signal (delta/theta/alpha/beta/gamma) with a non-stationary amplitude envelope
- 1/f pink background noise at a configurable SNR
- Probabilistic EOG (eye-blink) and EMG (muscle burst) artifacts
- Banded spatial mixing matrix as a proxy for volume conduction

No real patient data is used. All results are therefore **proof-of-concept** — the architectures and training pipelines are validated, but SNR/accuracy numbers don't transfer directly to real EEG without retraining on clinical data (e.g. BCI Competition IV, DEAP, SEED).

---

## 8. Key design choices and paper grounding

- **GAN for artifact removal** — WGAN-GP is the most-validated objective for EEG augmentation in the paper's Table 2; dual discriminator (local patch + global) is from the AR-WGAN line of work.
- **RCVAE for general denoising** — fully convolutional (no FC bottleneck), residual noise prediction for stable init, KL annealing + free-bits against posterior collapse.
- **Denoiseformer** — mask token provided as extra input channel; masked positions excluded from attention keys/values; loss only on masked positions.
- **DHCT-GAN** — parallel CNN and Transformer streams in each branch, fused with a learned gate; captures inter-channel correlations (volume conduction) that pure Conv1D misses.
- **GANSO** — graph-based: banded adjacency matrix, `GraphConv1D` neighbourhood aggregation, bilinear channel upsampling + learned delta refinement.
- **MMDA-VAE** — MMD loss in latent space replaces adversarial domain discriminator for truly zero-shot transfer (no target labels needed).
- **CompactVAE** — depthwise-separable convolutions, no batch norm, benchmarked with GPU-warmed latency measurement.

---

## 9. What to do next

1. **Real data loaders** — add DEAP (emotion), BCI Competition IV (MI), and TUH EEG (pathology) loaders to replace the synthetic dataset.
2. **Medium/large configs** — only tiny and small variants are trained; medium configs exist in every `CONFIGS` dict but have no saved weights.
3. **Diffusion model** — the original proposal called for a DDPM baseline; `main.py` v1 referenced it. Adding a lightweight DDPM would complete the latency-vs-quality curve at the high-quality end.
4. **Cross-model head-to-head on identical data** — `compare_results.py` now groups by task, but Denoiseformer and RCVAE are evaluated on different data splits (inpainting vs. full denoising). A unified test set would make the comparison cleaner.
5. **Classifier downstream evaluation** — measure whether RCVAE/GAN denoising actually improves MI classification accuracy through `eeg_transformer_classifier.py`.
