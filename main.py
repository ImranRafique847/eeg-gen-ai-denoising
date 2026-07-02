"""
Master runner for the EEG generative AI denoising benchmark.
Orchestrates training, evaluation, and Pareto analysis across all 10 sub-problems.

Usage:
    python main.py --quick              # fast sanity-check of all models
    python main.py --stage train        # re-train every model
    python main.py --stage eval         # evaluate all trained models
    python main.py --stage pareto       # regenerate Pareto frontier plot
    python main.py --stage all          # train -> eval -> pareto (default)
    python main.py --model gan          # run a single model only
"""

import argparse
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(__file__))

MODEL_REGISTRY = {
    "gan":       "gan_artifact_removal  (Artifact Removal)",
    "dhct":      "dhct_gan              (CNN-Transformer Artifact Removal)",
    "rcvae":     "rcvae                 (General Denoising)",
    "denoiseformer": "denoiseformer     (Missing Segment Inpainting)",
    "wgan_aug":  "wgan_augmentation     (MI Data Augmentation)",
    "vaegan":    "vae_gan_emotion        (Emotion Recognition)",
    "ganso":     "ganso_superres         (Spatial Super-Resolution)",
    "da_wgan":   "domain_adaptation WGAN-DA (Supervised Transfer)",
    "da_mmda":   "domain_adaptation MMDA-VAE (Unsupervised Transfer)",
    "compact":   "edge_deployment        (Edge Compact VAE)",
}


def run_training(model_filter: str | None, quick: bool):
    """Train all models (or a single one if model_filter is set)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = "tiny" if quick else "small"
    epochs = 3     if quick else 20
    n_train = 80   if quick else 400
    print(f"\nTraining config: {config}, epochs={epochs}, n_train={n_train}, device={device}\n")

    def should_run(key):
        return model_filter is None or model_filter == key

    if should_run("gan"):
        print("# GAN artifact removal")
        from train_gan import train
        train(config_name=config, n_epochs=epochs, n_train=n_train)

    if should_run("dhct"):
        print("# DHCT-GAN (CNN-Transformer)")
        from dhct_gan import train_dhct
        train_dhct(config_name=config, n_epochs=epochs, n_train=n_train)

    if should_run("rcvae"):
        print("# RCVAE general denoising")
        from rcvae import train_rcvae
        train_rcvae(config_name=config, n_epochs=epochs, n_train=n_train)

    if should_run("denoiseformer"):
        print("# Denoiseformer inpainting")
        from denoiseformer import train_denoiseformer
        train_denoiseformer(config_name=config, n_epochs=epochs, n_train=n_train)

    if should_run("wgan_aug"):
        print("# cWGAN-GP MI augmentation")
        from wgan_augmentation import train_wgan_aug
        train_wgan_aug(config_name=config, n_epochs=epochs, n_train=n_train)

    if should_run("vaegan"):
        print("# VAE-GAN emotion recognition")
        from vae_gan_emotion import train_vae_gan
        train_vae_gan(config_name=config, n_epochs=epochs, n_train=n_train)

    if should_run("ganso"):
        print("# GANSO spatial super-resolution")
        from ganso_superres import train_ganso
        train_ganso(config_name=config, n_epochs=epochs, n_train=n_train)

    if should_run("da_wgan"):
        print("# WGAN-DA supervised transfer")
        from domain_adaptation import train_wgan_da, DA_CONFIGS
        train_wgan_da(config=DA_CONFIGS[config], n_epochs=epochs, n_train=n_train)

    if should_run("da_mmda"):
        print("# MMDA-VAE unsupervised transfer")
        from domain_adaptation import train_mmda_vae, DA_CONFIGS
        train_mmda_vae(config=DA_CONFIGS[config], n_epochs=epochs, n_train=n_train)

    if should_run("compact"):
        print("# CompactVAE edge deployment")
        from edge_deployment import train_compact_vae
        nano_cfg = "nano" if quick else "mini"
        train_compact_vae(config_name=nano_cfg, n_epochs=epochs, n_train=n_train)


def run_evaluation():
    """Load all trained weights and print comparison table."""
    print("\n" + "=" * 60)
    print(" Running compare_results.py")
    print("=" * 60)
    import compare_results  # noqa: F401 — side-effectful, prints table


def run_pareto():
    """Regenerate the latency vs. SNR Pareto frontier plot."""
    print("\n" + "=" * 60)
    print(" Regenerating Pareto frontier plot")
    print("=" * 60)
    import pareto_plot  # noqa: F401 — side-effectful, saves PNG
    pareto_plot.main()


def main():
    parser = argparse.ArgumentParser(
        description="Master runner for EEG Generative AI Denoising benchmark."
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["train", "eval", "pareto", "all"],
    )
    parser.add_argument(
        "--model",
        default=None,
        choices=list(MODEL_REGISTRY.keys()),
        help="Run a single model only (training stage)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Tiny configs + 3 epochs for fast sanity-checks",
    )
    args = parser.parse_args()

    os.makedirs("results/plots", exist_ok=True)

    print("\nEEG Gen-AI Denoising Benchmark")
    print("10 sub-problems | GANs | VAEs | Transformers | Edge")
    print(f"\nStage: {args.stage}  |  Quick: {args.quick}  |"
          f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}\n")

    if args.stage in ("train", "all"):
        run_training(args.model, args.quick)

    if args.stage in ("eval", "all"):
        run_evaluation()

    if args.stage in ("pareto", "all"):
        run_pareto()

    print("\nDone. Check results/ for trained weights and results/plots/ for figures.")


if __name__ == "__main__":
    main()
