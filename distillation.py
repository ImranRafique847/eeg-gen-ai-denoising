"""
Objective 2 — Knowledge distillation from DHCT-GAN-small (teacher) to a
lightweight student that runs in a single forward pass.

Teacher: DHCTGenerator-small  (CNN + Transformer hybrid, ~181k params)
Student: A pure-CNN generator with the same I/O signature but no
         Transformer blocks — faster and smaller, trained to mimic the
         teacher's output distribution rather than ground-truth directly.

Loss = alpha * MSE(student_out, teacher_out)        (distillation)
     + (1-alpha) * L1(student_out, clean_ground_truth)  (supervision)

Using both terms: the teacher provides soft targets that encode its
learned signal/artifact separation, while L1 to clean keeps the student
grounded when the teacher is occasionally wrong.
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from eeg_dataset import EEGDenoisingDataset
from dhct_gan import build_dhct_generator
from gan_artifact_removal import ConvBlock1D
from metrics import evaluate_all


# ── Student architecture ─────────────────────────────────────────────

class StudentGenerator(nn.Module):
    """
    Pure-CNN student — same divide-and-fuse structure as the teacher's
    DHCTGenerator but without any Transformer blocks. Much faster to run.
    """

    def __init__(self, n_channels: int = 64, base_dim: int = 16, depth: int = 3):
        super().__init__()

        def make_branch():
            layers = [ConvBlock1D(n_channels, base_dim)]
            for _ in range(depth - 1):
                layers.append(ConvBlock1D(base_dim, base_dim))
            layers.append(nn.Conv1d(base_dim, n_channels, 3, padding=1))
            return nn.Sequential(*layers)

        self.clean_branch    = make_branch()
        self.artifact_branch = make_branch()

        self.fusion_gate = nn.Sequential(
            nn.Conv1d(n_channels * 2, base_dim, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(base_dim, n_channels, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        clean_est    = self.clean_branch(x)
        artifact_est = self.artifact_branch(x)
        gate = self.fusion_gate(torch.cat([clean_est, artifact_est], dim=1))
        return gate * clean_est + (1 - gate) * (x - artifact_est)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── Latency helper ───────────────────────────────────────────────────

def measure_ms(model, device, n_warmup=20, n_runs=100):
    model.eval()
    dummy = torch.randn(1, 64, 256, device=device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)
    if device == "cuda":
        torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
    import numpy as np
    return float(np.mean(times)), float(np.percentile(times, 95))


# ── Training ─────────────────────────────────────────────────────────

def train_distillation(
    n_epochs: int = 20,
    batch_size: int = 16,
    lr: float = 2e-4,
    alpha: float = 0.7,       # weight on distillation vs ground-truth loss
    n_train: int = 400,
    n_val: int = 80,
    seed: int = 0,
    verbose: bool = True,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load frozen teacher
    teacher = build_dhct_generator("small", n_channels=64).to(device)
    teacher.load_state_dict(
        torch.load("results/dhct_gan_small_trained.pt", map_location=device)
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = StudentGenerator(n_channels=64, base_dim=16, depth=3).to(device)
    optimizer = torch.optim.Adam(student.parameters(), lr=lr, betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.1
    )

    # artifact-only data — same domain the teacher was trained on
    train_ds = EEGDenoisingDataset(
        n_samples=n_train, n_channels=64, signal_length=256,
        artifact_prob=1.0, seed=seed
    )
    val_ds = EEGDenoisingDataset(
        n_samples=n_val, n_channels=64, signal_length=256,
        artifact_prob=1.0, seed=seed + 1
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    history = []
    for epoch in range(1, n_epochs + 1):
        student.train()
        run_loss = 0.0

        for noisy, clean in train_loader:
            noisy, clean = noisy.to(device), clean.to(device)

            with torch.no_grad():
                teacher_out = teacher(noisy)

            student_out = student(noisy)

            # distillation: match teacher's output
            loss_distill = F.mse_loss(student_out, teacher_out)
            # supervision: stay close to ground truth
            loss_gt      = F.l1_loss(student_out, clean)
            loss = alpha * loss_distill + (1 - alpha) * loss_gt

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            run_loss += loss.item()

        scheduler.step()

        student.eval()
        val_m = {"snr_db": [], "mse": [], "ssim": [], "pearson": []}
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                out = student(noisy)
                m = evaluate_all(clean, out)
                for k, v in m.items():
                    val_m[k].append(v)
        val_avg = {k: sum(v) / len(v) for k, v in val_m.items()}

        record = {"epoch": epoch, "loss": run_loss / len(train_loader), **val_avg}
        history.append(record)

        if verbose:
            print(
                f"[Distill] epoch {epoch:3d}/{n_epochs} "
                f"| loss={run_loss/len(train_loader):.4f} "
                f"| SNR={val_avg['snr_db']:.2f}dB "
                f"SSIM={val_avg['ssim']:.4f}"
            )

    return student, history


# ── Comparison report ────────────────────────────────────────────────

def compare_teacher_student(student, device):
    """Evaluate both models on the same held-out val set and print a table."""
    teacher = build_dhct_generator("small", n_channels=64).to(device)
    teacher.load_state_dict(
        torch.load("results/dhct_gan_small_trained.pt", map_location=device)
    )
    teacher.eval()

    val_ds = EEGDenoisingDataset(
        n_samples=80, n_channels=64, signal_length=256,
        artifact_prob=1.0, seed=999
    )
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    results = {}
    for name, model in [("teacher (DHCT-GAN-small)", teacher),
                        ("student (distilled CNN)",  student)]:
        model.eval()
        m_all = {"snr_db": [], "mse": [], "ssim": [], "pearson": []}
        with torch.no_grad():
            for noisy, clean in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                out = model(noisy)
                m = evaluate_all(clean, out)
                for k, v in m.items():
                    m_all[k].append(v)
        results[name] = {k: sum(v) / len(v) for k, v in m_all.items()}

    # latency
    lat_t = measure_ms(teacher, device)
    lat_s = measure_ms(student, device)

    params_t = sum(p.numel() for p in teacher.parameters())
    params_s = sum(p.numel() for p in student.parameters())

    print("\n" + "=" * 68)
    print(f"  {'Model':<30} {'SNR':>7} {'SSIM':>7} {'MSE':>8} {'Pearson':>8}")
    print("=" * 68)
    for name, m in results.items():
        print(f"  {name:<30} {m['snr_db']:>7.2f} {m['ssim']:>7.4f} "
              f"{m['mse']:>8.4f} {m['pearson']:>8.4f}")
    print("=" * 68)

    t_snr = results["teacher (DHCT-GAN-small)"]["snr_db"]
    s_snr = results["student (distilled CNN)"]["snr_db"]
    print(f"\n  SNR gap    : {t_snr - s_snr:+.2f} dB  (teacher - student)")
    print(f"  Latency    : teacher={lat_t[0]:.2f}ms  student={lat_s[0]:.2f}ms  "
          f"speedup={lat_t[0]/lat_s[0]:.1f}x")
    print(f"  Parameters : teacher={params_t:,}  student={params_s:,}  "
          f"reduction={1 - params_s/params_t:.1%}")

    return results, lat_t, lat_s


# ── Entry point ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--quick",  action="store_true")
    args = parser.parse_args()

    if args.quick:
        epochs, n_train, n_val = 3, 80, 20
        print("Quick mode\n")
    else:
        epochs, n_train, n_val = args.epochs, 400, 80

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    student = StudentGenerator()
    print(f"Student parameters: {student.count_parameters():,}\n")

    trained_student, history = train_distillation(
        n_epochs=epochs, n_train=n_train, n_val=n_val
    )

    os.makedirs("results", exist_ok=True)
    torch.save(trained_student.state_dict(), "results/distilled_student.pt")
    print("\nSaved: results/distilled_student.pt")

    compare_teacher_student(trained_student, device)
