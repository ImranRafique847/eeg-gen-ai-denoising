"""
Export trained CompactVAE models to ONNX format for cross-platform deployment.
Enables inference via TensorRT (NVIDIA Jetson), OpenVINO (Intel), and ONNX Runtime (CPU/mobile).

Requires: pip install onnx onnxruntime
"""

import os
import torch
import numpy as np
from edge_deployment import build_compact_vae


def export_compact_vae_onnx(config_name="nano", output_dir="results/onnx"):
    """Export one CompactVAE config to ONNX."""
    os.makedirs(output_dir, exist_ok=True)

    weight_path = f"results/compact_vae_{config_name}_trained.pt"
    if not os.path.exists(weight_path):
        print(f"  Weights not found: {weight_path}")
        return None

    model = build_compact_vae(config_name, n_channels=64, signal_length=256)
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()

    # Wrapper: exposes deterministic denoise as forward() for ONNX tracing
    class DenoiseWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, noisy_eeg):
            mu, _ = self.vae.encode(noisy_eeg)       # deterministic (no reparameterisation)
            noise_pred = self.vae.decode(mu)
            return noisy_eeg - noise_pred

    wrapped = DenoiseWrapper(model)
    dummy   = torch.randn(1, 64, 256)
    path    = os.path.join(output_dir, f"compact_vae_{config_name}.onnx")

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            wrapped, dummy, path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=["noisy_eeg"],
            output_names=["denoised_eeg"],
            dynamic_axes={"noisy_eeg": {0: "batch_size"},
                          "denoised_eeg": {0: "batch_size"}},
        )
    print(f"  Exported: {path}")

    # Validate with onnx package if installed
    try:
        import onnx
        onnx.checker.check_model(onnx.load(path))
        print(f"  Graph validated (onnx {onnx.__version__})")
    except ImportError:
        print(f"  (install onnx to validate graph)")

    # Parity check with onnxruntime if installed
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        ref  = wrapped(dummy).detach().numpy()
        out  = sess.run(["denoised_eeg"], {"noisy_eeg": dummy.numpy()})[0]
        mse  = np.mean((ref - out) ** 2)
        print(f"  PyTorch vs ORT parity — MSE: {mse:.2e} "
              f"({'ok' if mse < 1e-4 else 'WARN: large diff'})")
    except ImportError:
        print(f"  (install onnxruntime to check inference parity)")
    except Exception as e:
        print(f"  ORT check error: {e}")

    return path


def export_all_models():
    print("\n" + "=" * 60)
    print("  ONNX Export — CompactVAE Models")
    print("=" * 60 + "\n")

    try:
        import onnx  # noqa: F401
    except ImportError:
        print("  onnx package not installed.")
        print("  Install with: pip install onnx")
        print("  Then re-run this script.\n")
        return

    for config in ["nano", "micro", "mini"]:
        print(f"CompactVAE-{config}:")
        export_compact_vae_onnx(config)
        print()

    print("Exported to: results/onnx/")
    print("\nNext steps:")
    print("  Jetson (TensorRT FP16):")
    print("    trtexec --onnx=results/onnx/compact_vae_nano.onnx --fp16 --saveEngine=compact_vae_nano_fp16.trt")
    print("  Intel CPU (OpenVINO):")
    print("    mo --input_model results/onnx/compact_vae_nano.onnx --output_dir results/openvino/")
    print("  Any platform (ONNX Runtime):")
    print("    session = ort.InferenceSession('results/onnx/compact_vae_nano.onnx')")


if __name__ == "__main__":
    export_all_models()

import os
import torch
import numpy as np
from edge_deployment import build_compact_vae


def export_compact_vae_onnx(config_name="nano", output_dir="results/onnx"):
    """
    Export CompactVAE to ONNX format.
    Requires: pip install onnx onnxruntime
    """
    os.makedirs(output_dir, exist_ok=True)

    weight_path = f"results/compact_vae_{config_name}_trained.pt"
    if not os.path.exists(weight_path):
        print(f"Weights not found: {weight_path}")
        return

    model = build_compact_vae(config_name, n_channels=64, signal_length=256)
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    model.eval()

    class DenoiseWrapper(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, noisy_eeg):
            # run denoise logic inline (no .eval() side effects in export)
            mu, lv = self.vae.encode(noisy_eeg)
            noise_pred = self.vae.decode(mu)
            return noisy_eeg - noise_pred

    wrapped = DenoiseWrapper(model)
    dummy_input = torch.randn(1, 64, 256)

    onnx_path = os.path.join(output_dir, f"compact_vae_{config_name}.onnx")

    torch.onnx.export(
        wrapped,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["noisy_eeg"],
        output_names=["denoised_eeg"],
        dynamic_axes={
            "noisy_eeg":    {0: "batch_size"},
            "denoised_eeg": {0: "batch_size"},
        }
    )
    print(f"Exported: {onnx_path}")

    # Validate with onnx package if available
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"  ONNX model validated (onnx {onnx.__version__})")
    except ImportError:
        print(f"  Skipping validation (pip install onnx to enable)")

    # Test inference parity with onnxruntime if available
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        test_input = torch.randn(1, 64, 256)
        pytorch_out = wrapped(test_input).detach().numpy()
        onnx_out    = session.run(["denoised_eeg"], {"noisy_eeg": test_input.numpy()})[0]
        mse      = np.mean((pytorch_out - onnx_out) ** 2)
        max_diff = np.abs(pytorch_out - onnx_out).max()
        print(f"  PyTorch vs ONNX Runtime — MSE: {mse:.2e}  max_diff: {max_diff:.2e}")
        if mse < 1e-5:
            print(f"  Output parity verified")
        else:
            print(f"  Warning: larger-than-expected difference")
    except ImportError:
        print(f"  Skipping runtime check (pip install onnxruntime to enable)")

    return onnx_path


def export_all_models():
    """Export all three CompactVAE variants to ONNX."""
    print("\n" + "=" * 60)
    print("  ONNX Export — CompactVAE Models")
    print("=" * 60 + "\n")
    
    for config in ["nano", "micro", "mini"]:
        print(f"CompactVAE-{config}:")
        export_compact_vae_onnx(config)
        print()
    
    print("=" * 60)
    print("All models exported to: results/onnx/")
    print("\nNext steps:")
    print("  1. Test on target hardware with ONNX Runtime")
    print("  2. Convert to TensorRT (Jetson): trtexec --onnx=<file>.onnx --fp16")
    print("  3. Apply INT8 quantization for CPU deployment")
    print("=" * 60)


if __name__ == "__main__":
    export_all_models()
