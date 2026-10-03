"""Device selection for the PyTorch neural surrogates (MLP, FNO). Nothing else in the project uses it:
the pseudo-spectral solver, POD-Galerkin ROM and DEIM-ROM are NumPy and always run on the CPU.

Policy (see REMOTE_GPU.md):
  auto  the historical rule, unchanged: Apple MPS if available, otherwise CPU. CUDA is never chosen
        automatically, so every existing command keeps the backend its published results used.
  cpu / mps / cuda
        explicit; an unavailable accelerator raises instead of silently falling back to the CPU.

On CUDA, TF32 is switched off for matmul and cuDNN convolutions so the networks run in true float32,
as they do on CPU and MPS.
"""
import platform

import torch

DEVICE_CHOICES = ("auto", "cpu", "mps", "cuda")


def select_device(name: str = "auto") -> torch.device:
    if name not in DEVICE_CHOICES:
        raise ValueError(f"unknown device {name!r}; choose one of {DEVICE_CHOICES}")
    if name == "auto":
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        if device.type == "cpu" and torch.cuda.is_available():
            print("Note: CUDA is available but --device auto keeps the historical MPS-then-CPU rule; "
                  "pass --device cuda to use the GPU.")
        return device
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested but the MPS backend is not available on this machine")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False "
                               f"(torch {torch.__version__}, built with CUDA {torch.version.cuda})")
        _cuda_fp32()
    return torch.device(name)


def _cuda_fp32():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def timing_devices():
    """Backends for scripts that time the same model on every available device, each reported under
    its own label: always "cpu", then "mps" and "cuda" when present."""
    devs = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
    if torch.cuda.is_available():
        _cuda_fp32()
        devs.append("cuda")
    return devs


def add_device_argument(parser):
    parser.add_argument("--device", default="auto", choices=DEVICE_CHOICES,
                        help="neural-network backend (default auto: MPS if available, else CPU; "
                             "CUDA only when requested explicitly)")


def synchronize(device: torch.device) -> None:
    """Block until queued kernels on `device` finish (no-op on CPU)."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def device_info(device: torch.device) -> dict:
    """Hardware/software label stored with every result, so a number is never separated from the
    backend that produced it."""
    info = {"device": str(device), "torch_version": torch.__version__, "machine": platform.machine(),
            "platform": platform.platform()}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update({"gpu_name": props.name, "gpu_memory_gb": round(props.total_memory / 2**30, 1),
                     "compute_capability": f"{props.major}.{props.minor}",
                     "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                     "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                     "tf32_cudnn": torch.backends.cudnn.allow_tf32,
                     "cudnn_benchmark": torch.backends.cudnn.benchmark,
                     "cudnn_deterministic": torch.backends.cudnn.deterministic})
    elif device.type == "mps":
        info["accelerator"] = "Apple MPS"
    return info
