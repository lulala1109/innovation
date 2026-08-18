"""
Reproducibility helpers for attack experiments.
"""

import hashlib
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


DETERMINISM_MODES = {"off", "warn", "strict"}


def configure_reproducibility(
    seed: int,
    mode: str = "off"
) -> Dict[str, Any]:
    """
    Seed global RNGs and optionally require deterministic PyTorch algorithms.

    This function must run before the first CUDA operation. A separate,
    device-local PGD generator should still be used for random initialization
    so its state is isolated from unrelated global RNG consumption.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    if mode not in DETERMINISM_MODES:
        raise ValueError(
            f"mode must be one of {sorted(DETERMINISM_MODES)}"
        )

    deterministic = mode != "off"
    if deterministic:
        # Required by deterministic CUDA matrix multiplication. It must be set
        # before CUDA context initialization.
        os.environ.setdefault(
            "CUBLAS_WORKSPACE_CONFIG",
            ":4096:8"
        )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic

    if deterministic:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    torch.use_deterministic_algorithms(
        deterministic,
        warn_only=(mode == "warn")
    )

    return {
        "seed": seed,
        "mode": mode,
        "deterministic_algorithms":
            torch.are_deterministic_algorithms_enabled(),
        "deterministic_debug_mode":
            int(torch.get_deterministic_debug_mode()),
        "cublas_workspace_config":
            os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark":
            bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic":
            bool(torch.backends.cudnn.deterministic),
        "matmul_allow_tf32":
            bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32":
            bool(torch.backends.cudnn.allow_tf32),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_value(
    project_root: Path,
    *args: str
) -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None

    if result.returncode != 0:
        return None
    return result.stdout.strip()


def collect_run_metadata(
    model: Any,
    wav_path: Optional[str],
    reproducibility: Dict[str, Any],
    project_root: str
) -> Dict[str, Any]:
    """Collect a JSON-safe whitelist of experiment and environment metadata."""
    root = Path(project_root).resolve()

    input_metadata: Dict[str, Any] = {
        "path": wav_path,
        "sha256": None,
    }
    if wav_path:
        resolved_wav = (root / wav_path).resolve()
        if resolved_wav.is_file():
            input_metadata["path"] = str(resolved_wav)
            input_metadata["sha256"] = _sha256_file(resolved_wav)

    underlying_model = getattr(model, "model", None)
    model_config = getattr(underlying_model, "config", None)
    checkpoint = getattr(model, "model_id", None)
    if checkpoint is None and model_config is not None:
        checkpoint = getattr(model_config, "_name_or_path", None)

    gpu_metadata = None
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        gpu_metadata = {
            "index": int(device_index),
            "name": torch.cuda.get_device_name(device_index),
            "compute_capability": list(
                torch.cuda.get_device_capability(device_index)
            ),
        }

    commit = _git_value(root, "rev-parse", "HEAD")
    dirty_output = _git_value(root, "status", "--porcelain")

    try:
        import transformers
        transformers_version = transformers.__version__
    except (ImportError, AttributeError):
        transformers_version = None

    return {
        "reproducibility": dict(reproducibility),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "transformers": transformers_version,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "gpu": gpu_metadata,
            "cuda_visible_devices":
                os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pytorch_cuda_alloc_conf":
                os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "cublas_workspace_config":
                os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "pythonhashseed":
                os.environ.get("PYTHONHASHSEED"),
        },
        "model_runtime": {
            "checkpoint": checkpoint,
            "device": str(getattr(model, "device", None)),
            "dtype": str(getattr(model, "dtype", None)),
            "generation": {
                "decoding": "greedy",
                "do_sample": False,
            },
        },
        "input_audio": input_metadata,
        "source": {
            "git_commit": commit,
            "git_dirty": (
                bool(dirty_output)
                if dirty_output is not None
                else None
            ),
        },
        "command": list(sys.argv),
    }
