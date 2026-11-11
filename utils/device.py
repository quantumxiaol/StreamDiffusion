"""Helpers for picking a device, dtype and acceleration backend.

StreamDiffusion runs on CUDA, Apple Metal (MPS) and CPU.  The demos used to
hard-code one of them, which meant that a CUDA machine could silently fall
back to CPU (or an Apple machine tried to import TensorRT).  Keep the policy
in one place instead.

Environment variables (all optional):

* ``DEVICE``       - ``auto`` (default), ``cuda``, ``cuda:1``, ``mps``, ``cpu``
* ``DTYPE``        - ``auto`` (default), ``float16``, ``bfloat16``, ``float32``
* ``ACCELERATION`` - ``auto`` (default), ``none``, ``xformers``, ``sfast``,
  ``tensorrt``
"""

from __future__ import annotations

import importlib.util
import os
import warnings
from typing import Optional, Union

import torch


DeviceLike = Union[str, torch.device, None]
DTypeLike = Union[str, torch.dtype, None]

_AUTO_NAMES = {"", "auto", "default", None}
_DTYPE_ALIASES = {
    "auto": None,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "float": torch.float32,
}


def mps_available() -> bool:
    """Return True when PyTorch was built with a working MPS backend."""

    mps = getattr(torch.backends, "mps", None)
    if mps is None:
        return False
    try:
        return bool(mps.is_available())
    except (RuntimeError, AttributeError):  # very old / broken builds
        return False


def available_devices() -> list[str]:
    """List the backends that can actually be used on this machine."""

    devices = []
    if torch.cuda.is_available():
        devices.extend(f"cuda:{index}" for index in range(torch.cuda.device_count()))
    if mps_available():
        devices.append("mps")
    devices.append("cpu")
    return devices


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def _validate_device(device: torch.device) -> Optional[str]:
    """Return a reason why *device* cannot be used, or None if it is usable."""

    if device.type == "cuda":
        if not torch.cuda.is_available():
            return "CUDA is not available (no NVIDIA driver/GPU or CPU-only PyTorch build)"
        if device.index is not None and device.index >= torch.cuda.device_count():
            return (
                f"CUDA device index {device.index} is out of range "
                f"(found {torch.cuda.device_count()} device(s))"
            )
    elif device.type == "mps":
        if not mps_available():
            return "MPS is not available (Metal backend is missing or disabled)"
    elif device.type != "cpu":
        return f"unsupported device type: {device.type!r}"
    return None


def get_device(preferred: DeviceLike = None) -> torch.device:
    """Resolve *preferred* to a usable :class:`torch.device`.

    ``None``/``"auto"`` picks CUDA, then MPS, then CPU.  An explicitly
    requested backend that is unavailable falls back to auto-detection with a
    warning instead of crashing (or silently using CPU).
    """

    if preferred is None:
        preferred = os.environ.get("DEVICE", "auto")

    if isinstance(preferred, torch.device):
        device = preferred
    else:
        value = str(preferred).strip()
        if value.lower() in _AUTO_NAMES:
            return _auto_device()
        try:
            device = torch.device(value)
        except (RuntimeError, ValueError):
            warnings.warn(
                f"DEVICE={value!r} is not a valid torch device; using auto-detection",
                stacklevel=2,
            )
            return _auto_device()

    problem = _validate_device(device)
    if problem is not None:
        warnings.warn(
            f"Requested device {device!s} is not usable: {problem}. "
            "Falling back to auto-detection.",
            stacklevel=2,
        )
        return _auto_device()
    return device


def get_dtype(device: DeviceLike, preferred: DTypeLike = None) -> torch.dtype:
    """Resolve the inference dtype for *device*.

    ``auto`` uses float16 on CUDA/MPS and float32 on CPU.  float16 is not
    supported by most CPU kernels, so an explicit float16 request on CPU is
    downgraded to float32 as well.
    """

    if not isinstance(device, torch.device):
        device = get_device(device)

    if preferred is None:
        preferred = os.environ.get("DTYPE", "auto")

    if isinstance(preferred, torch.dtype):
        dtype = preferred
    else:
        key = str(preferred).strip().lower()
        if key not in _DTYPE_ALIASES:
            warnings.warn(
                f"DTYPE={preferred!r} is not supported; using auto-detection",
                stacklevel=2,
            )
            key = "auto"
        dtype = _DTYPE_ALIASES[key]
        if dtype is None:
            dtype = torch.float16 if device.type in ("cuda", "mps") else torch.float32

    if dtype == torch.float16 and device.type == "cpu":
        warnings.warn(
            "float16 is not well supported on CPU; using float32 instead",
            stacklevel=2,
        )
        return torch.float32
    return dtype


def resolve_acceleration(requested: Optional[str], device: DeviceLike) -> str:
    """Return an acceleration backend that is valid for *device*.

    ``auto`` enables xformers on CUDA when it is installed and does nothing on
    MPS/CPU.  TensorRT, StableFast and xformers all require CUDA; asking for
    them on another backend falls back to ``"none"`` with a warning.
    """

    if not isinstance(device, torch.device):
        device = get_device(device)

    if requested is None:
        requested = os.environ.get("ACCELERATION", "auto")
    requested = str(requested).strip().lower() or "auto"

    if requested in ("auto", "default"):
        if device.type != "cuda":
            return "none"
        if importlib.util.find_spec("xformers") is not None:
            return "xformers"
        return "none"

    if requested in ("tensorrt", "xformers", "sfast") and device.type != "cuda":
        warnings.warn(
            f"acceleration={requested!r} requires CUDA, but device is {device!s}; "
            "using no acceleration",
            stacklevel=2,
        )
        return "none"

    return requested


def describe(device: DeviceLike, dtype: DTypeLike = None, acceleration: Optional[str] = None) -> str:
    """Small human readable summary used by demos at startup."""

    if not isinstance(device, torch.device):
        device = get_device(device)
    resolved_dtype = get_dtype(device, dtype)
    parts = [f"device={device!s}", f"dtype={str(resolved_dtype).replace('torch.', '')}"]
    if acceleration is not None:
        parts.append(f"acceleration={resolve_acceleration(acceleration, device)}")
    if device.type == "mps":
        parts.append("backend=Apple Metal")
    elif device.type == "cuda":
        parts.append(f"gpu={torch.cuda.get_device_name(device)}")
    return ", ".join(parts)
