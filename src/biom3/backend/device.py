"""Device detection, selection, and helper utilities

"""

import logging
import os
import platform
import socket
import sys

import psutil
import torch

from biom3.core._dist_env import get_global_rank

_CPU = "cpu"
_CUDA = "cuda"
_XPU = "xpu"


def get_backend_name() -> str:
    if torch.cuda.is_available():
        return _CUDA
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return _XPU
    return _CPU


# Computed once at import time. The active backend never changes within a
# process, so every helper below references this constant directly instead
# of recomputing it on each call.
BACKEND_NAME = get_backend_name()


def get_device():
    return torch.device(BACKEND_NAME)


DEVICE_CHOICES = ("auto", _CPU, _CUDA, _XPU)


def resolve_device(requested, *, allow_cpu: bool = True) -> str:
    """Resolve a ``--device`` value to a concrete device type.

    ``"auto"`` (or ``None``) becomes the detected backend: CUDA, then XPU,
    then CPU. With ``allow_cpu=False`` an automatic fallback to CPU raises
    instead, so a training run never lands on CPU by accident; an explicit
    ``"cpu"`` is always honoured.
    """
    if requested not in (None, "auto"):
        return requested
    if BACKEND_NAME == _CPU and not allow_cpu:
        raise RuntimeError(
            "--device auto found no GPU backend (neither CUDA nor XPU is "
            "available to torch). Pass --device cpu to run on CPU deliberately."
        )
    return BACKEND_NAME


def _visible_device_count(device: str) -> int:
    if device == _CUDA:
        return torch.cuda.device_count()
    if device == _XPU and hasattr(torch, "xpu"):
        return torch.xpu.device_count()
    return 0


def check_devices_per_node(device: str, devices_per_node: int) -> None:
    """Fail early if a run asks for more devices per node than it can see.

    The count is never inferred: it is a layout choice (one rank per tile,
    one rank per node, ...). This only catches a request the node cannot
    satisfy, e.g. 12 on an Aurora node exposing 6 because
    ``ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE``. Skipped for XPU when
    ``ZE_AFFINITY_MASK`` pins each process to its own tile(s), since each
    rank then legitimately sees fewer devices than the node has.
    """
    if device == _CPU:
        return
    if device == _XPU and os.environ.get("ZE_AFFINITY_MASK"):
        return
    devices_per_node = int(devices_per_node or 1)
    visible = _visible_device_count(device)
    if devices_per_node <= visible:
        return
    if device == _XPU:
        hint = ("ZE_FLAT_DEVICE_HIERARCHY="
                f"{os.environ.get('ZE_FLAT_DEVICE_HIERARCHY', '<unset>')}; FLAT "
                "exposes each tile as a device, COMPOSITE each GPU.")
    else:
        hint = f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}."
    raise RuntimeError(
        f"Requested {devices_per_node} {device} device(s) per node, but this "
        f"process sees {visible}. {hint}"
    )


def reset_peak_memory_stats():
    """Reset peak memory counters for the active device backend.

    No-op on CPU.  CUDA and XPU expose the same-named function.
    """
    if BACKEND_NAME == _CUDA:
        torch.cuda.reset_peak_memory_stats()
    elif BACKEND_NAME == _XPU:
        torch.xpu.reset_peak_memory_stats()


def get_peak_memory_stats():
    """Return ``(peak_allocated_bytes, peak_reserved_bytes)`` for the local device.

    Returns ``(None, None)`` on CPU or if the backend does not expose the
    relevant APIs.  CUDA and XPU expose identical function names.
    """
    if BACKEND_NAME == _CUDA:
        return (torch.cuda.max_memory_allocated(),
                torch.cuda.max_memory_reserved())
    if BACKEND_NAME == _XPU:
        try:
            return (torch.xpu.max_memory_allocated(),
                    torch.xpu.max_memory_reserved())
        except AttributeError:
            return (None, None)
    return (None, None)


def device_sync():
    """Block the host until all queued work on the active device finishes.

    Intended for benchmark timing — wrap before ``perf_counter`` reads so
    asynchronous kernel launches are accounted for.  No-op on CPU.
    """
    if BACKEND_NAME == _CUDA:
        torch.cuda.synchronize()
    elif BACKEND_NAME == _XPU:
        torch.xpu.synchronize()


def get_device_name(index: int = 0) -> str:
    """Return a human-readable name for the active device.

    CUDA → ``torch.cuda.get_device_name``; XPU → ``torch.xpu.get_device_name``;
    CPU → ``platform.processor()`` (falls back to ``platform.machine()``).
    """
    if BACKEND_NAME == _CUDA:
        return torch.cuda.get_device_name(index)
    if BACKEND_NAME == _XPU:
        try:
            return torch.xpu.get_device_name(index)
        except AttributeError:
            return _XPU
    return platform.processor() or platform.machine() or _CPU


def get_device_info(index: int = 0) -> dict:
    """Return a dict describing the active compute device and host environment.

    Suitable for serialising alongside benchmark results so runs on different
    machines (Spark / Polaris / Aurora / CPU) remain distinguishable.
    """
    info = {
        "backend": BACKEND_NAME,
        "device_name": get_device_name(index),
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if BACKEND_NAME == _CUDA:
        info["device_count"] = torch.cuda.device_count()
        try:
            cap = torch.cuda.get_device_capability(index)
            info["cuda_capability"] = f"{cap[0]}.{cap[1]}"
        except Exception:
            pass
        info["cuda_version"] = torch.version.cuda
    elif BACKEND_NAME == _XPU:
        try:
            info["device_count"] = torch.xpu.device_count()
        except AttributeError:
            info["device_count"] = None
    else:
        info["device_count"] = 0
    return info

def setup_logger(name: str = "biom3", level: int = logging.INFO) -> logging.Logger:
    """Return a rank-aware logger that only emits on rank 0.

    Non-zero ranks are silenced (set to CRITICAL) so that duplicate
    messages are never printed in multi-node / multi-GPU settings.

    Call once per module::

        from biom3.backend.device import setup_logger
        logger = setup_logger(__name__)
        logger.info("only printed on rank 0")
    """
    logger = logging.getLogger(name)
    logger.propagate = False  # prevent duplicate output via root logger
    rank = get_global_rank()
    if rank == 0:
        logger.setLevel(level)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            ))
            logger.addHandler(handler)
    else:
        logger.setLevel(logging.CRITICAL)
    return logger


_logger = setup_logger(__name__)


def print_memory_usage() -> float:
    """Log and return this Python process's resident-set size in MB.

    Backend-agnostic: measures the host process, not the device. Useful
    on every machine including CPU. Logs via the rank-aware logger so
    only rank 0 prints.
    """
    mb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    _logger.info("CPU memory used by this script: %.2f MB", mb)
    return mb


_MATMUL_PRECISIONS = ("highest", "high", "medium")


def set_float32_matmul_precision(precision: str = "high") -> str:
    """Set the global fp32 matmul precision and return the applied value.

    ``"high"`` routes fp32 GEMMs through TF32 tensor cores (~1.5-1.8x faster
    on Ampere-and-later NVIDIA GPUs; a harmless no-op on hardware without TF32
    tensor cores). ``"highest"`` keeps full fp32 for bitwise reproducibility;
    ``"medium"`` uses the bf16 path. This is a process-global runtime setting,
    so it must be called before the first matmul.
    """
    if precision not in _MATMUL_PRECISIONS:
        raise ValueError(
            f"float32_matmul_precision must be one of {_MATMUL_PRECISIONS}, "
            f"got {precision!r}"
        )
    torch.set_float32_matmul_precision(precision)
    _logger.info("float32 matmul precision: %s", precision)
    return precision


# Pull in the active backend's symbols (DIST_BACKEND, resolve_/set_device_for_local_rank,
# print_gpu_initialization, print_gpu_utilization, etc.). BACKEND_NAME was
# computed near the top of this module.
if BACKEND_NAME == _CUDA:
    from .cuda import *
elif BACKEND_NAME == _XPU:
    from .xpu import *
elif BACKEND_NAME == _CPU:
    from .cpu import *
else:
    raise RuntimeError(f"Unexpected backend name: {BACKEND_NAME}")
