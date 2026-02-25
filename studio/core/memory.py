"""
VRAM management and memory optimization utilities.

Restored original production formula + preventative checks.
"""
from __future__ import annotations

import gc

import torch

from .logging_utils import stealth_print
from ..schema.errors import VRAMError

__all__ = [
    "cleanup_memory",
    "get_vram_info",
    "required_vram_reserve",
    "preventative_memory_check",
]


def cleanup_memory(aggressive: bool = False) -> None:
    """Force garbage collection and CUDA cache clearing."""
    try:
        gc.collect()
        if torch.cuda.is_available():
            if aggressive:
                torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


def get_vram_info() -> dict[str, dict[str, float]]:
    vram_info = {}

    if not torch.cuda.is_available():
        return vram_info

    for i in range(torch.cuda.device_count()):
        device_name = f"cuda:{i}"
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = props.total_memory / 1024**3
        available = total - reserved

        vram_info[device_name] = {
            "total_gb": round(total, 2),
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "free_gb": round(available, 2),
        }

    return vram_info


def required_vram_reserve(
    width: int,
    height: int,
    steps1: int,
    steps2: int,
    cfg1: float,
    cfg2: float,
    userefiner: bool = False,
    init_noise_sigma: float = 1.0,
    is_refiner_pass: bool = False,
) -> float:
    """
    Production reserve formula (ported verbatim from original script).

    Note:
    - cfg* and init_noise_sigma are accepted for signature parity; formula uses width/height/steps.
    """
    area_mp = (width * height) / 1_000_000

    if is_refiner_pass:
        reserve = 2.8 + (area_mp * 0.7) + (steps2 * 0.004)
    else:
        reserve = 3.2 + (area_mp * 0.85) + (steps1 * 0.005)

    if userefiner:
        reserve += 0.8

    return min(reserve, 11.5)


def preventative_memory_check(device_id: int, required_gb: float) -> None:
    """
    Production preventative VRAM guard (ported verbatim from original script).

    Uses total - reserved as available headroom.
    """
    if not torch.cuda.is_available():
        return

    props = torch.cuda.get_device_properties(device_id)
    total_gb = props.total_memory / 1024**3
    reserved_gb = torch.cuda.memory_reserved(device_id) / 1024**3
    available_gb = total_gb - reserved_gb

    stealth_print(
        f"GPU{device_id} check need {required_gb:.1f}GB, have {available_gb:.1f}GB free",
        "progress",
    )

    if available_gb >= required_gb:
        return

    stealth_print(f"Preventative cleanup GPU{device_id}", "progress")
    cleanup_memory(aggressive=True)

    reserved_gb = torch.cuda.memory_reserved(device_id) / 1024**3
    new_available_gb = total_gb - reserved_gb

    if new_available_gb < required_gb:
        raise VRAMError(
            f"Insufficient VRAM on GPU{device_id} (need {required_gb:.1f}GB, have {new_available_gb:.1f}GB)"
        )
