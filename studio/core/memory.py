"""
VRAM management and memory optimization utilities.
"""
from __future__ import annotations

import gc
from typing import Any

import torch

from .logging_utils import stealth_print

__all__ = [
    "cleanup_memory",
    "get_vram_info",
]


def cleanup_memory() -> None:
    """
    Force garbage collection and CUDA cache clearing.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    stealth_print("Memory cleanup completed", "progress")


def get_vram_info() -> dict[str, dict[str, float]]:
    """
    Get VRAM usage for all available CUDA devices.
    
    Returns:
        Dict mapping device name to memory stats in GB
    """
    vram_info = {}
    
    if not torch.cuda.is_available():
        return vram_info
    
    for i in range(torch.cuda.device_count()):
        device_name = f"cuda:{i}"
        props = torch.cuda.get_device_properties(i)
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        total = props.total_memory / 1024**3
        free = total - allocated
        
        vram_info[device_name] = {
            "total_gb": round(total, 2),
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "free_gb": round(free, 2),
        }
    
    return vram_info
