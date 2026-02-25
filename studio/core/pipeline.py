"""
SDXL pipeline construction and management.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from diffusers import StableDiffusionXLPipeline

from ..schema.errors import ModelLoadError
from .models import ensure_scheduler
from .logging_utils import stealth_print

__all__ = [
    "get_stage1_pipeline",
    "load_refiner_pipeline",
]


def get_stage1_pipeline(
    model_name: str,
    base_models: dict[str, Path],
    models_dir: Path,
) -> Any:
    """
    Load or retrieve cached stage 1 pipeline.
    
    Args:
        model_name: Name of base model (juggernaut, lustify, intorealism)
        base_models: Dict of model name to local path
        models_dir: Models directory path
        
    Returns:
        Configured SDXL pipeline on cuda:0
    """
    if model_name not in base_models:
        raise ModelLoadError(f"Unknown model: {model_name}")
    
    model_path = base_models[model_name]
    
    stealth_print(f"Loading pipeline for {model_name}...", "progress")
    
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(model_path),
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.to("cuda:0")
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_tiling()
    
    ensure_scheduler(pipe)
    
    stealth_print(f"Pipeline {model_name} ready on cuda:0", "success")
    return pipe
