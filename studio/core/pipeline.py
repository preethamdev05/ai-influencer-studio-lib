"""
SDXL pipeline construction and management.

Audit fix requirements:
- Restore explicit LoRA unload safety when switching models.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import gc
import torch
from diffusers import StableDiffusionXLPipeline

from ..schema.errors import ModelLoadError
from .models import ensure_scheduler
from .logging_utils import stealth_print

__all__ = [
    "get_stage1_pipeline",
]

_STAGE1_PIPE: Optional[Any] = None
_STAGE1_MODEL: Optional[str] = None


def _destroy_stage1_pipeline(pipe: Any) -> None:
    try:
        if hasattr(pipe, "unload_lora_weights"):
            pipe.unload_lora_weights()
    except Exception:
        pass
    try:
        if hasattr(pipe, "set_adapters"):
            pipe.set_adapters([], adapter_weights=[])
    except Exception:
        pass

    try:
        del pipe
    except Exception:
        pass

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_stage1_pipeline(
    model_name: str,
    base_models: dict[str, Path],
    models_dir: Path,
) -> Any:
    global _STAGE1_PIPE, _STAGE1_MODEL

    if model_name not in base_models:
        raise ModelLoadError(f"Unknown model: {model_name}")

    if _STAGE1_PIPE is not None and _STAGE1_MODEL == model_name:
        return _STAGE1_PIPE

    if _STAGE1_PIPE is not None and _STAGE1_MODEL != model_name:
        stealth_print(f"Destroying stage1 pipeline for {_STAGE1_MODEL}", "progress")
        _destroy_stage1_pipeline(_STAGE1_PIPE)
        _STAGE1_PIPE = None
        _STAGE1_MODEL = None

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

    _STAGE1_PIPE = pipe
    _STAGE1_MODEL = model_name

    return pipe
