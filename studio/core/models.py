"""
Model loading and management for SDXL pipelines.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from safetensors.torch import load_file as load_safetensors

from ..schema.errors import ModelLoadError
from .download import download_file_civitai, download_file_huggingface
from .memory import cleanup_memory
from .logging_utils import stealth_print, log_metrics_async

__all__ = [
    'MODEL_URLS',
    'ensure_scheduler',
    'load_base_models',
    'get_stage1_pipeline',
    'load_refiner_pipeline',
    'ensure_loras_loaded',
]

MODEL_URLS = {
    "juggernaut":    "https://civitai.com/api/download/models/1759168",
    "lustify":       "https://civitai.com/api/download/models/2155386",
    "intorealism":   "https://civitai.com/api/download/models/2650268",
    "refiner":       "https://huggingface.co/SG161222/RealVisXL_V5.0/resolve/main/RealVisXL_V5.0_fp16.safetensors",
    "lora_detail":   "https://civitai.com/api/download/models/2524277",
    "lora_identity": "https://civitai.com/api/download/models/1981288",
}


def ensure_scheduler(pipe: Any) -> None:
    """
    Ensure pipeline uses DPMSolverMultistepScheduler unless v_prediction.
    """
    try:
        if isinstance(pipe.scheduler, DPMSolverMultistepScheduler):
            return

        prediction_type = getattr(pipe.scheduler.config, "prediction_type", "epsilon")
        current_scheduler = type(pipe.scheduler).__name__

        if prediction_type == "v_prediction":
            stealth_print(f"Preserving scheduler {current_scheduler} (v_prediction)", "progress")
            return

        stealth_print(f"Upgrading scheduler from {current_scheduler}", "progress")
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++"
        )
    except Exception as e:
        stealth_print(f"Scheduler setup warning: {e}", "warning")


def load_base_models(models_dir: Path) -> dict[str, Path]:
    """
    Download and cache base model checkpoints.
    
    Returns:
        Dict mapping model names to local paths
    """
    base_models = {}
    
    for name in ["juggernaut", "lustify", "intorealism"]:
        url = MODEL_URLS[name]
        local_path = models_dir / f"{name}.safetensors"
        
        if not local_path.exists():
            stealth_print(f"Downloading {name}...", "progress")
            download_file_civitai(url, local_path)
        
        base_models[name] = local_path
        stealth_print(f"Model {name} ready at {local_path}", "success")
    
    return base_models


def load_refiner_pipeline(models_dir: Path) -> Any:
    """
    Load refiner pipeline on cuda:1.
    """
    refiner_path = models_dir / "refiner.safetensors"
    
    if not refiner_path.exists():
        stealth_print("Downloading refiner model...", "progress")
        download_file_huggingface(MODEL_URLS["refiner"], refiner_path)
    
    stealth_print("Loading refiner pipeline...", "progress")
    
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(refiner_path),
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.to("cuda:1")
    pipe.enable_xformers_memory_efficient_attention()
    pipe.enable_vae_tiling()
    
    ensure_scheduler(pipe)
    
    stealth_print("Refiner loaded on cuda:1", "success")
    return pipe


def ensure_loras_loaded(
    pipe: Any,
    lora_names: list[str],
    vram_info: dict,
    models_dir: Path,
) -> list[str]:
    """
    Load requested LoRAs if VRAM permits.
    
    Returns:
        List of actually loaded LoRA names
    """
    # Check VRAM headroom
    vram_reserve = 3.5  # GB base reserve
    vram_available = vram_info.get("cuda:0", {}).get("free_gb", 0)
    
    if vram_available < vram_reserve:
        stealth_print(f"Insufficient VRAM for LoRAs ({vram_available:.1f}GB < {vram_reserve}GB)", "warning")
        return []
    
    loaded = []
    lora_map = {
        "detail": ("lora_detail", 0.6),
        "identity": ("lora_identity", 0.8),
    }
    
    for lora_name in lora_names:
        if lora_name not in lora_map:
            stealth_print(f"Unknown LoRA: {lora_name}", "warning")
            continue
        
        url_key, weight = lora_map[lora_name]
        url = MODEL_URLS[url_key]
        local_path = models_dir / f"{url_key}.safetensors"
        
        if not local_path.exists():
            stealth_print(f"Downloading {lora_name} LoRA...", "progress")
            download_file_civitai(url, local_path)
        
        try:
            pipe.load_lora_weights(str(local_path), adapter_name=lora_name)
            pipe.set_adapters([lora_name], adapter_weights=[weight])
            loaded.append(lora_name)
            stealth_print(f"LoRA {lora_name} loaded (weight={weight})", "success")
        except Exception as e:
            stealth_print(f"Failed to load LoRA {lora_name}: {e}", "error")
    
    return loaded
