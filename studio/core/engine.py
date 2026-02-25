"""
Main Studio Engine - orchestrates generation pipeline.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable, Any

import torch
from PIL import Image

from ..schema.params import GenerationParams
from ..schema.errors import GenerationError, VRAMError
from ..schema.state import EngineState, GenerationResult
from .pipeline import get_stage1_pipeline, load_refiner_pipeline
from .models import load_base_models, ensure_loras_loaded
from .memory import cleanup_memory, get_vram_info
from .prompts import stage_prompt_builder, validate_resolution
from .logging_utils import stealth_print, log_metrics_async

__all__ = ["StudioEngine"]


class StudioEngine:
    """
    Main inference engine encapsulating model loading, pipeline management,
    and generation orchestration.
    """

    def __init__(self, models_dir: Optional[Path] = None):
        self.models_dir = models_dir or Path.home() / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.state = EngineState()
        self.queue: asyncio.Queue = asyncio.Queue()
        self.current_job: Optional[dict] = None
        self.metrics: dict[str, Any] = {}

    async def load_models(self) -> None:
        """Load base models and prepare pipelines."""
        stealth_print("Loading base models...", "progress")
        self.state.base_models = await asyncio.to_thread(
            load_base_models, self.models_dir
        )
        stealth_print(f"Loaded {len(self.state.base_models)} base models", "success")

    async def generate(
        self,
        params: GenerationParams,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> GenerationResult:
        """
        Generate image based on parameters.
        
        Args:
            params: Generation parameters
            progress_callback: Optional callback for progress updates
            
        Returns:
            GenerationResult with image and metadata
        """
        start_time = time.time()
        
        try:
            # Validate resolution
            validate_resolution(params.width, params.height)
            
            # Get stage 1 pipeline
            if progress_callback:
                progress_callback("Loading pipeline...")
            
            pipe_stage1 = await asyncio.to_thread(
                get_stage1_pipeline,
                params.base_model,
                self.state.base_models,
                self.models_dir,
            )
            
            # Build prompts
            prompts = stage_prompt_builder(
                params.prompt,
                params.negative_prompt,
                params.base_model,
            )
            
            # Check VRAM and LoRAs
            if params.loras:
                if progress_callback:
                    progress_callback("Loading LoRAs...")
                    
                vram_info = get_vram_info()
                final_loras = await asyncio.to_thread(
                    ensure_loras_loaded,
                    pipe_stage1,
                    params.loras,
                    vram_info,
                    self.models_dir,
                )
                params = params.replace(loras=final_loras)
            
            # Stage 1 generation
            if progress_callback:
                progress_callback(f"Generating ({params.steps} steps)...")
            
            gen_kwargs = {
                "prompt": prompts["positive"],
                "negative_prompt": prompts["negative"],
                "width": params.width,
                "height": params.height,
                "num_inference_steps": params.steps,
                "guidance_scale": params.cfg,
                "output_type": "latent" if params.use_refiner else "pil",
            }
            
            if params.seed != -1:
                gen_kwargs["generator"] = torch.Generator(device="cuda:0").manual_seed(params.seed)
            
            result_stage1 = await asyncio.to_thread(
                pipe_stage1, **gen_kwargs
            )
            
            stage1_time = time.time() - start_time
            
            # Stage 2 refinement
            final_image = None
            stage2_time = 0.0
            
            if params.use_refiner:
                if progress_callback:
                    progress_callback("Refining...")
                
                refiner_start = time.time()
                pipe_refiner = await asyncio.to_thread(
                    load_refiner_pipeline, self.models_dir
                )
                
                refiner_kwargs = {
                    "prompt": prompts["positive"],
                    "negative_prompt": prompts["negative"],
                    "image": result_stage1.images[0],
                    "num_inference_steps": 15,
                    "strength": 0.3,
                    "guidance_scale": params.cfg,
                }
                
                result_refiner = await asyncio.to_thread(
                    pipe_refiner, **refiner_kwargs
                )
                final_image = result_refiner.images[0]
                stage2_time = time.time() - refiner_start
            else:
                final_image = result_stage1.images[0]
            
            # Cleanup
            await asyncio.to_thread(cleanup_memory)
            
            total_time = time.time() - start_time
            
            # Log metrics
            await log_metrics_async({
                "stage1_time": stage1_time,
                "stage2_time": stage2_time,
                "total_time": total_time,
                "resolution": f"{params.width}x{params.height}",
                "steps": params.steps,
                "loras": params.loras,
            })
            
            return GenerationResult(
                image=final_image,
                params=params,
                stage1_time=stage1_time,
                stage2_time=stage2_time,
                total_time=total_time,
            )
            
        except Exception as e:
            stealth_print(f"Generation failed: {e}", "error")
            raise GenerationError(f"Generation failed: {e}") from e

    def get_status(self) -> dict[str, Any]:
        """Get current engine status."""
        vram_info = get_vram_info()
        return {
            "models_loaded": len(self.state.base_models),
            "queue_size": self.queue.qsize(),
            "current_job": self.current_job is not None,
            "vram_stage1": vram_info.get("cuda:0", {}),
            "vram_stage2": vram_info.get("cuda:1", {}),
        }
