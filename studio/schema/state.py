"""
Runtime state management.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from .params import GenerationParams

__all__ = [
    "EngineState",
    "GenerationResult",
]


@dataclass
class EngineState:
    """
    Engine runtime state.
    """
    base_models: dict[str, Path] = field(default_factory=dict)
    active_pipeline: Optional[Any] = None
    refiner_pipeline: Optional[Any] = None
    loaded_loras: list[str] = field(default_factory=list)


@dataclass
class GenerationResult:
    """
    Result of image generation.
    """
    image: Image.Image
    params: GenerationParams
    stage1_time: float
    stage2_time: float
    total_time: float
