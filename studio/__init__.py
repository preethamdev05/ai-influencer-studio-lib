"""
AI Influencer Studio - Professional SDXL Inference Library
"""
from .core.engine import StudioEngine
from .schema.params import GenerationParams, BotConfig
from .schema.errors import (
    StudioError,
    ModelLoadError,
    GenerationError,
    VRAMError,
    DownloadError,
)

__version__ = "1.0.0"

__all__ = [
    "StudioEngine",
    "GenerationParams",
    "BotConfig",
    "StudioError",
    "ModelLoadError",
    "GenerationError",
    "VRAMError",
    "DownloadError",
]
