"""
Custom exception definitions.
"""
from __future__ import annotations

__all__ = [
    "StudioError",
    "ModelLoadError",
    "GenerationError",
    "VRAMError",
    "DownloadError",
]


class StudioError(Exception):
    """Base exception for studio library."""
    pass


class ModelLoadError(StudioError):
    """Failed to load model."""
    pass


class GenerationError(StudioError):
    """Image generation failed."""
    pass


class VRAMError(StudioError):
    """Insufficient VRAM."""
    pass


class DownloadError(StudioError):
    """Model download failed."""
    pass
