"""
Prompt building and enhancement logic.
"""
from __future__ import annotations

from typing import Dict

from ..schema.errors import GenerationError

__all__ = [
    "stage_prompt_builder",
    "validate_resolution",
]

# Quality enhancement tags
QUALITY_POSITIVE = (
    "masterpiece, best quality, highly detailed, sharp focus, "
    "professional photography, 8k uhd, realistic lighting"
)

QUALITY_NEGATIVE = (
    "low quality, blurry, pixelated, jpeg artifacts, distorted, "
    "low resolution, watermark, signature, text, cropped, "
    "out of frame, duplicate, mutation, deformed"
)


def validate_resolution(width: int, height: int) -> None:
    """
    Validate that resolution is within SDXL constraints.
    
    Raises:
        GenerationError: If resolution invalid
    """
    min_dim = 512
    max_dim = 2048
    
    if width < min_dim or width > max_dim:
        raise GenerationError(f"Width must be between {min_dim} and {max_dim}")
    
    if height < min_dim or height > max_dim:
        raise GenerationError(f"Height must be between {min_dim} and {max_dim}")
    
    # Check divisibility by 8 (SDXL requirement)
    if width % 8 != 0 or height % 8 != 0:
        raise GenerationError("Width and height must be divisible by 8")
    
    # Check total pixel count (prevent VRAM overflow)
    max_pixels = 2048 * 2048
    if width * height > max_pixels:
        raise GenerationError(f"Total pixels exceed maximum ({max_pixels})")


def stage_prompt_builder(
    user_prompt: str,
    user_negative: str,
    model_name: str,
) -> Dict[str, str]:
    """
    Build enhanced prompts with quality tags.
    
    Args:
        user_prompt: User's positive prompt
        user_negative: User's negative prompt
        model_name: Base model name for style-specific enhancement
        
    Returns:
        Dict with 'positive' and 'negative' keys
    """
    # Combine user prompt with quality tags
    positive = f"{user_prompt}, {QUALITY_POSITIVE}"
    negative = f"{user_negative}, {QUALITY_NEGATIVE}" if user_negative else QUALITY_NEGATIVE
    
    # Model-specific enhancements
    if model_name == "intorealism":
        positive += ", photorealistic, natural skin texture"
    elif model_name == "lustify":
        positive += ", vibrant colors, dramatic lighting"
    
    return {
        "positive": positive,
        "negative": negative,
    }
