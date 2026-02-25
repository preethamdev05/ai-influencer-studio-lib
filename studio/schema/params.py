"""
Generation parameter definitions.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

__all__ = [
    "GenerationParams",
    "BotConfig",
]


@dataclass(frozen=True)
class GenerationParams:
    """
    Image generation parameters.
    """
    base_model: str = "juggernaut"
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 832
    height: int = 1216
    steps: int = 25
    cfg: float = 7.0
    seed: int = -1
    use_refiner: bool = True
    loras: list[str] = field(default_factory=list)
    
    def replace(self, **changes):
        """Create new instance with modified fields."""
        return replace(self, **changes)


@dataclass
class BotConfig:
    """
    Telegram bot configuration.
    """
    token: str
    admin_ids: list[int] = field(default_factory=list)
    max_queue_size: int = 10
    generation_timeout: int = 300
