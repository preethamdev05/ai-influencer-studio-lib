"""
Logging and metrics utilities.
"""
from __future__ import annotations

import asyncio
from typing import Any, Literal

__all__ = [
    "stealth_print",
    "log_metrics_async",
]

# ANSI color codes
COLORS = {
    "progress": "\033[94m",  # Blue
    "success": "\033[92m",   # Green
    "warning": "\033[93m",   # Yellow
    "error": "\033[91m",     # Red
    "reset": "\033[0m",
}


def stealth_print(message: str, level: Literal["progress", "success", "warning", "error"] = "progress") -> None:
    """
    Print colored log message.
    
    Args:
        message: Log message
        level: Message severity level
    """
    color = COLORS.get(level, COLORS["progress"])
    reset = COLORS["reset"]
    print(f"{color}[{level.upper()}]{reset} {message}")


async def log_metrics_async(metrics: dict[str, Any]) -> None:
    """
    Asynchronously log generation metrics.
    
    Args:
        metrics: Dict of metric name to value
    """
    await asyncio.sleep(0)  # Yield control
    
    # Format metrics for display
    formatted = []
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted.append(f"{key}={value:.2f}")
        else:
            formatted.append(f"{key}={value}")
    
    stealth_print(f"Metrics: {', '.join(formatted)}", "success")
