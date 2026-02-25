"""
External adapters for Telegram, REST API, etc.
"""
from .telegram_bot import create_bot_application
from .worker import start_worker_loop

__all__ = [
    "create_bot_application",
    "start_worker_loop",
]
