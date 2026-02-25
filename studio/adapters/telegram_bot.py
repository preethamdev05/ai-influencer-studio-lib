"""
Telegram bot application setup and configuration.
"""
from __future__ import annotations

import logging
from typing import Optional

from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ConversationHandler,
    filters,
)

from ..core.engine import StudioEngine
from .handlers import (
    start_command,
    help_command,
    status_command,
    cancel_command,
    model_selection,
    style_selection,
    resolution_selection,
    prompt_received,
    settings_callback,
    regenerate_callback,
    button_callback,
    SELECTING_MODEL,
    SELECTING_STYLE,
    SELECTING_RESOLUTION,
    ENTERING_PROMPT,
)

logger = logging.getLogger(__name__)

__all__ = ["create_bot_application"]

# Global engine instance (shared across handlers)
engine: Optional[StudioEngine] = None


def get_engine() -> StudioEngine:
    """Get or create global engine instance."""
    global engine
    if engine is None:
        engine = StudioEngine()
    return engine


def create_bot_application(token: str) -> Application:
    """
    Create configured Telegram bot application.
    
    Args:
        token: Telegram bot token
        
    Returns:
        Configured Application instance
    """
    # Build application
    application = Application.builder().token(token).build()
    
    # Conversation handler for generation flow
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("generate", start_command)],
        states={
            SELECTING_MODEL: [
                CallbackQueryHandler(model_selection, pattern="^model_")
            ],
            SELECTING_STYLE: [
                CallbackQueryHandler(style_selection, pattern="^style_")
            ],
            SELECTING_RESOLUTION: [
                CallbackQueryHandler(resolution_selection, pattern="^res_")
            ],
            ENTERING_PROMPT: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, prompt_received)
            ],
        },
        fallbacks=[CommandHandler("cancel", cancel_command)],
    )
    
    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(conv_handler)
    
    # Callback handlers for settings and regeneration
    application.add_handler(CallbackQueryHandler(settings_callback, pattern="^settings_"))
    application.add_handler(CallbackQueryHandler(regenerate_callback, pattern="^regen_"))
    application.add_handler(CallbackQueryHandler(button_callback))
    
    logger.info("Bot application configured")
    return application
