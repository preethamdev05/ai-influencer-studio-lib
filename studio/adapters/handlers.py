"""
Telegram bot command and callback handlers.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, ConversationHandler

from ..schema.params import GenerationParams
from ..schema.errors import GenerationError
from .telegram_bot import get_engine

logger = logging.getLogger(__name__)

__all__ = [
    "start_command",
    "help_command",
    "status_command",
    "cancel_command",
    "model_selection",
    "style_selection",
    "resolution_selection",
    "prompt_received",
    "settings_callback",
    "regenerate_callback",
    "button_callback",
    "SELECTING_MODEL",
    "SELECTING_STYLE",
    "SELECTING_RESOLUTION",
    "ENTERING_PROMPT",
]

# Conversation states
SELECTING_MODEL = 1
SELECTING_STYLE = 2
SELECTING_RESOLUTION = 3
ENTERING_PROMPT = 4

# Resolution presets
RESOLUTIONS = {
    "portrait": (832, 1216),
    "landscape": (1216, 832),
    "square": (1024, 1024),
}

# Model display names
MODEL_NAMES = {
    "juggernaut": "Juggernaut XL",
    "lustify": "Lustify",
    "intorealism": "IntoRealism",
}


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Start command - initialize generation flow.
    """
    keyboard = [
        [
            InlineKeyboardButton("Juggernaut XL", callback_data="model_juggernaut"),
            InlineKeyboardButton("Lustify", callback_data="model_lustify"),
        ],
        [
            InlineKeyboardButton("IntoRealism", callback_data="model_intorealism"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "Welcome to AI Influencer Studio! 🎨\n\n"
        "Select a base model to start:",
        reply_markup=reply_markup,
    )
    
    # Initialize user data
    context.user_data["params"] = GenerationParams()
    
    return SELECTING_MODEL


async def model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Handle model selection callback.
    """
    query = update.callback_query
    await query.answer()
    
    model_name = query.data.replace("model_", "")
    params = context.user_data["params"]
    context.user_data["params"] = params.replace(base_model=model_name)
    
    keyboard = [
        [
            InlineKeyboardButton("Realistic", callback_data="style_realistic"),
            InlineKeyboardButton("Cinematic", callback_data="style_cinematic"),
        ],
        [
            InlineKeyboardButton("Artistic", callback_data="style_artistic"),
            InlineKeyboardButton("Default", callback_data="style_default"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        f"Model: {MODEL_NAMES[model_name]} ✓\n\n"
        "Select a style:",
        reply_markup=reply_markup,
    )
    
    return SELECTING_STYLE


async def style_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Handle style selection callback.
    """
    query = update.callback_query
    await query.answer()
    
    style = query.data.replace("style_", "")
    context.user_data["style"] = style
    
    keyboard = [
        [
            InlineKeyboardButton("Portrait (832x1216)", callback_data="res_portrait"),
        ],
        [
            InlineKeyboardButton("Landscape (1216x832)", callback_data="res_landscape"),
        ],
        [
            InlineKeyboardButton("Square (1024x1024)", callback_data="res_square"),
        ],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        f"Style: {style.title()} ✓\n\n"
        "Select resolution:",
        reply_markup=reply_markup,
    )
    
    return SELECTING_RESOLUTION


async def resolution_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Handle resolution selection callback.
    """
    query = update.callback_query
    await query.answer()
    
    res_key = query.data.replace("res_", "")
    width, height = RESOLUTIONS[res_key]
    
    params = context.user_data["params"]
    context.user_data["params"] = params.replace(width=width, height=height)
    
    await query.edit_message_text(
        f"Resolution: {width}x{height} ✓\n\n"
        "Now send your prompt (describe what you want to generate):"
    )
    
    return ENTERING_PROMPT


async def prompt_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Handle prompt text and start generation.
    """
    prompt_text = update.message.text
    params = context.user_data["params"]
    params = params.replace(prompt=prompt_text)
    
    # Apply style modifications
    style = context.user_data.get("style", "default")
    if style == "realistic":
        params = params.replace(loras=["detail", "identity"])
    elif style == "cinematic":
        params = params.replace(cfg=8.0, loras=["detail"])
    elif style == "artistic":
        params = params.replace(cfg=9.0, use_refiner=True)
    
    # Store final params
    context.user_data["params"] = params
    
    # Send initial status
    status_msg = await update.message.reply_text(
        "⏳ Adding to queue..."
    )
    
    try:
        # Get engine and generate
        engine = get_engine()
        
        # Ensure models loaded
        if not engine.state.base_models:
            await status_msg.edit_text("🔄 Loading models (first time)...")
            await engine.load_models()
        
        # Progress callback
        async def progress_callback(message: str):
            await status_msg.edit_text(f"⏳ {message}")
        
        # Generate image
        await status_msg.edit_text("🎨 Generating...")
        result = await engine.generate(params, progress_callback=progress_callback)
        
        # Send result
        await update.message.reply_photo(
            photo=result.image,
            caption=(
                f"✨ Generated in {result.total_time:.1f}s\n"
                f"Model: {MODEL_NAMES[params.base_model]}\n"
                f"Resolution: {params.width}x{params.height}\n"
                f"Steps: {params.steps} | CFG: {params.cfg}"
            ),
        )
        
        await status_msg.delete()
        
    except GenerationError as e:
        await status_msg.edit_text(f"❌ Generation failed: {e}")
        logger.error(f"Generation error: {e}")
    except Exception as e:
        await status_msg.edit_text(f"❌ Unexpected error: {e}")
        logger.exception("Unexpected error during generation")
    
    return ConversationHandler.END


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Help command - show available commands.
    """
    help_text = (
        "🤖 AI Influencer Studio Commands:\n\n"
        "/generate - Start new generation\n"
        "/status - Check bot status\n"
        "/cancel - Cancel current operation\n"
        "/help - Show this message\n\n"
        "Features:\n"
        "• Multiple SDXL models\n"
        "• Style presets\n"
        "• Resolution options\n"
        "• LoRA support\n"
        "• Dual-GPU refinement\n"
    )
    await update.message.reply_text(help_text)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Status command - show engine status.
    """
    engine = get_engine()
    status = engine.get_status()
    
    status_text = (
        f"🔧 Engine Status:\n\n"
        f"Models loaded: {status['models_loaded']}\n"
        f"Queue size: {status['queue_size']}\n"
        f"Active job: {'Yes' if status['current_job'] else 'No'}\n\n"
        f"GPU 0 VRAM: {status['vram_stage1'].get('allocated_gb', 0):.1f}GB / "
        f"{status['vram_stage1'].get('total_gb', 0):.1f}GB\n"
        f"GPU 1 VRAM: {status['vram_stage2'].get('allocated_gb', 0):.1f}GB / "
        f"{status['vram_stage2'].get('total_gb', 0):.1f}GB"
    )
    
    await update.message.reply_text(status_text)


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Cancel command - abort current conversation.
    """
    await update.message.reply_text("Operation cancelled.")
    return ConversationHandler.END


async def settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle settings adjustment callbacks.
    """
    query = update.callback_query
    await query.answer()
    
    # Parse setting change
    data = query.data.replace("settings_", "")
    # Implementation for advanced settings UI
    await query.edit_message_text("Settings panel (to be implemented)")


async def regenerate_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Handle image regeneration callbacks.
    """
    query = update.callback_query
    await query.answer()
    
    # Regenerate with stored params
    params = context.user_data.get("params")
    if not params:
        await query.edit_message_text("No previous generation found")
        return
    
    await query.edit_message_text("🎨 Regenerating...")
    
    try:
        engine = get_engine()
        result = await engine.generate(params)
        
        await query.message.reply_photo(
            photo=result.image,
            caption=f"✨ Regenerated in {result.total_time:.1f}s"
        )
    except Exception as e:
        await query.edit_message_text(f"❌ Regeneration failed: {e}")


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Generic button callback handler.
    """
    query = update.callback_query
    await query.answer()
    logger.info(f"Button callback: {query.data}")
