"""
Telegram bot command and callback handlers.

Audit fix requirements:
- Handlers must NEVER call await engine.generate(params) directly.
- Enqueue all generation jobs to engine.generation_queue.
- Status must report queue size from engine.generation_queue.
- Engine instance must be retrieved from context.application.bot_data['engine'].
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes, ConversationHandler

from ..schema.params import GenerationParams

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

SELECTING_MODEL = 1
SELECTING_STYLE = 2
SELECTING_RESOLUTION = 3
ENTERING_PROMPT = 4

RESOLUTIONS = {
    "portrait": (832, 1216),
    "landscape": (1216, 832),
    "square": (1024, 1024),
}

MODEL_NAMES = {
    "juggernaut": "Juggernaut XL",
    "lustify": "Lustify",
    "intorealism": "IntoRealism",
}


def _get_engine(context: ContextTypes.DEFAULT_TYPE):
    engine = context.application.bot_data.get("engine")
    if engine is None:
        raise RuntimeError("StudioEngine not initialized in bot_data")
    return engine


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
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

    if update.message:
        await update.message.reply_text(
            "Welcome to AI Influencer Studio!\n\nSelect a base model to start:",
            reply_markup=reply_markup,
        )

    context.user_data["params"] = GenerationParams()
    return SELECTING_MODEL


async def model_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
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
        f"Model: {MODEL_NAMES.get(model_name, model_name)} ✓\n\nSelect a style:",
        reply_markup=reply_markup,
    )

    return SELECTING_STYLE


async def style_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    style = query.data.replace("style_", "")
    context.user_data["style"] = style

    keyboard = [
        [InlineKeyboardButton("Portrait (832x1216)", callback_data="res_portrait")],
        [InlineKeyboardButton("Landscape (1216x832)", callback_data="res_landscape")],
        [InlineKeyboardButton("Square (1024x1024)", callback_data="res_square")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await query.edit_message_text(
        f"Style: {style.title()} ✓\n\nSelect resolution:",
        reply_markup=reply_markup,
    )

    return SELECTING_RESOLUTION


async def resolution_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()

    res_key = query.data.replace("res_", "")
    width, height = RESOLUTIONS[res_key]

    params = context.user_data["params"]
    context.user_data["params"] = params.replace(width=width, height=height)

    await query.edit_message_text(
        f"Resolution: {width}x{height} ✓\n\nNow send your prompt:",
    )

    return ENTERING_PROMPT


async def prompt_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    engine = _get_engine(context)

    prompt_text = update.message.text
    params = context.user_data["params"].replace(prompt=prompt_text)

    style = context.user_data.get("style", "default")
    if style == "realistic":
        params = params.replace(loras=["detail", "identity"])
    elif style == "cinematic":
        params = params.replace(cfg=8.0, loras=["detail"])
    elif style == "artistic":
        params = params.replace(cfg=9.0, use_refiner=True)

    context.user_data["params"] = params

    queued_msg = await update.message.reply_text("⏳ Queued")

    job_payload = {
        "id": f"job_{update.effective_user.id}_{int(time.time() * 1000)}",
        "user_id": update.effective_user.id,
        "chat_id": update.effective_chat.id,
        "message_id": queued_msg.message_id,
        "params": params,
    }

    await engine.generation_queue.put(job_payload)

    return ConversationHandler.END


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "AI Influencer Studio Commands:\n\n"
        "/generate - Start new generation\n"
        "/status - Check bot status\n"
        "/cancel - Cancel current operation\n"
        "/help - Show this message\n"
    )
    await update.message.reply_text(help_text)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    engine = _get_engine(context)
    status = engine.get_status()

    queue_size = engine.generation_queue.qsize()

    status_text = (
        f"Engine Status:\n\n"
        f"Models loaded: {status['models_loaded']}\n"
        f"Queue size: {queue_size}\n"
        f"Active job: {'Yes' if status['current_job'] else 'No'}\n\n"
        f"GPU 0 VRAM: {status['vram_stage1'].get('allocated_gb', 0):.1f}GB / "
        f"{status['vram_stage1'].get('total_gb', 0):.1f}GB\n"
        f"GPU 1 VRAM: {status['vram_stage2'].get('allocated_gb', 0):.1f}GB / "
        f"{status['vram_stage2'].get('total_gb', 0):.1f}GB"
    )

    await update.message.reply_text(status_text)


async def cancel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Operation cancelled.")
    return ConversationHandler.END


async def settings_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Settings panel (to be implemented)")


async def regenerate_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    engine = _get_engine(context)
    params = context.user_data.get("params")
    if not params:
        await query.edit_message_text("No previous generation found")
        return

    await query.edit_message_text("⏳ Queued (regen)")

    job_payload = {
        "id": f"regen_{update.effective_user.id}_{int(time.time() * 1000)}",
        "user_id": update.effective_user.id,
        "chat_id": query.message.chat_id,
        "message_id": query.message.message_id,
        "params": params,
    }

    await engine.generation_queue.put(job_payload)


async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    logger.info(f"Button callback: {query.data}")
