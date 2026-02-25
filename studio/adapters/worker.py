"""
Background worker for async queue processing.

Audit fix requirements:
- Worker must pull jobs from engine.generation_queue only.
- Worker must await engine.generate() internally.
- Worker must use the SAME StudioEngine instance injected into application.bot_data.
- Remove global engine variables.
"""
from __future__ import annotations

import asyncio
import logging
from io import BytesIO

from telegram.ext import Application

logger = logging.getLogger(__name__)

__all__ = ["start_worker_loop"]


async def start_worker_loop(application: Application) -> None:
    engine = application.bot_data.get("engine")
    if engine is None:
        raise RuntimeError("StudioEngine not initialized in application.bot_data")

    logger.info("Worker loop started")

    if not engine.state.base_models:
        await engine.load_models()
        logger.info("Worker models loaded")

    while True:
        job = await engine.generation_queue.get()
        engine.current_job = job

        chat_id = job.get("chat_id")
        message_id = job.get("message_id")
        params = job.get("params")

        async def progress_callback(message: str):
            try:
                await application.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=f"⏳ {message}",
                )
            except Exception:
                return

        try:
            await progress_callback("Generating...")
            result = await engine.generate(params, progress_callback=progress_callback)

            bio = BytesIO()
            result.image.save(bio, format="PNG")
            bio.seek(0)

            await application.bot.send_photo(
                chat_id=chat_id,
                photo=bio,
                caption=(
                    f"✨ Generated in {result.total_time:.1f}s\n"
                    f"Model: {params.base_model}\n"
                    f"Resolution: {params.width}x{params.height}\n"
                    f"Steps: {params.steps} | CFG: {params.cfg}"
                ),
            )

            try:
                await application.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text="✅ Complete",
                )
            except Exception:
                pass

        except Exception as e:
            logger.exception(f"Job failed: {e}")
            try:
                await application.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=f"❌ Failed: {e}",
                )
            except Exception:
                pass
        finally:
            engine.current_job = None
            engine.generation_queue.task_done()
