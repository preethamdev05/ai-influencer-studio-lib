#!/usr/bin/env python3
"""
Telegram Bot Entrypoint

Starts the Telegram bot with worker queue.
"""
import os
import sys
import asyncio
import logging

from studio.adapters.telegram_bot import create_bot_application
from studio.adapters.worker import start_worker_loop

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


async def main():
    """Main entrypoint for bot execution."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN environment variable not set")
        sys.exit(1)

    logger.info("Initializing AI Influencer Studio Bot...")

    application = create_bot_application(token)

    worker_task = asyncio.create_task(start_worker_loop(application))

    try:
        await application.initialize()
        await application.start()
        logger.info("Bot started successfully")
        await application.updater.start_polling()

        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down bot...")
    finally:
        worker_task.cancel()
        await application.stop()
        await application.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
