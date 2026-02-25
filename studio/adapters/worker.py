"""
Background worker for async queue processing.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from ..core.engine import StudioEngine
from ..schema.params import GenerationParams

logger = logging.getLogger(__name__)

__all__ = ["start_worker_loop"]

# Global worker state
worker_engine: Optional[StudioEngine] = None
worker_running = False


async def start_worker_loop() -> None:
    """
    Start background worker loop for processing generation queue.
    
    This runs continuously and processes jobs from the engine's queue.
    """
    global worker_engine, worker_running
    
    if worker_running:
        logger.warning("Worker loop already running")
        return
    
    worker_running = True
    worker_engine = StudioEngine()
    
    logger.info("Worker loop started")
    
    try:
        # Load models on startup
        await worker_engine.load_models()
        logger.info("Worker models loaded")
        
        while worker_running:
            try:
                # Check for jobs in queue with timeout
                try:
                    job = await asyncio.wait_for(
                        worker_engine.queue.get(),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # Process job
                logger.info(f"Processing job: {job.get('id')}")
                worker_engine.current_job = job
                
                params = job["params"]
                callback = job.get("callback")
                
                try:
                    result = await worker_engine.generate(
                        params,
                        progress_callback=callback
                    )
                    
                    # Notify completion
                    if callback:
                        await callback(f"✅ Complete ({result.total_time:.1f}s)")
                    
                    logger.info(f"Job {job.get('id')} completed")
                    
                except Exception as e:
                    logger.error(f"Job {job.get('id')} failed: {e}")
                    if callback:
                        await callback(f"❌ Failed: {e}")
                
                finally:
                    worker_engine.current_job = None
                    worker_engine.queue.task_done()
            
            except Exception as e:
                logger.exception(f"Worker loop error: {e}")
                await asyncio.sleep(1)
    
    except asyncio.CancelledError:
        logger.info("Worker loop cancelled")
    finally:
        worker_running = False
        logger.info("Worker loop stopped")
