# codes/main.py
import asyncio
import logging
import signal
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from cachetools import TTLCache

from API.matrix_api import MatrixClient
from graph.bot_graph import app
from states.bot_state import AgentState

load_dotenv()
logger = logging.getLogger("TigraMain")

# CONFIGURATION
MAX_WORKERS = 20
MAX_CONCURRENT_TASKS = 50
MAX_HISTORY_LENGTH = 10 # Prevent unbounded message growth
execution_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
shutdown_event = asyncio.Event()

# Idempotency cache (stores event_ids for 5 mins to prevent double-processing)
event_id_cache = TTLCache(maxsize=1000, ttl=300)

async def process_with_retry(input_data, config, retries=2):
    """Execution wrapper using dedicated executor, retries, and JITTER"""
    loop = asyncio.get_running_loop()
    for attempt in range(retries + 1):
        try:
            return await loop.run_in_executor(
                execution_executor,
                lambda: app.invoke(input_data, config)
            )
        except Exception as e:
            if attempt == retries: raise
            sleep_time = (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"Retry {attempt+1} in {sleep_time:.2f}s due to: {e}")
            await asyncio.sleep(sleep_time)

async def production_message_processor(event, matrix_client):
    """Asynchronous orchestrator with telemetry and idempotency"""
    # 1. IDEMPOTENCY CHECK
    event_id = event.get('event_id')
    if event_id in event_id_cache:
        logger.debug(f"Skipping duplicate event: {event_id}")
        return
    event_id_cache[event_id] = True

    user_input = event.get('body', '').strip()
    room_id = event.get('room_id')
    user_id = event.get('sender')
    thread_id = f"matrix_{room_id}_{user_id}"

    if not user_input or len(user_input) > 1500 or shutdown_event.is_set():
        return

    config = {"configurable": {"thread_id": thread_id}}

    # Unbounded message growth (Trimming)
    # We retrieve the state to ensure we don't exceed MAX_HISTORY_LENGTH
    existing_state = app.get_state(config)
    messages = existing_state.values.get("messages", []) if existing_state.values else []
    if len(messages) > MAX_HISTORY_LENGTH:
        # Keep only the most recent messages to save context/costs
        messages = messages[-MAX_HISTORY_LENGTH:]

    input_data = {
        "user_input": user_input,
        "messages": messages + [HumanMessage(content=user_input)]
    }

    try:
        # Semaphore placement (only lock during actual graph work)
        async with semaphore:
            # Telemetry
            start_time = time.perf_counter()
            final_state = await process_with_retry(input_data, config)
            duration = time.perf_counter() - start_time
            logger.info(f"Graph executed in {duration:.2f}s for user {user_id}")

        response_text = final_state.get("response", "System Busy")
        if final_state.get("requires_human_escalation"):
            response_text = f"📍 {response_text}\n\n*Notified human support.*"

        await asyncio.to_thread(matrix_client.send_message, room_id, response_text)

    except Exception as e:
        logger.error(f"Execution failure for {user_id}: {e}", exc_info=True)
        await asyncio.to_thread(matrix_client.send_message, room_id, "Temporary system error.")

async def shutdown(loop):
    """Graceful shutdown with timeout protection"""
    logger.info("Shutdown initiated...")
    shutdown_event.set()

    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    [t.cancel() for t in tasks]

    # Zombie Task Gathering (Timeout protection)
    logger.info(f"Cancelling {len(tasks)} active tasks...")
    try:
        await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=5.0)
    except asyncio.TimeoutError:
        logger.warning("Shutdown timed out; forcing exit.")

    execution_executor.shutdown(wait=False)
    loop.stop()

def setup_signals(loop):
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(loop)))

async def main():
    token = os.getenv("MATRIX_ACCESS_TOKEN")
    homeserver = os.getenv("MATRIX_HOMESERVER")
    client = MatrixClient(homeserver, token)

    loop = asyncio.get_running_loop()
    setup_signals(loop)

    def bridge(msg):
        if not shutdown_event.is_set():
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(production_message_processor(msg, client))
            )

    logger.info("Tigra Production Online (Workers: %d, History Limit: %d)", MAX_WORKERS, MAX_HISTORY_LENGTH)
    await loop.run_in_executor(None, lambda: client.listen_for_messages(process_message_callback=bridge))

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
