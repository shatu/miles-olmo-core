"""When the trainer process runs Python's cyclic garbage collector.

A full collection holds the GIL for as long as it walks the live heap. After a checkpoint save and HF export the
heap holds many more objects, and on A4B the first weight publication after such a save froze the trainer's rank-0
process for ~17 s in the middle of a bucket (steps 119 and 139 of 01M86732NA). Bucket requests to the engines could
not be sent or read during the freeze, and the engines' connections then failed with httpx.ReadError, which fails
the publication. So the collector is paused while a publication runs, a full collection runs right after each save
(no engine request is in flight then), and any collection slower than a second is logged so a long one shows up.
"""

import contextlib
import functools
import gc
import time

from miles.utils import logger_utils

logger = logger_utils.setup_logger(__name__)

SLOW_COLLECTION_S = 1.0
_started: dict[int, float] = {}


@contextlib.contextmanager
def automatic_collection_paused():
    """Pause automatic collections for the block; an explicit gc.collect() still runs. Restores the prior state."""
    was_enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if was_enabled:
            gc.enable()


def paused_during(function):
    """Decorator form of automatic_collection_paused."""

    @functools.wraps(function)
    def wrapped(*args, **kwargs):
        with automatic_collection_paused():
            return function(*args, **kwargs)

    return wrapped


def collect(reason: str) -> float:
    """Run a full collection now and log how long it took; returns the seconds."""
    started = time.perf_counter()
    found = gc.collect()
    seconds = time.perf_counter() - started
    logger.info("gc.collect after %s: %.2f s, %d unreachable objects", reason, seconds, found)
    return seconds


def _on_collection(phase: str, info: dict) -> None:
    generation = info.get("generation", -1)
    if phase == "start":
        _started[generation] = time.perf_counter()
        return
    started = _started.pop(generation, None)
    if started is None:
        return
    seconds = time.perf_counter() - started
    if seconds >= SLOW_COLLECTION_S:
        logger.warning(
            "Slow garbage collection: generation %s took %.2f s (collected %s, uncollectable %s)",
            generation,
            seconds,
            info.get("collected"),
            info.get("uncollectable"),
        )


def log_slow_collections() -> None:
    """Log every collection that holds the GIL for SLOW_COLLECTION_S or longer. Idempotent."""
    if _on_collection not in gc.callbacks:
        gc.callbacks.append(_on_collection)
