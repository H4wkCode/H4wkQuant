"""
H4wkQuant - Async Retry Decorator
Exponential backoff + jitter
"""
import asyncio
import functools
import random
from typing import Tuple, Type, Optional, Callable
from loguru import logger


def async_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None,
):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_error = e
                    if attempt >= max_retries - 1:
                        break

                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    if jitter:
                        delay = delay * (0.5 + random.random())

                    logger.warning(
                        f"Retry {attempt + 1}/{max_retries} [{func.__qualname__}]: "
                        f"{type(e).__name__}: {e} -> {delay:.1f}s"
                    )

                    if on_retry:
                        try:
                            on_retry(attempt + 1, e, delay)
                        except Exception:
                            pass

                    await asyncio.sleep(delay)

            raise last_error

        return wrapper
    return decorator
