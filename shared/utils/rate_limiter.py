"""
H4wkQuant - Binance Rate Limiter
Weight-based rate limiting with 429/418 handling
"""
import asyncio
import time
from typing import Dict, Optional, Tuple
import aiohttp
from loguru import logger


class BinanceRateLimiter:
    def __init__(self, weight_limit: int = 6000):
        self.weight_limit = weight_limit
        self.used_weight = 0
        self.last_weight_update = 0
        self._lock = asyncio.Lock()
        self._throttle_threshold = int(weight_limit * 0.8)
        self._banned_until = 0

    def _update_weight_from_headers(self, headers: Dict) -> None:
        weight_str = headers.get("X-MBX-USED-WEIGHT-1M", "")
        if weight_str:
            try:
                self.used_weight = int(weight_str)
                self.last_weight_update = time.time()
            except (ValueError, TypeError):
                pass

    async def _wait_if_throttled(self) -> None:
        now = time.time()

        if now < self._banned_until:
            wait_time = self._banned_until - now
            logger.critical(f"IP ban active, waiting {wait_time:.0f}s...")
            await asyncio.sleep(wait_time)
            self.used_weight = 0
            return

        if self.used_weight > self._throttle_threshold:
            logger.warning(f"Rate limit approaching: {self.used_weight}/{self.weight_limit}, throttling 2s")
            await asyncio.sleep(2)

        if now - self.last_weight_update > 60:
            self.used_weight = 0

    async def execute(
        self,
        session: aiohttp.ClientSession,
        method: str,
        url: str,
        **kwargs
    ) -> Tuple[int, Dict, Optional[Dict]]:
        async with self._lock:
            await self._wait_if_throttled()

        max_retries = 3
        last_error = None

        for attempt in range(max_retries):
            try:
                async with session.request(method, url, **kwargs) as response:
                    headers = dict(response.headers)
                    self._update_weight_from_headers(headers)
                    status = response.status

                    if status == 418:
                        self._banned_until = time.time() + 300
                        logger.critical("Binance IP ban (418)! Waiting 300s.")
                        await asyncio.sleep(300)
                        self.used_weight = 0
                        continue

                    if status == 429:
                        retry_after = int(headers.get("Retry-After", "60"))
                        logger.warning(f"Rate limit exceeded (429), waiting {retry_after}s...")
                        await asyncio.sleep(retry_after)
                        self.used_weight = 0
                        continue

                    data = await response.json()

                    if status != 200:
                        raise Exception(f"API Error {status}: {data.get('msg', 'Unknown error')}")

                    return status, data, headers

            except aiohttp.ClientError as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = 1.0 * (2 ** attempt)
                    logger.warning(f"HTTP error (attempt {attempt+1}/{max_retries}): {e}, waiting {delay}s")
                    await asyncio.sleep(delay)
                continue

        raise last_error or Exception("Rate-limited request failed after retries")


_rate_limiter: Optional[BinanceRateLimiter] = None


def get_rate_limiter() -> BinanceRateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = BinanceRateLimiter()
    return _rate_limiter
