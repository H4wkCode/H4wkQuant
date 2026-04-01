"""
H4wkQuant - Bybit WebSocket Client
Connects to Bybit linear futures WebSocket for cross-exchange arb.
"""
import asyncio
import json
import time
from typing import Callable, Dict, List, Optional, Set

import aiohttp
from loguru import logger


class BybitWSClient:
    """Bybit Futures WebSocket client for real-time data."""

    WS_URL = "wss://stream.bybit.com/v5/public/linear"
    WS_TESTNET_URL = "wss://stream-testnet.bybit.com/v5/public/linear"

    def __init__(self, testnet: bool = False):
        self.ws_url = self.WS_TESTNET_URL if testnet else self.WS_URL
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._callbacks: Dict[str, List[Callable]] = {}
        self._running = False
        self._subscribed_topics: Set[str] = set()
        self._reconnect_count = 0

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def connect(self):
        """Connect to Bybit WebSocket."""
        session = await self._get_session()
        try:
            self._ws = await session.ws_connect(
                self.ws_url,
                heartbeat=20,
                timeout=aiohttp.ClientTimeout(total=30),
            )
            self._running = True
            logger.info(f"Bybit WS connected: {self.ws_url}")

            # Re-subscribe existing topics
            if self._subscribed_topics:
                await self._send_subscribe(list(self._subscribed_topics))

            asyncio.create_task(self._listen())
            asyncio.create_task(self._ping_loop())
        except Exception as e:
            logger.error(f"Bybit WS connect failed: {e}")
            await asyncio.sleep(5)
            await self.connect()

    async def _listen(self):
        """Listen for messages."""
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_message(msg.data)
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        except Exception as e:
            logger.error(f"Bybit WS listen error: {e}")

        if self._running:
            self._reconnect_count += 1
            logger.warning(f"Bybit WS disconnected, reconnecting ({self._reconnect_count})...")
            await asyncio.sleep(min(5 * self._reconnect_count, 30))
            await self.connect()

    async def _handle_message(self, raw: str):
        try:
            data = json.loads(raw)
            topic = data.get("topic", "")
            if not topic:
                return

            for cb in self._callbacks.get(topic, []):
                try:
                    await cb(data.get("data", {}))
                except Exception as e:
                    logger.error(f"Bybit callback error: {e}")
        except json.JSONDecodeError:
            pass

    async def _ping_loop(self):
        while self._running and self._ws and not self._ws.closed:
            try:
                await self._ws.send_json({"op": "ping"})
            except Exception:
                break
            await asyncio.sleep(20)

    async def _send_subscribe(self, topics: List[str]):
        if self._ws and not self._ws.closed:
            await self._ws.send_json({
                "op": "subscribe",
                "args": topics,
            })

    async def subscribe_trades(self, symbols: List[str], callback: Callable):
        """Subscribe to trade stream."""
        for symbol in symbols:
            topic = f"publicTrade.{symbol}"
            self._subscribed_topics.add(topic)
            self._callbacks.setdefault(topic, []).append(callback)

        if self._ws and not self._ws.closed:
            await self._send_subscribe([f"publicTrade.{s}" for s in symbols])

    async def subscribe_orderbook(self, symbols: List[str], callback: Callable, depth: int = 25):
        """Subscribe to orderbook."""
        for symbol in symbols:
            topic = f"orderbook.{depth}.{symbol}"
            self._subscribed_topics.add(topic)
            self._callbacks.setdefault(topic, []).append(callback)

        if self._ws and not self._ws.closed:
            await self._send_subscribe([f"orderbook.{depth}.{s}" for s in symbols])

    async def subscribe_tickers(self, symbols: List[str], callback: Callable):
        """Subscribe to ticker (mark price, funding rate)."""
        for symbol in symbols:
            topic = f"tickers.{symbol}"
            self._subscribed_topics.add(topic)
            self._callbacks.setdefault(topic, []).append(callback)

        if self._ws and not self._ws.closed:
            await self._send_subscribe([f"tickers.{s}" for s in symbols])

    async def close(self):
        self._running = False
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("Bybit WS closed")
