"""
H4wkQuant - Binance WebSocket Client
Real-time orderbook, trades, markprice streams
"""
import asyncio
import json
import time
from typing import Callable, Dict, List, Optional, Set
import websockets
from loguru import logger

from shared.config.settings import settings


class BinanceWSClient:
    def __init__(self):
        self.base_url = settings.binance.ws_url
        self._connections: Dict[str, websockets.WebSocketClientProtocol] = {}
        self._callbacks: Dict[str, List[Callable]] = {}
        self._running = False
        self._reconnect_tasks: Dict[str, asyncio.Task] = {}

    async def subscribe_orderbook(self, symbols: List[str], callback: Callable, depth: int = 20):
        """Subscribe to orderbook depth streams"""
        streams = [f"{s.lower()}@depth{depth}@100ms" for s in symbols]
        await self._subscribe(streams, "orderbook", callback)

    async def subscribe_trades(self, symbols: List[str], callback: Callable):
        """Subscribe to aggregate trade streams"""
        streams = [f"{s.lower()}@aggTrade" for s in symbols]
        await self._subscribe(streams, "trades", callback)

    async def subscribe_mark_price(self, symbols: List[str], callback: Callable):
        """Subscribe to mark price streams (1s update)"""
        streams = [f"{s.lower()}@markPrice@1s" for s in symbols]
        await self._subscribe(streams, "markprice", callback)

    async def subscribe_kline(self, symbols: List[str], callback: Callable, interval: str = "1m"):
        """Subscribe to kline/candlestick streams"""
        streams = [f"{s.lower()}@kline_{interval}" for s in symbols]
        await self._subscribe(streams, f"kline_{interval}", callback)

    async def _subscribe(self, streams: List[str], stream_type: str, callback: Callable):
        self._callbacks[stream_type] = self._callbacks.get(stream_type, [])
        self._callbacks[stream_type].append(callback)

        # Binance allows max 200 streams per connection
        batch_size = 200
        for i in range(0, len(streams), batch_size):
            batch = streams[i:i + batch_size]
            stream_path = "/".join(batch)
            url = f"{self.base_url}/stream?streams={stream_path}"
            conn_id = f"{stream_type}_{i}"
            self._reconnect_tasks[conn_id] = asyncio.create_task(
                self._maintain_connection(conn_id, url, stream_type)
            )

    async def _maintain_connection(self, conn_id: str, url: str, stream_type: str):
        self._running = True
        while self._running:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=settings.binance.ws_ping_interval,
                    ping_timeout=settings.binance.ws_pong_timeout,
                    max_size=10 * 1024 * 1024,
                ) as ws:
                    self._connections[conn_id] = ws
                    logger.info(f"WS connected: {conn_id}")

                    async for message in ws:
                        try:
                            data = json.loads(message)
                            if "stream" in data and "data" in data:
                                for cb in self._callbacks.get(stream_type, []):
                                    try:
                                        await cb(data["data"])
                                    except Exception as e:
                                        logger.error(f"WS callback error [{stream_type}]: {e}")
                        except json.JSONDecodeError:
                            continue

            except Exception as e:
                logger.warning(f"WS disconnected [{conn_id}]: {e}, reconnecting in 5s...")
                await asyncio.sleep(5)

    async def close(self):
        self._running = False
        for conn_id, ws in self._connections.items():
            try:
                await ws.close()
            except Exception:
                pass
        for task in self._reconnect_tasks.values():
            task.cancel()
        self._connections.clear()
        self._reconnect_tasks.clear()
        logger.info("All WS connections closed")
