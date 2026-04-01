"""
H4wkQuant - Data Collector Service (:9101)
Collects orderbook, trades, mark price, funding rate, OI from Binance
Publishes to Redis for other services
"""
import asyncio
import json
import time
from typing import Dict, Set
import redis.asyncio as redis
from aiohttp import web
from loguru import logger

from shared.config.settings import settings, setup_service_logging, apply_overrides, load_api_keys_from_redis
from shared.utils.redis_helper import get_redis_client
from shared.utils.metrics import MetricsRegistry
from shared.clients.binance_rest import get_binance_client
from shared.clients.binance_ws import BinanceWSClient


class DataCollector:
    def __init__(self):
        self.redis: redis.Redis = None
        self.rest_client = get_binance_client()
        self.ws_client = BinanceWSClient()
        self.running = False

        # Base symbols from config
        self.base_symbols: Set[str] = set()
        for pair in settings.arbitrage.pairs:
            a, b = pair.split("/")
            self.base_symbols.add(a)
            self.base_symbols.add(b)
        self.base_symbols.update(["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])

        self.symbols = list(self.base_symbols)
        self.tracked_symbols: Set[str] = set(self.symbols)
        logger.info(f"Tracking {len(self.symbols)} symbols: {self.symbols}")

        # Bybit support (optional)
        self.bybit_ws = None
        self.bybit_enabled = settings.bybit.enabled

        # Metrics
        self.metrics = MetricsRegistry("data_collector")
        self.m_ticks = self.metrics.counter("h4wkquant_ticks_received", "Trade ticks received")
        self.m_ws_reconnects = self.metrics.counter("h4wkquant_ws_reconnects", "WebSocket reconnections")
        self.m_symbols = self.metrics.gauge("h4wkquant_symbols_tracked", "Symbols being tracked")

    async def start(self):
        setup_service_logging("data_collector")
        self.redis = await get_redis_client(settings.database.redis_url)
        self.running = True

        logger.info("Data Collector starting...")

        # Load API keys from panel (encrypted in Redis)
        await load_api_keys_from_redis(self.redis)
        logger.info(f"API key loaded: {'YES' if settings.binance.api_key else 'NO'} (len={len(settings.binance.api_key)})")
        # Re-init REST client with loaded keys
        self.rest_client = get_binance_client(force_new=True)

        # Start WebSocket streams
        await self.ws_client.subscribe_orderbook(
            self.symbols, self._on_orderbook, depth=20
        )
        await self.ws_client.subscribe_trades(
            self.symbols, self._on_trade
        )
        await self.ws_client.subscribe_mark_price(
            self.symbols, self._on_mark_price
        )

        # Cache exchange info to Redis (executor reads from here)
        await self._cache_exchange_info()

        # Start REST polling tasks
        asyncio.create_task(self._poll_funding_rates())
        asyncio.create_task(self._poll_open_interest())
        asyncio.create_task(self._sync_account_balance())
        asyncio.create_task(self._watch_dynamic_pairs())

        # Health check server
        asyncio.create_task(self._health_server())

        # Bybit WebSocket (optional)
        if self.bybit_enabled:
            asyncio.create_task(self._start_bybit_ws())

        # Apply config overrides
        await apply_overrides(self.redis)

        self.m_symbols.set(len(self.symbols))
        logger.info("Data Collector started successfully")

        while self.running:
            await asyncio.sleep(1)

    async def _on_orderbook(self, data: Dict):
        """Process orderbook depth update"""
        try:
            symbol = data.get("s", "")
            if not symbol:
                return

            ob_data = {
                "bids": data.get("b", [])[:20],
                "asks": data.get("a", [])[:20],
                "time": data.get("T", int(time.time() * 1000)),
            }

            # Store in Redis (60s TTL)
            await self.redis.setex(
                f"q:orderbook:{symbol}", 60,
                json.dumps(ob_data)
            )

            # Publish for real-time consumers
            await self.redis.publish(
                "q:orderbook_update",
                json.dumps({"symbol": symbol, **ob_data})
            )
        except Exception as e:
            logger.error(f"Orderbook processing error: {e}")

    async def _on_trade(self, data: Dict):
        """Process aggregate trade"""
        try:
            symbol = data.get("s", "")
            price = float(data.get("p", 0))
            quantity = float(data.get("q", 0))
            is_buyer_maker = data.get("m", False)
            trade_time = data.get("T", int(time.time() * 1000))

            trade_data = {
                "symbol": symbol,
                "price": price,
                "quantity": quantity,
                "side": "SELL" if is_buyer_maker else "BUY",
                "time": trade_time,
            }

            # Store latest price
            await self.redis.setex(
                f"q:price:{symbol}", 30,
                json.dumps({"price": price, "time": trade_time})
            )

            # Publish tick
            await self.redis.publish("q:trade", json.dumps(trade_data))
            self.m_ticks.inc()
        except Exception as e:
            logger.error(f"Trade processing error: {e}")

    async def _on_mark_price(self, data: Dict):
        """Process mark price update"""
        try:
            symbol = data.get("s", "")
            mark_price = float(data.get("p", 0))
            index_price = float(data.get("i", 0))
            funding_rate = float(data.get("r", 0))
            next_funding_time = data.get("T", 0)

            mp_data = {
                "symbol": symbol,
                "mark_price": mark_price,
                "index_price": index_price,
                "funding_rate": funding_rate,
                "next_funding_time": next_funding_time,
                "time": int(time.time() * 1000),
            }

            await self.redis.setex(
                f"q:markprice:{symbol}", 10,
                json.dumps(mp_data)
            )
        except Exception as e:
            logger.error(f"Mark price processing error: {e}")

    async def _poll_funding_rates(self):
        """Poll funding rates every 5 minutes"""
        while self.running:
            try:
                for symbol in self.symbols:
                    try:
                        data = await self.rest_client.get_premium_index(symbol)
                        funding_data = {
                            "symbol": symbol,
                            "funding_rate": float(data.get("lastFundingRate", 0)),
                            "next_funding_time": int(data.get("nextFundingTime", 0)),
                            "mark_price": float(data.get("markPrice", 0)),
                            "index_price": float(data.get("indexPrice", 0)),
                        }
                        await self.redis.setex(
                            f"q:funding:{symbol}", 600,
                            json.dumps(funding_data)
                        )
                    except Exception as e:
                        logger.warning(f"Funding rate fetch failed for {symbol}: {e}")
                    await asyncio.sleep(0.2)  # Rate limit spacing
            except Exception as e:
                logger.error(f"Funding rate poll error: {e}")

            await asyncio.sleep(300)  # Every 5 minutes

    async def _poll_open_interest(self):
        """Poll open interest every 5 minutes"""
        while self.running:
            try:
                for symbol in self.symbols:
                    try:
                        data = await self.rest_client.get_open_interest(symbol)
                        oi_data = {
                            "symbol": symbol,
                            "open_interest": float(data.get("openInterest", 0)),
                            "time": int(data.get("time", time.time() * 1000)),
                        }
                        await self.redis.setex(
                            f"q:oi:{symbol}", 600,
                            json.dumps(oi_data)
                        )
                    except Exception as e:
                        logger.warning(f"OI fetch failed for {symbol}: {e}")
                    await asyncio.sleep(0.2)
            except Exception as e:
                logger.error(f"OI poll error: {e}")

            await asyncio.sleep(300)

    async def _watch_dynamic_pairs(self):
        """Check Redis for new screened pairs every 30s and subscribe to their symbols"""
        while self.running:
            try:
                screened = await self.redis.get("q:config:screened_pairs")
                if screened:
                    pairs = json.loads(screened)
                    new_symbols = set()
                    for pair in pairs:
                        a, b = pair.split("/")
                        new_symbols.add(a)
                        new_symbols.add(b)

                    # Find symbols we're not yet tracking
                    to_add = new_symbols - self.tracked_symbols
                    if to_add:
                        logger.info(f"Dynamic pairs: subscribing to {to_add}")
                        self.tracked_symbols.update(to_add)
                        self.symbols = list(self.tracked_symbols)

                        # Subscribe to new symbols
                        to_add_list = list(to_add)
                        await self.ws_client.subscribe_trades(to_add_list, self._on_trade)
                        await self.ws_client.subscribe_mark_price(to_add_list, self._on_mark_price)
            except Exception as e:
                logger.error(f"Dynamic pair watch error: {e}")

            await asyncio.sleep(30)

    async def _sync_account_balance(self):
        """Sync real Binance account balance every 30 seconds (only in live mode)"""
        while self.running:
            try:
                # In paper mode we don't write Binance balance - executor manages virtual balance
                if settings.trading_mode.value == "live":
                    balance = await self.rest_client.get_balance()
                    if balance:
                        account_data = {
                            "total_balance": float(balance.get("balance", 0)),
                            "available_balance": float(balance.get("availableBalance", 0)),
                            "unrealized_pnl": float(balance.get("crossUnPnl", 0)),
                            "time": int(time.time() * 1000),
                        }
                        await self.redis.set("q:account:balance", json.dumps(account_data))
                        logger.debug(f"Balance synced: ${account_data['total_balance']:.2f}")
                # In paper mode don't make Binance REST calls - wastes unnecessary rate limits
            except Exception as e:
                logger.warning(f"Balance sync failed: {e}")

            await asyncio.sleep(30)

    async def _start_bybit_ws(self):
        """Start Bybit WebSocket for cross-exchange arb"""
        try:
            from shared.clients.bybit_ws import BybitWSClient
            self.bybit_ws = BybitWSClient(testnet=settings.bybit.testnet)
            await self.bybit_ws.connect()

            symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
            await self.bybit_ws.subscribe_trades(symbols, self._on_bybit_trade)
            logger.info(f"Bybit WS connected, tracking {len(symbols)} symbols")
        except Exception as e:
            logger.error(f"Bybit WS start failed: {e}")

    async def _on_bybit_trade(self, data):
        """Process Bybit trade data"""
        try:
            if isinstance(data, list):
                for item in data:
                    symbol = item.get("s", "")
                    price = float(item.get("p", 0))
                    if symbol and price > 0:
                        await self.redis.setex(
                            f"q:bybit:price:{symbol}", 30,
                            json.dumps({"price": price, "time": int(time.time() * 1000)})
                        )
            elif isinstance(data, dict):
                symbol = data.get("s", "")
                price = float(data.get("p", 0))
                if symbol and price > 0:
                    await self.redis.setex(
                        f"q:bybit:price:{symbol}", 30,
                        json.dumps({"price": price, "time": int(time.time() * 1000)})
                    )
        except Exception as e:
            logger.error(f"Bybit trade processing error: {e}")

    async def _health_server(self):
        """Health check endpoint on :9101"""
        app = web.Application()
        app.router.add_get("/health", self._health_handler)
        app.router.add_get("/metrics", self._metrics_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", 9101)
        await site.start()
        logger.info("Health check on :9101")

    async def _health_handler(self, request):
        return web.json_response({
            "service": "data_collector",
            "status": "healthy",
            "symbols": len(self.symbols),
            "ws_connections": len(self.ws_client._connections),
        })

    async def _metrics_handler(self, request):
        return web.Response(text=self.metrics.render(), content_type="text/plain")

    async def _cache_exchange_info(self):
        """Fetch exchange info once and cache to Redis for other services"""
        try:
            # Skip if cache is still fresh (< 12h old)
            ttl = await self.redis.ttl("q:exchange_info_cache")
            if ttl and ttl > 43200:  # > 12h remaining = fresh enough
                logger.info("Exchange info cache still fresh, skipping API call")
                return

            info = await self.rest_client.get_exchange_info()
            symbol_info = {}
            for s in info.get("symbols", []):
                symbol = s["symbol"]
                for f in s.get("filters", []):
                    if f["filterType"] == "LOT_SIZE":
                        symbol_info[symbol] = {
                            "step_size": float(f["stepSize"]),
                            "min_qty": float(f["minQty"]),
                        }
                    elif f["filterType"] == "PRICE_FILTER":
                        symbol_info.setdefault(symbol, {})
                        symbol_info[symbol]["tick_size"] = float(f["tickSize"])
            await self.redis.set("q:exchange_info_cache", json.dumps(symbol_info), ex=86400)
            logger.info(f"Exchange info cached to Redis: {len(symbol_info)} symbols (24h TTL)")
        except Exception as e:
            logger.warning(f"Failed to cache exchange info: {e} - executor will retry")

    async def stop(self):
        self.running = False
        await self.ws_client.close()
        await self.rest_client.close()
        logger.info("Data Collector stopped")


async def main():
    collector = DataCollector()
    try:
        await collector.start()
    except KeyboardInterrupt:
        await collector.stop()


if __name__ == "__main__":
    asyncio.run(main())
