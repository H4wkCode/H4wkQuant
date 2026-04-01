"""
H4wkQuant - Spread Engine Service (:9102)
Core arbitrage signal generator.
Runs Bayesian + Edge + Kelly pipeline on tick data.
Publishes arb.signal to Redis.
"""
import asyncio
import json
import time
from typing import Dict, Optional
import redis.asyncio as redis
from aiohttp import web
from loguru import logger

from shared.config.settings import settings, setup_service_logging, apply_overrides
from shared.utils.redis_helper import get_redis_client, resilient_subscribe
from shared.utils.metrics import MetricsRegistry
from strategies.stat_arb import StatArbStrategy
from strategies.funding_arb import FundingArbStrategy
from strategies.momentum_div import MomentumDivStrategy
from strategies.cross_exchange import CrossExchangeStrategy
from shared.schemas.models import ArbSignal
from models.regime import RegimeDetector
from models.timeframe import MultiTimeframeAnalyzer
from models.portfolio import PortfolioOptimizer


class SpreadEngine:
    def __init__(self):
        self.redis: redis.Redis = None
        self.running = False

        # Initialize strategies
        self.stat_arb = StatArbStrategy()
        self.funding_arb = FundingArbStrategy()
        self.momentum_div = MomentumDivStrategy()
        self.cross_exchange = CrossExchangeStrategy()

        # Regime detection
        arb = settings.arbitrage
        self.regime_enabled = arb.regime_detection_enabled
        self.regime_detector = RegimeDetector(
            vol_window=arb.regime_vol_window,
            history_window=arb.regime_history_window,
            high_percentile=arb.regime_high_percentile,
            extreme_percentile=arb.regime_extreme_percentile,
        ) if self.regime_enabled else None

        # Multi-timeframe analyzer
        self.mtf_analyzer = MultiTimeframeAnalyzer()

        # Portfolio optimizer
        self.portfolio = PortfolioOptimizer()

        # Track active positions
        self.active_positions: Dict[str, dict] = {}

        # Signal cooldown: don't spam same signal
        self.signal_cooldown: Dict[str, float] = {}  # pair_id -> next_allowed timestamp

        # Warmup: no signals until 2 hours after start
        self.start_time = time.time()
        self.warmup_seconds = 1800  # 30 dakika
        self.pair_added_time: Dict[str, float] = {}  # pair_id -> when added

        # Metrics
        self.metrics = MetricsRegistry("spread_engine")
        self.m_signals = self.metrics.counter("h4wkquant_signals_generated", "Signals generated")
        self.m_ticks = self.metrics.counter("h4wkquant_ticks_processed", "Ticks processed")
        self.m_regime_changes = self.metrics.counter("h4wkquant_regime_changes", "Regime changes")
        self.m_positions = self.metrics.gauge("h4wkquant_positions_tracked", "Active positions tracked")

        # Stats
        self.signals_generated = 0
        self.ticks_processed = 0

    async def start(self):
        setup_service_logging("spread_engine")
        self.redis = await get_redis_client(settings.database.redis_url)
        self.running = True

        logger.info("Spread Engine starting...")

        # Apply config overrides
        await apply_overrides(self.redis)

        # Save start time for warmup tracking
        await self.redis.set("q:spread_engine:start_time", str(time.time()))

        # Subscribe to market data channels
        asyncio.create_task(self._subscribe_trades())
        asyncio.create_task(self._check_positions_loop())
        asyncio.create_task(self._check_funding_loop())
        asyncio.create_task(self._check_cross_exchange_loop())
        asyncio.create_task(self._watch_dynamic_pairs())
        asyncio.create_task(self._cache_spreads_loop())
        asyncio.create_task(self._listen_pair_changes())
        asyncio.create_task(self._health_server())
        asyncio.create_task(self._config_reload_loop())

        # Load active positions from Redis
        await self._load_positions()

        logger.info("Spread Engine started")

        while self.running:
            await asyncio.sleep(1)

    async def _subscribe_trades(self):
        """Subscribe to trade ticks and compute spreads (with auto-reconnect)"""
        async def handle_message(message):
            try:
                data = json.loads(message["data"])
                await self._on_tick(data)
            except Exception as e:
                logger.error(f"Tick processing error: {e}")

        await resilient_subscribe(self.redis, "q:trade", handle_message, "spread_engine")

    async def _on_tick(self, trade_data: Dict):
        """Process price tick - update strategies and check for signals"""
        symbol = trade_data["symbol"]
        price = trade_data["price"]
        self.ticks_processed += 1
        self.m_ticks.inc()

        # Update regime detector with BTC prices
        if self.regime_detector and symbol == "BTCUSDT":
            self.regime_detector.update(price)

        # Update momentum divergence
        self.momentum_div.update_tick(symbol, price)

        # Update multi-timeframe aggregator
        self.mtf_analyzer.update_tick(symbol, price)

        # Update stat arb price history
        for pair in self.stat_arb.pairs:
            sym_a, sym_b = pair.split("/")
            if symbol == sym_a:
                price_b = await self._get_price(sym_b)
                if price_b:
                    self.stat_arb.update_prices(pair, price, price_b)
                    await self._evaluate_stat_arb(pair, price, price_b)
            elif symbol == sym_b:
                price_a = await self._get_price(sym_a)
                if price_a:
                    self.stat_arb.update_prices(pair, price_a, price)
                    await self._evaluate_stat_arb(pair, price_a, price)

        # Check momentum divergence
        if symbol in self.momentum_div.followers:
            btc_price = await self._get_price("BTCUSDT")
            if btc_price:
                signal = await self.momentum_div.evaluate({
                    "symbol": symbol,
                    "price": price,
                    "btc_price": btc_price,
                    "equity": await self._get_equity(),
                })
                if signal:
                    await self._emit_signal(signal)

    async def _evaluate_stat_arb(self, pair_id: str, price_a: float, price_b: float):
        """Evaluate stat arb for a pair"""
        # Skip if already in position for this pair
        if pair_id in self.active_positions:
            return

        # Get additional market data
        sym_a, sym_b = pair_id.split("/")
        market_data = {
            "pair_id": pair_id,
            "price_a": price_a,
            "price_b": price_b,
            "equity": await self._get_equity(),
        }

        # Try to add orderbook data
        ob_a = await self._get_orderbook(sym_a)
        ob_b = await self._get_orderbook(sym_b)
        if ob_a:
            market_data["orderbook_a"] = ob_a
        if ob_b:
            market_data["orderbook_b"] = ob_b

        # Add funding rates
        funding_a = await self._get_funding(sym_a)
        funding_b = await self._get_funding(sym_b)
        if funding_a:
            market_data["funding_rate_a"] = funding_a
        if funding_b:
            market_data["funding_rate_b"] = funding_b

        signal = await self.stat_arb.evaluate(market_data)
        if signal:
            # Multi-timeframe check (only if enabled)
            if settings.arbitrage.multi_tf_enabled:
                import numpy as np
                history = self.stat_arb.price_history.get(pair_id)
                if history and len(history["a"]) >= 60:
                    mtf_result = self.mtf_analyzer.check_cointegration(
                        pair_id,
                        np.array(history["a"]),
                        np.array(history["b"]),
                        self.stat_arb.spread_model,
                    )
                    if not mtf_result.get("passes", True):
                        logger.debug(f"{pair_id}: MTF check failed ({mtf_result['agreement']}/3 timeframes)")
                        return
                    signal.metadata = signal.metadata or {}
                    signal.metadata["mtf_agreement"] = mtf_result["agreement"]

            # Portfolio correlation check (only if enabled)
            if settings.arbitrage.portfolio_corr_enabled:
                existing_pairs = list(self.active_positions.keys())
                allowed, correlations = self.portfolio.check_correlation(pair_id, existing_pairs)
                if not allowed:
                    logger.debug(f"{pair_id}: Portfolio correlation too high")
                    return

            await self._emit_signal(signal)

    async def _check_positions_loop(self):
        """Periodically check if positions should be closed"""
        while self.running:
            try:
                # Reload positions from Redis every cycle (executor writes here)
                raw = await self.redis.get("q:active_positions")
                if raw:
                    self.active_positions = json.loads(raw)
                else:
                    self.active_positions = {}

                self.m_positions.set(len(self.active_positions))

                for pair_id, pos_data in list(self.active_positions.items()):
                    strategy = pos_data.get("strategy", "stat_arb")

                    if strategy == "stat_arb":
                        sym_a, sym_b = pair_id.split("/")
                        # Auto-init price history for dynamic pairs
                        if pair_id not in self.stat_arb.price_history:
                            from collections import deque
                            self.stat_arb.price_history[pair_id] = {
                                "a": deque(maxlen=settings.arbitrage.lookback_window),
                                "b": deque(maxlen=settings.arbitrage.lookback_window),
                            }
                            if pair_id not in self.stat_arb.pairs:
                                self.stat_arb.pairs.append(pair_id)
                        price_a = await self._get_price(sym_a)
                        price_b = await self._get_price(sym_b)
                        if price_a and price_b:
                            self.stat_arb.update_prices(pair_id, price_a, price_b)
                            close_signal = await self.stat_arb.should_close(
                                pos_data, {"pair_id": pair_id}
                            )
                            if close_signal:
                                await self._emit_signal(close_signal)

                    elif strategy == "momentum_div":
                        close_signal = await self.momentum_div.should_close(
                            pos_data, {}
                        )
                        if close_signal:
                            await self._emit_signal(close_signal)

                    elif strategy == "funding_arb":
                        symbol = pos_data.get("leg_a_symbol", "")
                        funding_rate = await self._get_funding(symbol)
                        close_signal = await self.funding_arb.should_close(
                            pos_data, {"funding_rate": funding_rate or 0}
                        )
                        if close_signal:
                            await self._emit_signal(close_signal)

                    elif strategy == "cross_exchange":
                        symbol = pos_data.get("leg_a_symbol", "")
                        binance_price = await self._get_price(symbol)
                        bybit_price_raw = await self.redis.get(f"q:bybit:price:{symbol}")
                        bybit_price = json.loads(bybit_price_raw).get("price", 0) if bybit_price_raw else 0
                        if binance_price and bybit_price:
                            close_signal = await self.cross_exchange.should_close(
                                pos_data, {"binance_price": binance_price, "bybit_price": bybit_price}
                            )
                            if close_signal:
                                await self._emit_signal(close_signal)

            except Exception as e:
                logger.error(f"Position check error: {e}")

            await asyncio.sleep(5)  # Check every 5 seconds (was 1s, caused whipsaw exits)

    async def _check_funding_loop(self):
        """Check funding arb opportunities every minute"""
        while self.running:
            try:
                for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]:
                    funding_rate = await self._get_funding(symbol)
                    price = await self._get_price(symbol)

                    if funding_rate is None or price is None:
                        continue

                    hedge_map = self.funding_arb.hedge_map
                    hedge_symbol = hedge_map.get(symbol)
                    if not hedge_symbol:
                        continue

                    hedge_price = await self._get_price(hedge_symbol)
                    if not hedge_price:
                        continue

                    pair_id = f"{symbol}/{hedge_symbol}"
                    if pair_id in self.active_positions:
                        continue

                    signal = await self.funding_arb.evaluate({
                        "symbol": symbol,
                        "funding_rate": funding_rate,
                        "price": price,
                        "hedge_symbol": hedge_symbol,
                        "hedge_price": hedge_price,
                        "equity": await self._get_equity(),
                    })
                    if signal:
                        await self._emit_signal(signal)

            except Exception as e:
                logger.error(f"Funding check error: {e}")

            await asyncio.sleep(60)

    async def _check_cross_exchange_loop(self):
        """Check cross-exchange arb opportunities every 5 seconds"""
        while self.running:
            try:
                if not settings.bybit.enabled:
                    await asyncio.sleep(60)
                    continue

                for symbol in self.cross_exchange.symbols:
                    binance_price = await self._get_price(symbol)
                    bybit_raw = await self.redis.get(f"q:bybit:price:{symbol}")
                    bybit_price = json.loads(bybit_raw).get("price", 0) if bybit_raw else 0

                    if not binance_price or not bybit_price:
                        continue

                    pair_id = f"{symbol}:CE"
                    if pair_id in self.active_positions:
                        continue

                    signal = await self.cross_exchange.evaluate({
                        "symbol": symbol,
                        "binance_price": binance_price,
                        "bybit_price": bybit_price,
                        "equity": await self._get_equity(),
                    })
                    if signal:
                        await self._emit_signal(signal)

            except Exception as e:
                logger.error(f"Cross-exchange check error: {e}")

            await asyncio.sleep(5)

    async def _emit_signal(self, signal: ArbSignal):
        """Publish signal to Redis (with per-pair cooldown to prevent spam)"""
        now = time.time()
        pair_id = signal.pair_id

        # Open signals: block during warmup + regime check + max 1 per pair per 60 seconds
        if signal.action.value == "open":
            # Global warmup (engine restart)
            if now - self.start_time < self.warmup_seconds:
                return
            # Per-pair warmup (dynamic pair added)
            pair_start = self.pair_added_time.get(pair_id, self.start_time)
            if now - pair_start < self.warmup_seconds:
                return
            # Regime check: block new positions in HIGH/EXTREME
            if self.regime_detector:
                regime_result = self.regime_detector.last_result
                if regime_result and not regime_result.allow_new_positions:
                    logger.debug(f"Signal blocked by regime: {regime_result.regime.value} (pair={pair_id})")
                    return
            next_allowed = self.signal_cooldown.get(pair_id, 0)
            if now < next_allowed:
                return
            self.signal_cooldown[pair_id] = now + 180  # was 60s - prevent overtrading

        signal_data = signal.model_dump(mode="json")
        signal_data["timestamp"] = now

        await self.redis.publish("arb.signal", json.dumps(signal_data))

        # Store in history
        await self.redis.lpush("q:signals:history", json.dumps(signal_data))
        await self.redis.ltrim("q:signals:history", 0, 499)

        self.signals_generated += 1
        self.m_signals.inc()
        logger.info(
            f"Signal: {signal.strategy.value} {signal.action.value} "
            f"{signal.pair_id} z={signal.zscore:.2f} edge={signal.edge_net:.6f}"
        )

    async def _get_price(self, symbol: str) -> Optional[float]:
        data = await self.redis.get(f"q:price:{symbol}")
        if data:
            return json.loads(data)["price"]
        return None

    async def _get_orderbook(self, symbol: str) -> Optional[Dict]:
        data = await self.redis.get(f"q:orderbook:{symbol}")
        if data:
            return json.loads(data)
        return None

    async def _get_funding(self, symbol: str) -> Optional[float]:
        data = await self.redis.get(f"q:funding:{symbol}")
        if data:
            return json.loads(data).get("funding_rate")
        return None

    async def _get_equity(self) -> float:
        data = await self.redis.get("q:account:balance")
        if data:
            return json.loads(data).get("total_balance", 174.0)
        return settings.risk.paper_initial_balance

    async def _cache_spreads_loop(self):
        """Cache z-score data for all pairs every 2 seconds for dashboard"""
        import numpy as np
        while self.running:
            try:
                now = time.time()
                for pair_id in list(self.stat_arb.pairs):
                    history = self.stat_arb.price_history.get(pair_id)
                    ticks = len(history["a"]) if history else 0

                    pair_start = self.pair_added_time.get(pair_id, self.start_time)
                    warmup_remaining = max(0, int(self.warmup_seconds - (now - pair_start)))

                    spread_data = {
                        "ticks": ticks,
                        "warmup_remaining": warmup_remaining,
                        "is_ready": warmup_remaining == 0,
                    }

                    if history and ticks >= 60:
                        try:
                            result = self.stat_arb.spread_model.compute(
                                np.array(history["a"]), np.array(history["b"]),
                                pair_id=pair_id,
                            )
                            spread_data["zscore"] = round(result.zscore, 4)
                            spread_data["half_life"] = round(result.half_life, 1)
                            spread_data["coint_pvalue"] = round(result.coint_pvalue, 4)
                            spread_data["hedge_ratio"] = round(result.hedge_ratio, 6)

                            # Update portfolio spread tracking
                            self.portfolio.update_spread(pair_id, result.spread)

                            # Multi-timeframe cointegration info
                            mtf = self.mtf_analyzer.check_cointegration(
                                pair_id, np.array(history["a"]), np.array(history["b"]),
                                self.stat_arb.spread_model,
                            )
                            spread_data["tf_5m_coint"] = bool(mtf.get("5m", {}).get("cointegrated", False))
                            spread_data["tf_15m_coint"] = bool(mtf.get("15m", {}).get("cointegrated", False))
                        except Exception:
                            pass

                    await self.redis.set(f"q:spread:{pair_id}", json.dumps(spread_data), ex=10)

                # Cache regime data
                if self.regime_detector:
                    await self.redis.set("q:regime", json.dumps(self.regime_detector.to_dict()), ex=10)

                # Cache portfolio correlation matrix
                all_pairs = list(self.stat_arb.pairs)
                if len(all_pairs) >= 2:
                    corr_matrix = self.portfolio.compute_correlation_matrix(all_pairs)
                    await self.redis.set("q:portfolio:correlation", json.dumps(corr_matrix), ex=30)

            except Exception as e:
                logger.error(f"Spread cache error: {e}")
            await asyncio.sleep(2)

    async def _watch_dynamic_pairs(self):
        """Sync screened pairs from Redis - add new, remove stale"""
        from collections import deque
        default_pairs = set(settings.arbitrage.pairs)

        while self.running:
            try:
                screened = await self.redis.get("q:config:screened_pairs")
                redis_pairs = set(json.loads(screened)) if screened else set()

                # Add new pairs
                for pair in redis_pairs:
                    if pair not in self.stat_arb.pairs:
                        self.stat_arb.pairs.append(pair)
                        self.stat_arb.price_history[pair] = {
                            "a": deque(maxlen=settings.arbitrage.lookback_window),
                            "b": deque(maxlen=settings.arbitrage.lookback_window),
                        }
                        self.pair_added_time[pair] = time.time()
                        logger.info(f"Dynamic pair added: {pair}")

                # Remove pairs no longer in Redis (keep defaults + active positions)
                active_pairs = set(self.active_positions.keys())
                keep = default_pairs | redis_pairs | active_pairs
                removed = [p for p in self.stat_arb.pairs if p not in keep]
                for pair in removed:
                    self.stat_arb.pairs.remove(pair)
                    self.stat_arb.price_history.pop(pair, None)
                    logger.info(f"Dynamic pair removed: {pair}")

            except Exception as e:
                logger.error(f"Dynamic pair watch error: {e}")

            await asyncio.sleep(30)

    async def _listen_pair_changes(self):
        """Listen for pair cleanup commands - react instantly"""
        async def handle_message(message):
            cmd = message["data"]
            if cmd == "clear":
                logger.info("Received pair clear command - removing all dynamic pairs")
                default_pairs = set(settings.arbitrage.pairs)
                removed = [p for p in self.stat_arb.pairs if p not in default_pairs]
                for pair in removed:
                    self.stat_arb.pairs.remove(pair)
                    self.stat_arb.price_history.pop(pair, None)
                    self.signal_cooldown.pop(pair, None)
                    logger.info(f"Dynamic pair removed: {pair}")

        await resilient_subscribe(self.redis, "q:pairs:command", handle_message, "spread_engine")

    async def _config_reload_loop(self):
        """Reload config overrides from Redis every 30 seconds"""
        while self.running:
            try:
                await apply_overrides(self.redis)
                # Update warmup_seconds from overrides
                overrides_raw = await self.redis.get("q:config:overrides")
                if overrides_raw:
                    overrides = json.loads(overrides_raw)
                    if "warmup_seconds" in overrides:
                        self.warmup_seconds = int(overrides["warmup_seconds"])
            except Exception as e:
                logger.error(f"Config reload error: {e}")
            await asyncio.sleep(30)

    async def _load_positions(self):
        data = await self.redis.get("q:active_positions")
        if data:
            self.active_positions = json.loads(data)
            logger.info(f"Loaded {len(self.active_positions)} active positions")

    async def _health_server(self):
        app = web.Application()
        app.router.add_get("/health", self._health_handler)
        app.router.add_get("/metrics", self._metrics_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", 9102)
        await site.start()

    async def _health_handler(self, request):
        return web.json_response({
            "service": "spread_engine",
            "status": "healthy",
            "signals_generated": self.signals_generated,
            "ticks_processed": self.ticks_processed,
            "active_positions": len(self.active_positions),
        })

    async def _metrics_handler(self, request):
        return web.Response(text=self.metrics.render(), content_type="text/plain")

    async def stop(self):
        self.running = False
        logger.info("Spread Engine stopped")


async def main():
    engine = SpreadEngine()
    try:
        await engine.start()
    except KeyboardInterrupt:
        await engine.stop()


if __name__ == "__main__":
    asyncio.run(main())
