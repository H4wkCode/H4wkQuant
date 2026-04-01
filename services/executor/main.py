"""
H4wkQuant - Executor Service (:9104)
Executes approved arb signals on Binance Futures.
Manages two-legged orders with Stoikov limit pricing.
Paper trading mode supported.
"""
import asyncio
import json
import time
import uuid
from typing import Dict, Optional
import redis.asyncio as redis
from aiohttp import web
from loguru import logger

from shared.config.settings import settings, setup_service_logging, apply_overrides, load_api_keys_from_redis
from shared.utils.redis_helper import get_redis_client, resilient_subscribe
from shared.utils.metrics import MetricsRegistry
from shared.clients.binance_rest import get_binance_client
from shared.database.connection import get_engine, get_session_factory
from shared.database.models import Base, ArbTrade, AccountSnapshot
from models.stoikov import StoikovModel
from datetime import datetime


class Executor:
    def __init__(self):
        self.redis: redis.Redis = None
        self.rest_client = get_binance_client()
        self.stoikov = StoikovModel(
            gamma=settings.arbitrage.stoikov_gamma,
            k=settings.arbitrage.stoikov_k,
        )
        self.running = False

        # Active positions
        self.positions: Dict[str, dict] = {}

        # Cooldown: prevent re-opening same pair right after close
        self.close_cooldown: Dict[str, float] = {}  # pair_id -> cooldown_until timestamp

        # Paper trading
        self.is_paper = settings.trading_mode != "live"

        # Exchange info cache
        self.symbol_info: Dict[str, dict] = {}

        # Database
        self.db_factory = None
        self.initial_equity = settings.risk.paper_initial_balance

        # Bybit client (optional)
        self.bybit_client = None

        # Metrics
        self.metrics = MetricsRegistry("executor")
        self.m_orders = self.metrics.counter("h4wkquant_orders_executed", "Orders executed")
        self.m_positions_opened = self.metrics.counter("h4wkquant_positions_opened", "Positions opened")
        self.m_positions_closed = self.metrics.counter("h4wkquant_positions_closed", "Positions closed")
        self.m_total_pnl = self.metrics.gauge("h4wkquant_total_pnl", "Total PnL")
        self.m_positions_open = self.metrics.gauge("h4wkquant_positions_open", "Currently open positions")

        # Stats
        self.orders_executed = 0

    async def start(self):
        setup_service_logging("executor")
        self.redis = await get_redis_client(settings.database.redis_url)
        self.running = True

        logger.info(f"Executor starting in {'PAPER' if self.is_paper else 'LIVE'} mode...")

        # Initialize PostgreSQL
        try:
            engine = get_engine()
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            self.db_factory = get_session_factory()
            logger.info("PostgreSQL initialized - trade persistence active")
        except Exception as e:
            logger.warning(f"PostgreSQL not available, using Redis only: {e}")
            self.db_factory = None

        # Load API keys from panel (encrypted in Redis)
        await load_api_keys_from_redis(self.redis)
        # Re-init REST client with loaded keys
        self.rest_client = get_binance_client(force_new=True)

        # Apply config overrides
        await apply_overrides(self.redis)

        # Initialize virtual balance if not set
        if self.is_paper:
            existing_balance = await self.redis.get("q:account:balance")
            if not existing_balance:
                initial = self.initial_equity
                await self.redis.set("q:account:balance", json.dumps({
                    "total_balance": initial,
                    "available_balance": initial,
                    "unrealized_pnl": 0,
                }))
                await self.redis.set("q:risk:equity_peak", str(initial))
                logger.info(f"Paper balance initialized: ${initial}")

        # Load exchange info
        await self._load_exchange_info()

        # Load existing positions
        await self._load_positions()

        # Initialize Bybit client if enabled
        if settings.bybit.enabled:
            try:
                from shared.clients.bybit_rest import get_bybit_client
                self.bybit_client = get_bybit_client()
                logger.info("Bybit client initialized for cross-exchange execution")
            except Exception as e:
                logger.warning(f"Bybit client init failed: {e}")

        # Subscribe to approved signals
        asyncio.create_task(self._subscribe_approved())
        asyncio.create_task(self._health_server())
        asyncio.create_task(self._config_reload_loop())

        logger.info("Executor started")

        while self.running:
            await asyncio.sleep(1)

    async def _subscribe_approved(self):
        """Listen for risk-approved signals (with auto-reconnect)"""
        async def handle_message(message):
            try:
                signal_data = json.loads(message["data"])
                await self._execute_signal(signal_data)
            except Exception as e:
                logger.error(f"Execution error: {e}")

        await resilient_subscribe(self.redis, "arb.approved", handle_message, "executor")

    async def _execute_signal(self, signal: Dict):
        """Execute an approved arb signal"""
        action = signal.get("action")
        pair_id = signal.get("pair_id")

        if action == "open":
            # Atomic lock per pair to prevent race condition (Bug #1)
            lock_key = f"q:lock:open:{pair_id}"
            acquired = await self.redis.set(lock_key, "1", nx=True, ex=30)
            if not acquired:
                logger.warning(f"Lock held for {pair_id}, skipping duplicate open")
                return
            try:
                # Cooldown check: don't re-open right after close (5 min)
                cooldown_until = self.close_cooldown.get(pair_id, 0)
                if time.time() < cooldown_until:
                    remaining = int(cooldown_until - time.time())
                    logger.info(f"Cooldown active for {pair_id}, {remaining}s remaining, skipping")
                    return

                # Re-check position count atomically (Bug #4)
                pos_raw = await self.redis.get("q:active_positions")
                current_positions = json.loads(pos_raw) if pos_raw else {}
                if pair_id in current_positions:
                    logger.warning(f"Already in position for {pair_id}, skipping")
                    return
                if len(current_positions) >= settings.risk.max_open_positions:
                    logger.warning(f"Max positions reached ({len(current_positions)}), skipping")
                    return
                await self._open_position(signal)
            finally:
                await self.redis.delete(lock_key)
        elif action == "close":
            await self._close_position(signal)
        else:
            logger.warning(f"Unknown action: {action}")

    async def _open_position(self, signal: Dict):
        """Open a two-legged arbitrage position"""
        pair_id = signal["pair_id"]
        leg_a = signal["leg_a"]
        leg_b = signal["leg_b"]
        position_size = signal.get("position_size_usd", 50.0)

        sym_a = leg_a["symbol"]
        sym_b = leg_b["symbol"]
        side_a = leg_a["side"]
        side_b = leg_b["side"]

        # Get current prices for Stoikov optimization
        price_a = await self._get_price(sym_a)
        price_b = await self._get_price(sym_b)

        if not price_a or not price_b:
            logger.error(f"Cannot get prices for {pair_id}")
            return

        # Calculate quantities - position_size is margin, multiply by leverage for notional
        leverage = settings.risk.default_leverage
        notional_per_leg = position_size * leverage
        qty_a = self._calculate_quantity(sym_a, notional_per_leg, price_a)
        qty_b = self._calculate_quantity(sym_b, notional_per_leg, price_b)

        if qty_a <= 0 or qty_b <= 0:
            logger.error(f"Invalid quantities: {sym_a}={qty_a}, {sym_b}={qty_b}")
            return

        trade_id = str(uuid.uuid4())[:8]

        if self.is_paper:
            # Paper trading - simulate fills
            logger.info(
                f"PAPER ENTRY [{trade_id}] {pair_id}: "
                f"{side_a} {sym_a} qty={qty_a:.6f} @ {price_a:.2f} | "
                f"{side_b} {sym_b} qty={qty_b:.6f} @ {price_b:.2f}"
            )
            fill_a = price_a
            fill_b = price_b
        else:
            # Live trading
            try:
                # Set leverage
                await self.rest_client.set_leverage(sym_a, leverage)
                await self.rest_client.set_leverage(sym_b, leverage)
                await self.rest_client.set_margin_type(sym_a, "ISOLATED")
                await self.rest_client.set_margin_type(sym_b, "ISOLATED")

                # Get Stoikov-optimized prices
                vol_a = await self._get_volatility(sym_a)
                vol_b = await self._get_volatility(sym_b)

                exec_price_a = self.stoikov.execution_price(
                    price_a, "BUY" if side_a == "LONG" else "SELL",
                    vol_a, urgency=0.7
                )
                exec_price_b = self.stoikov.execution_price(
                    price_b, "BUY" if side_b == "LONG" else "SELL",
                    vol_b, urgency=0.7
                )

                # Place both legs (try limit first, fallback to market)
                binance_side_a = "BUY" if side_a == "LONG" else "SELL"
                binance_side_b = "BUY" if side_b == "LONG" else "SELL"

                result_a = await self.rest_client.place_order(
                    sym_a, binance_side_a, "LIMIT", qty_a, price=exec_price_a
                )
                result_b = await self.rest_client.place_order(
                    sym_b, binance_side_b, "LIMIT", qty_b, price=exec_price_b
                )

                fill_a = float(result_a.get("avgPrice", price_a))
                fill_b = float(result_b.get("avgPrice", price_b))

                logger.info(
                    f"LIVE ENTRY [{trade_id}] {pair_id}: "
                    f"{side_a} {sym_a} @ {fill_a:.2f} | "
                    f"{side_b} {sym_b} @ {fill_b:.2f}"
                )
            except Exception as e:
                logger.error(f"Order placement failed: {e}")
                # Cancel any partially filled orders
                try:
                    await self.rest_client.cancel_all_orders(sym_a)
                    await self.rest_client.cancel_all_orders(sym_b)
                except Exception:
                    pass
                return

        # Record position
        position = {
            "trade_id": trade_id,
            "pair_id": pair_id,
            "strategy": signal.get("strategy", "stat_arb"),
            "leg_a_symbol": sym_a,
            "leg_a_side": side_a,
            "leg_a_entry": fill_a,
            "leg_a_qty": qty_a,
            "leg_b_symbol": sym_b,
            "leg_b_side": side_b,
            "leg_b_entry": fill_b,
            "leg_b_qty": qty_b,
            "entry_zscore": signal.get("zscore", 0),
            "entry_time": time.time(),
            "leverage": leverage,
        }

        self.positions[pair_id] = position
        await self._save_positions()
        await self._save_account_snapshot()

        self.orders_executed += 1
        self.m_orders.inc()
        self.m_positions_opened.inc()
        self.m_positions_open.set(len(self.positions))

        # Publish execution update
        await self.redis.publish("arb.execution", json.dumps({
            "type": "entry",
            **position,
        }))

    async def _close_position(self, signal: Dict):
        """Close an existing arbitrage position"""
        pair_id = signal["pair_id"]

        if pair_id not in self.positions:
            logger.warning(f"No position to close for {pair_id}")
            return

        pos = self.positions[pair_id]
        sym_a = pos["leg_a_symbol"]
        sym_b = pos["leg_b_symbol"]

        price_a = await self._get_price(sym_a)
        price_b = await self._get_price(sym_b)

        if not price_a or not price_b:
            logger.error(f"Cannot get exit prices for {pair_id}")
            return

        if self.is_paper:
            exit_a = price_a
            exit_b = price_b
        else:
            try:
                # Close leg A
                close_side_a = "SELL" if pos["leg_a_side"] == "LONG" else "BUY"
                result_a = await self.rest_client.place_order(
                    sym_a, close_side_a, "MARKET", pos["leg_a_qty"], reduce_only=True
                )
                # Close leg B
                close_side_b = "SELL" if pos["leg_b_side"] == "LONG" else "BUY"
                result_b = await self.rest_client.place_order(
                    sym_b, close_side_b, "MARKET", pos["leg_b_qty"], reduce_only=True
                )
                exit_a = float(result_a.get("avgPrice", price_a))
                exit_b = float(result_b.get("avgPrice", price_b))
            except Exception as e:
                logger.error(f"Close order failed: {e}")
                return

        # Calculate PnL
        pnl_a = self._calculate_leg_pnl(
            pos["leg_a_side"], pos["leg_a_entry"], exit_a, pos["leg_a_qty"]
        )
        pnl_b = self._calculate_leg_pnl(
            pos["leg_b_side"], pos["leg_b_entry"], exit_b, pos["leg_b_qty"]
        )
        combined_pnl = pnl_a + pnl_b

        # Commission
        commission = (
            pos["leg_a_qty"] * pos["leg_a_entry"] * settings.risk.maker_fee +
            pos["leg_a_qty"] * exit_a * settings.risk.taker_fee +
            pos["leg_b_qty"] * pos["leg_b_entry"] * settings.risk.maker_fee +
            pos["leg_b_qty"] * exit_b * settings.risk.taker_fee
        )
        net_pnl = combined_pnl - commission

        exit_reason = signal.get("metadata", {}).get("exit_reason", "manual")

        logger.info(
            f"{'PAPER' if self.is_paper else 'LIVE'} EXIT [{pos['trade_id']}] {pair_id}: "
            f"PnL=${net_pnl:.4f} (gross=${combined_pnl:.4f} - comm=${commission:.4f}) "
            f"reason={exit_reason}"
        )

        # Update daily PnL
        daily_pnl = float(await self.redis.get("q:risk:daily_pnl") or "0")
        await self.redis.set("q:risk:daily_pnl", str(daily_pnl + net_pnl))

        # Update virtual balance for paper trading
        if self.is_paper:
            balance_raw = await self.redis.get("q:account:balance")
            balance = json.loads(balance_raw) if balance_raw else {"total_balance": self.initial_equity, "available_balance": self.initial_equity}
            balance["total_balance"] = round(balance["total_balance"] + net_pnl, 4)
            balance["available_balance"] = round(balance["total_balance"], 4)
            balance["unrealized_pnl"] = 0
            await self.redis.set("q:account:balance", json.dumps(balance))
            # Update equity peak for drawdown tracking
            peak = float(await self.redis.get("q:risk:equity_peak") or str(self.initial_equity))
            if balance["total_balance"] > peak:
                await self.redis.set("q:risk:equity_peak", str(balance["total_balance"]))

        # Store trade record
        trade_record = {
            **pos,
            "leg_a_exit": exit_a,
            "leg_b_exit": exit_b,
            "combined_pnl": round(net_pnl, 6),
            "gross_pnl": round(combined_pnl, 6),
            "commission": round(commission, 6),
            "exit_reason": exit_reason,
            "exit_time": time.time(),
            "exit_zscore": signal.get("zscore", 0),
            "trading_mode": "paper" if self.is_paper else "live",
        }
        await self.redis.lpush("q:trades:history", json.dumps(trade_record))
        await self.redis.ltrim("q:trades:history", 0, 499)

        # Persist to PostgreSQL
        await self._persist_trade(pos, exit_a, exit_b, net_pnl, combined_pnl, commission, exit_reason, signal)

        # Remove position + set 10 min cooldown (was 5 min - prevent whipsaw)
        del self.positions[pair_id]
        self.close_cooldown[pair_id] = time.time() + 600  # 10 min cooldown
        await self._save_positions()
        self.m_positions_closed.inc()
        self.m_positions_open.set(len(self.positions))
        self.m_total_pnl.inc(net_pnl)

        # Publish execution update
        await self.redis.publish("arb.execution", json.dumps({
            "type": "exit",
            **trade_record,
        }))

    def _calculate_leg_pnl(self, side: str, entry: float, exit_price: float, qty: float) -> float:
        if side == "LONG":
            return (exit_price - entry) * qty
        else:
            return (entry - exit_price) * qty

    def _calculate_quantity(self, symbol: str, size_usd: float, price: float) -> float:
        if price <= 0:
            return 0
        qty = size_usd / price
        # Round to symbol's step size
        info = self.symbol_info.get(symbol, {})
        step_size = info.get("step_size", 0.001)
        if step_size > 0:
            # Use proper floor rounding with precision to avoid truncating to 0
            import math
            precision = max(0, -int(math.log10(step_size))) if step_size < 1 else 0
            qty = math.floor(qty / step_size) * step_size
            qty = round(qty, precision + 2)  # Avoid floating point artifacts
        # Ensure minimum notional: Binance requires min ~$5 per leg
        if qty * price < settings.risk.min_leg_size_usd:
            logger.warning(f"{symbol}: qty={qty} too small (notional=${qty*price:.2f} < ${settings.risk.min_leg_size_usd})")
            return 0
        return qty

    async def _get_price(self, symbol: str) -> Optional[float]:
        data = await self.redis.get(f"q:price:{symbol}")
        if data:
            return json.loads(data)["price"]
        return None

    async def _get_volatility(self, symbol: str) -> float:
        """Estimate recent volatility from price ticks"""
        # Simple: use 0.1% of price as default volatility
        price = await self._get_price(symbol)
        return (price or 50000) * 0.001

    async def _persist_trade(self, pos, exit_a, exit_b, net_pnl, gross_pnl, commission, exit_reason, signal):
        """Write trade to PostgreSQL for permanent storage"""
        if not self.db_factory:
            return
        try:
            async with self.db_factory() as session:
                trade = ArbTrade(
                    pair_id=pos["pair_id"],
                    strategy=pos.get("strategy", "stat_arb"),
                    leg_a_symbol=pos["leg_a_symbol"],
                    leg_a_side=pos["leg_a_side"],
                    leg_a_entry_price=pos["leg_a_entry"],
                    leg_a_exit_price=exit_a,
                    leg_a_quantity=pos["leg_a_qty"],
                    leg_b_symbol=pos["leg_b_symbol"],
                    leg_b_side=pos["leg_b_side"],
                    leg_b_entry_price=pos["leg_b_entry"],
                    leg_b_exit_price=exit_b,
                    leg_b_quantity=pos["leg_b_qty"],
                    leverage=pos.get("leverage", 3),
                    entry_zscore=pos.get("entry_zscore", 0),
                    exit_zscore=signal.get("zscore", 0),
                    combined_pnl=net_pnl,
                    total_commission=commission,
                    exit_reason=exit_reason,
                    trading_mode="paper" if self.is_paper else "live",
                    entry_time=datetime.utcfromtimestamp(pos.get("entry_time", time.time())),
                    exit_time=datetime.utcnow(),
                )
                session.add(trade)
                await session.commit()
                logger.debug(f"Trade persisted to PostgreSQL: {pos['pair_id']}")
        except Exception as e:
            logger.error(f"DB persist failed: {e}")

    async def _save_account_snapshot(self):
        """Save account balance snapshot to PostgreSQL"""
        if not self.db_factory:
            return
        try:
            balance_data = await self.redis.get("q:account:balance")
            balance = json.loads(balance_data) if balance_data else {}

            async with self.db_factory() as session:
                snapshot = AccountSnapshot(
                    total_balance=balance.get("total_balance", self.initial_equity),
                    available_balance=balance.get("available_balance", self.initial_equity),
                    unrealized_pnl=balance.get("unrealized_pnl", 0),
                    realized_pnl_today=float(await self.redis.get("q:risk:daily_pnl") or "0"),
                    open_positions=len(self.positions),
                    total_leverage=len(self.positions) * settings.risk.default_leverage,
                )
                session.add(snapshot)
                await session.commit()
        except Exception as e:
            logger.error(f"Snapshot save failed: {e}")

    async def _load_exchange_info(self):
        """Load symbol trading rules - Redis cache first, then API fallback"""
        # 1. Try Redis cache (no Binance REST call needed)
        try:
            cached = await self.redis.get("q:exchange_info_cache")
            if cached:
                self.symbol_info = json.loads(cached)
                logger.info(f"Exchange info loaded from Redis cache: {len(self.symbol_info)} symbols")
                return
        except Exception:
            pass

        # 2. In paper mode don't make API calls - data_collector will cache
        if self.is_paper:
            logger.warning("Exchange info cache not found, waiting for data_collector to populate...")
            # Retry cache 5 times, 10s intervals
            for i in range(5):
                await asyncio.sleep(10)
                try:
                    cached = await self.redis.get("q:exchange_info_cache")
                    if cached:
                        self.symbol_info = json.loads(cached)
                        logger.info(f"Exchange info loaded from Redis cache (retry {i+1}): {len(self.symbol_info)} symbols")
                        return
                except Exception:
                    pass
            logger.warning("Exchange info cache still empty - will use defaults")
            return

        # 3. In live mode fetch from API and cache
        try:
            info = await self.rest_client.get_exchange_info()
            for s in info.get("symbols", []):
                symbol = s["symbol"]
                for f in s.get("filters", []):
                    if f["filterType"] == "LOT_SIZE":
                        self.symbol_info[symbol] = {
                            "step_size": float(f["stepSize"]),
                            "min_qty": float(f["minQty"]),
                        }
                    elif f["filterType"] == "PRICE_FILTER":
                        self.symbol_info.setdefault(symbol, {})
                        self.symbol_info[symbol]["tick_size"] = float(f["tickSize"])
            logger.info(f"Loaded exchange info from API: {len(self.symbol_info)} symbols")
            # Cache to Redis (24h TTL)
            await self.redis.set("q:exchange_info_cache", json.dumps(self.symbol_info), ex=86400)
        except Exception as e:
            logger.error(f"Failed to load exchange info: {e}")

    async def _load_positions(self):
        data = await self.redis.get("q:active_positions")
        if data:
            self.positions = json.loads(data)
            logger.info(f"Loaded {len(self.positions)} active positions")

    async def _save_positions(self):
        await self.redis.set("q:active_positions", json.dumps(self.positions))

    async def _config_reload_loop(self):
        """Reload config overrides every 30 seconds"""
        while self.running:
            try:
                await apply_overrides(self.redis)
            except Exception:
                pass
            await asyncio.sleep(30)

    async def _health_server(self):
        app = web.Application()
        app.router.add_get("/health", self._health_handler)
        app.router.add_get("/metrics", self._metrics_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", 9104)
        await site.start()

    async def _health_handler(self, request):
        return web.json_response({
            "service": "executor",
            "status": "healthy",
            "mode": "paper" if self.is_paper else "live",
            "positions": len(self.positions),
            "orders_executed": self.orders_executed,
        })

    async def _metrics_handler(self, request):
        return web.Response(text=self.metrics.render(), content_type="text/plain")

    async def stop(self):
        self.running = False
        await self.rest_client.close()
        logger.info("Executor stopped")


async def main():
    executor = Executor()
    try:
        await executor.start()
    except KeyboardInterrupt:
        await executor.stop()


if __name__ == "__main__":
    asyncio.run(main())
