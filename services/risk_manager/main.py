"""
H4wkQuant - Risk Manager Service (:9103)
Validates arb signals against risk limits before execution.
Exposure tracking, drawdown limits, kill switch.
"""
import asyncio
import json
import time
from typing import Dict, List
import redis.asyncio as redis
from aiohttp import web
from loguru import logger

from shared.config.settings import settings, setup_service_logging, apply_overrides
from shared.utils.redis_helper import get_redis_client, resilient_subscribe
from shared.utils.metrics import MetricsRegistry
from shared.schemas.models import ArbSignal, ArbSignalAction, RiskAssessment


class RiskManager:
    def __init__(self):
        self.redis: redis.Redis = None
        self.running = False
        self.kill_switch_active = False

        # Tracking
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.open_positions: Dict[str, dict] = {}
        self.total_exposure_usd = 0.0

        # Metrics
        self.metrics = MetricsRegistry("risk_manager")
        self.m_approved = self.metrics.counter("h4wkquant_signals_approved", "Signals approved")
        self.m_rejected = self.metrics.counter("h4wkquant_signals_rejected", "Signals rejected")

        # Stats
        self.signals_received = 0
        self.signals_approved = 0
        self.signals_rejected = 0

    async def start(self):
        setup_service_logging("risk_manager")
        self.redis = await get_redis_client(settings.database.redis_url)
        self.running = True

        logger.info("Risk Manager starting...")

        # Load state
        await self._load_state()

        # Apply config overrides
        await apply_overrides(self.redis)

        # Subscribe to signals
        asyncio.create_task(self._subscribe_signals())
        asyncio.create_task(self._monitor_drawdown())
        asyncio.create_task(self._health_server())
        asyncio.create_task(self._config_reload_loop())

        logger.info("Risk Manager started")

        while self.running:
            await asyncio.sleep(1)

    async def _subscribe_signals(self):
        """Listen for arb signals and validate (with auto-reconnect)"""
        async def handle_message(message):
            try:
                signal_data = json.loads(message["data"])
                await self._process_signal(signal_data)
            except Exception as e:
                logger.error(f"Signal processing error: {e}")

        await resilient_subscribe(self.redis, "arb.signal", handle_message, "risk_manager")

    async def _process_signal(self, signal_data: Dict):
        """Validate signal against risk limits"""
        self.signals_received += 1

        pair_id = signal_data["pair_id"]
        action = signal_data["action"]

        # Close signals bypass most checks
        if action == "close":
            await self._approve_signal(signal_data, "close_signal")
            return

        # Kill switch check
        if self.kill_switch_active:
            await self._reject_signal(signal_data, "kill_switch_active")
            return

        # Check kill switch from Redis
        ks = await self.redis.get("q:control:kill_switch")
        if ks and json.loads(ks).get("active"):
            self.kill_switch_active = True
            await self._reject_signal(signal_data, "kill_switch_active")
            return

        warnings = []

        # Reload positions from Redis for accurate count (Bug #4 fix)
        pos_raw = await self.redis.get("q:active_positions")
        self.open_positions = json.loads(pos_raw) if pos_raw else {}

        # 1. Position count limit
        if len(self.open_positions) >= settings.risk.max_open_positions:
            await self._reject_signal(signal_data, f"max_positions ({settings.risk.max_open_positions})")
            return

        # 2. Duplicate pair check
        if pair_id in self.open_positions:
            await self._reject_signal(signal_data, f"already_in_position: {pair_id}")
            return

        # 3. Daily loss limit
        equity = await self._get_equity()
        max_daily_loss = equity * settings.risk.max_daily_loss_percent / 100
        if self.daily_pnl < -max_daily_loss:
            await self._reject_signal(signal_data, f"daily_loss_limit (${self.daily_pnl:.2f})")
            return

        # 4. Weekly loss limit
        max_weekly_loss = equity * settings.risk.max_weekly_loss_percent / 100
        if self.weekly_pnl < -max_weekly_loss:
            await self._reject_signal(signal_data, f"weekly_loss_limit (${self.weekly_pnl:.2f})")
            return

        # 5. Position size check
        position_size = signal_data.get("position_size_usd", 0)
        if position_size < settings.risk.min_leg_size_usd:
            await self._reject_signal(signal_data, f"below_min_size (${position_size:.2f})")
            return

        if position_size > settings.risk.max_leg_size_usd:
            signal_data["position_size_usd"] = settings.risk.max_leg_size_usd
            warnings.append(f"size_capped: ${settings.risk.max_leg_size_usd}")

        # 6. Total exposure check
        new_exposure = self.total_exposure_usd + position_size * 2  # 2 legs
        max_exposure = equity * settings.risk.max_total_leverage
        if new_exposure > max_exposure:
            await self._reject_signal(signal_data, f"max_exposure (${new_exposure:.2f} > ${max_exposure:.2f})")
            return

        # 7. Drawdown check
        dd_pct = await self._get_drawdown()
        if dd_pct > settings.risk.max_drawdown_percent:
            self.kill_switch_active = True
            await self.redis.set("q:control:kill_switch", json.dumps({
                "active": True, "reason": f"drawdown_{dd_pct:.1f}%", "time": time.time()
            }))
            await self._reject_signal(signal_data, f"drawdown_kill_switch ({dd_pct:.1f}%)")
            return

        # 8. Edge check - must be positive
        if signal_data.get("edge_net", 0) <= 0:
            await self._reject_signal(signal_data, "negative_edge")
            return

        # All checks passed
        if warnings:
            signal_data["warnings"] = warnings

        await self._approve_signal(signal_data, "all_checks_passed")

    async def _approve_signal(self, signal_data: Dict, reason: str):
        """Forward approved signal to executor"""
        self.signals_approved += 1
        self.m_approved.inc()
        signal_data["risk_approved"] = True
        signal_data["risk_reason"] = reason
        signal_data["risk_time"] = time.time()

        await self.redis.publish("arb.approved", json.dumps(signal_data))
        logger.info(f"APPROVED: {signal_data['pair_id']} ({reason})")

    async def _reject_signal(self, signal_data: Dict, reason: str):
        """Log rejected signal"""
        self.signals_rejected += 1
        self.m_rejected.inc()
        logger.info(f"REJECTED: {signal_data['pair_id']} ({reason})")

        # Store rejection for analytics
        rejection = {
            "pair_id": signal_data["pair_id"],
            "reason": reason,
            "time": time.time(),
        }
        await self.redis.lpush("q:risk:rejections", json.dumps(rejection))
        await self.redis.ltrim("q:risk:rejections", 0, 199)

    async def _monitor_drawdown(self):
        """Monitor equity and update drawdown tracking"""
        while self.running:
            try:
                equity = await self._get_equity()

                # Update peak
                peak = await self.redis.get("q:risk:equity_peak")
                peak = float(peak) if peak else equity
                if equity > peak:
                    peak = equity
                    await self.redis.set("q:risk:equity_peak", str(peak))

                # Calculate drawdown
                dd_pct = (peak - equity) / peak * 100 if peak > 0 else 0
                await self.redis.setex("q:risk:drawdown_pct", 60, str(dd_pct))

                # Update daily PnL from Redis
                daily_pnl = await self.redis.get("q:risk:daily_pnl")
                if daily_pnl:
                    self.daily_pnl = float(daily_pnl)

            except Exception as e:
                logger.error(f"Drawdown monitor error: {e}")

            await asyncio.sleep(10)

    async def _get_equity(self) -> float:
        data = await self.redis.get("q:account:balance")
        if data:
            return json.loads(data).get("total_balance", settings.risk.paper_initial_balance)
        return settings.risk.paper_initial_balance

    async def _get_drawdown(self) -> float:
        dd = await self.redis.get("q:risk:drawdown_pct")
        return float(dd) if dd else 0.0

    async def _load_state(self):
        """Load positions and PnL from Redis"""
        pos = await self.redis.get("q:active_positions")
        if pos:
            self.open_positions = json.loads(pos)

        dpnl = await self.redis.get("q:risk:daily_pnl")
        if dpnl:
            self.daily_pnl = float(dpnl)

        ks = await self.redis.get("q:control:kill_switch")
        if ks:
            self.kill_switch_active = json.loads(ks).get("active", False)

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
        site = web.TCPSite(runner, "0.0.0.0", 9103)
        await site.start()

    async def _health_handler(self, request):
        return web.json_response({
            "service": "risk_manager",
            "status": "healthy",
            "kill_switch": self.kill_switch_active,
            "signals_received": self.signals_received,
            "signals_approved": self.signals_approved,
            "signals_rejected": self.signals_rejected,
            "open_positions": len(self.open_positions),
            "daily_pnl": round(self.daily_pnl, 4),
        })

    async def _metrics_handler(self, request):
        return web.Response(text=self.metrics.render(), content_type="text/plain")

    async def stop(self):
        self.running = False
        logger.info("Risk Manager stopped")


async def main():
    manager = RiskManager()
    try:
        await manager.start()
    except KeyboardInterrupt:
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())
