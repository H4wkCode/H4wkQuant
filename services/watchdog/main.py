"""
H4wkQuant - Watchdog Service (:9105)
System health monitoring, kill switch, heartbeat, Telegram alerts.
"""
import asyncio
import json
import time
from typing import Dict
import redis.asyncio as redis
import aiohttp
from aiohttp import web
from loguru import logger

from shared.config.settings import settings, setup_service_logging
from shared.utils.redis_helper import get_redis_client
from shared.utils.metrics import MetricsRegistry
from shared.clients.telegram_client import get_telegram_notifier


import os

# In Docker: use container names. Locally: use localhost.
_host = "localhost"
if os.environ.get("SERVICE_NAME"):
    # Running inside Docker - use compose service names
    SERVICE_ENDPOINTS = {
        "data_collector": "http://data_collector:9101/health",
        "spread_engine": "http://spread_engine:9102/health",
        "risk_manager": "http://risk_manager:9103/health",
        "executor": "http://executor:9104/health",
    }
else:
    SERVICE_ENDPOINTS = {
        "data_collector": "http://localhost:9101/health",
        "spread_engine": "http://localhost:9102/health",
        "risk_manager": "http://localhost:9103/health",
        "executor": "http://localhost:9104/health",
    }


class Watchdog:
    def __init__(self):
        self.redis: redis.Redis = None
        self.telegram = get_telegram_notifier()
        self.running = False
        self.service_status: Dict[str, dict] = {}
        self.started_at = time.time()

        # Metrics
        self.metrics = MetricsRegistry("watchdog")
        self.m_health_ok = self.metrics.counter("h4wkquant_health_checks_ok", "Successful health checks")
        self.m_health_fail = self.metrics.counter("h4wkquant_health_checks_fail", "Failed health checks")

    async def start(self):
        setup_service_logging("watchdog")
        self.redis = await get_redis_client(settings.database.redis_url)
        self.running = True

        logger.info("Watchdog starting...")

        # Send startup notification
        await self.telegram.notify_system(
            f"H4wkQuant starting\n"
            f"Mode: {settings.trading_mode.value}\n"
            f"Pairs: {', '.join(settings.arbitrage.pairs)}\n"
            f"Kelly: {settings.arbitrage.kelly_fraction}"
        )

        # Start monitoring tasks
        asyncio.create_task(self._heartbeat_loop())
        asyncio.create_task(self._subscribe_executions())
        asyncio.create_task(self._daily_report_loop())
        asyncio.create_task(self._health_server())
        asyncio.create_task(self._telegram_config_loop())

        logger.info("Watchdog started")

        while self.running:
            await asyncio.sleep(1)

    async def _heartbeat_loop(self):
        """Check all services every 30 seconds"""
        while self.running:
            try:
                async with aiohttp.ClientSession() as session:
                    for service, url in SERVICE_ENDPOINTS.items():
                        try:
                            async with session.get(
                                url,
                                timeout=aiohttp.ClientTimeout(total=5)
                            ) as resp:
                                data = await resp.json()
                                was_down = self.service_status.get(service, {}).get("status") == "down"
                                self.service_status[service] = {
                                    "status": "healthy",
                                    "data": data,
                                    "last_check": time.time(),
                                }
                                self.m_health_ok.inc()
                                if was_down:
                                    await self.telegram.notify_system(f"{service} recovered")
                        except Exception:
                            self.m_health_fail.inc()
                            was_up = self.service_status.get(service, {}).get("status") == "healthy"
                            self.service_status[service] = {
                                "status": "down",
                                "last_check": time.time(),
                            }
                            if was_up:
                                await self.telegram.notify_system(f"ALERT: {service} is DOWN!")
                                logger.error(f"{service} is DOWN!")

                # Write status to Redis
                await self.redis.setex(
                    "q:watchdog:status", 60,
                    json.dumps(self.service_status, default=str)
                )

            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

            await asyncio.sleep(30)

    async def _subscribe_executions(self):
        """Forward execution events to Telegram"""
        pubsub = self.redis.pubsub()
        await pubsub.subscribe("arb.execution")

        async for message in pubsub.listen():
            if not self.running:
                break
            if message["type"] != "message":
                continue

            try:
                data = json.loads(message["data"])
                if data.get("type") == "entry":
                    await self.telegram.notify_arb_entry(
                        data["pair_id"],
                        data.get("strategy", "unknown"),
                        data.get("entry_zscore", 0),
                        0,  # edge
                        data.get("leg_a_qty", 0) * data.get("leg_a_entry", 0),
                    )
                elif data.get("type") == "exit":
                    await self.telegram.notify_arb_exit(
                        data["pair_id"],
                        data.get("combined_pnl", 0),
                        data.get("exit_reason", "unknown"),
                    )
            except Exception as e:
                logger.error(f"Execution notification error: {e}")

    async def _daily_report_loop(self):
        """Send daily summary at 00:00 UTC"""
        while self.running:
            try:
                # Wait until next midnight UTC
                now = time.time()
                next_midnight = (int(now / 86400) + 1) * 86400
                wait_seconds = next_midnight - now
                await asyncio.sleep(min(wait_seconds, 3600))  # Check hourly

                if wait_seconds <= 60:
                    # Generate and send report
                    daily_pnl = float(await self.redis.get("q:risk:daily_pnl") or "0")
                    trades_data = await self.redis.lrange("q:trades:history", 0, -1)
                    today_trades = 0
                    for t in trades_data:
                        td = json.loads(t)
                        if td.get("exit_time", 0) > now - 86400:
                            today_trades += 1

                    uptime = (time.time() - self.started_at) / 3600

                    await self.telegram.notify_system(
                        f"Daily Report\n"
                        f"PnL: ${daily_pnl:.4f}\n"
                        f"Trades: {today_trades}\n"
                        f"Uptime: {uptime:.1f}h\n"
                        f"Services: {sum(1 for s in self.service_status.values() if s.get('status') == 'healthy')}/{len(SERVICE_ENDPOINTS)}"
                    )

                    # Reset daily PnL
                    await self.redis.set("q:risk:daily_pnl", "0")

            except Exception as e:
                logger.error(f"Daily report error: {e}")

    async def _telegram_config_loop(self):
        """Check for Telegram config changes every 60 seconds"""
        last_config = None
        while self.running:
            try:
                tg_raw = await self.redis.get("q:config:telegram")
                if tg_raw:
                    config = json.loads(tg_raw)
                    config_str = json.dumps(config, sort_keys=True)
                    if config_str != last_config:
                        self.telegram.update_config(
                            bot_token=config.get("bot_token"),
                            chat_id=config.get("chat_id"),
                            enabled=config.get("enabled", False),
                        )
                        last_config = config_str
                        logger.info(f"Telegram config updated: enabled={config.get('enabled')}")
            except Exception as e:
                logger.error(f"Telegram config reload error: {e}")
            await asyncio.sleep(60)

    async def _health_server(self):
        app = web.Application()
        app.router.add_get("/health", self._health_handler)
        app.router.add_get("/metrics", self._metrics_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", 9105)
        await site.start()

    async def _health_handler(self, request):
        return web.json_response({
            "service": "watchdog",
            "status": "healthy",
            "uptime_hours": round((time.time() - self.started_at) / 3600, 2),
            "services": self.service_status,
        })

    async def _metrics_handler(self, request):
        return web.Response(text=self.metrics.render(), content_type="text/plain")

    async def stop(self):
        self.running = False
        await self.telegram.notify_system("H4wkQuant shutting down")
        await self.telegram.close()
        logger.info("Watchdog stopped")


async def main():
    watchdog = Watchdog()
    try:
        await watchdog.start()
    except KeyboardInterrupt:
        await watchdog.stop()


if __name__ == "__main__":
    asyncio.run(main())
