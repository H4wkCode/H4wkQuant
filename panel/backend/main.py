"""
H4wkQuant - Panel Backend (:8180)
FastAPI dashboard for monitoring and controlling the arbitrage system.
"""
import asyncio
import json
import time
import os
import subprocess
import hashlib
import base64
from typing import Dict, List, Optional
from datetime import datetime

import redis.asyncio as redis
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Depends, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, PlainTextResponse
from contextlib import asynccontextmanager
from loguru import logger
from pydantic import BaseModel

from shared.config.settings import settings, setup_service_logging
from shared.utils.redis_helper import get_redis_client
from models.montecarlo import MonteCarloModel
from panel.backend.auth import (
    hash_password, verify_password, create_jwt, decode_jwt,
    ensure_default_user, get_current_user
)
from shared.utils.metrics import MetricsRegistry


# ============================================================
# App Lifecycle
# ============================================================

redis_client: redis.Redis = None
ws_clients: List[WebSocket] = []

# Metrics
metrics = MetricsRegistry("panel")
m_ws_connections = metrics.gauge("h4wkquant_panel_ws_connections", "Active WebSocket connections")
m_api_requests = metrics.counter("h4wkquant_panel_api_requests", "Total API requests")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client
    setup_service_logging("panel")
    redis_client = await get_redis_client(settings.database.redis_url)
    logger.info("Panel backend started on :8180")

    # Ensure default admin user exists
    await ensure_default_user(redis_client)

    # Start background tasks
    asyncio.create_task(broadcast_loop())
    asyncio.create_task(subscribe_events())
    asyncio.create_task(collect_metrics_loop())

    # Auto scan restore
    try:
        raw = await redis_client.get("q:config:autoscan")
        if raw:
            import json as _json
            cfg = _json.loads(raw)
            if cfg.get("enabled"):
                global _auto_scan_task
                interval = cfg.get("interval_minutes", 30)
                _auto_scan_task = asyncio.create_task(_autoscan_loop(interval))
                logger.info(f"Auto scan restored: every {interval} minutes")
    except Exception as e:
        logger.debug(f"Auto scan restore error: {e}")

    yield

    logger.info("Panel backend shutting down")


app = FastAPI(title="H4wkQuant Panel", version="2.0.0", lifespan=lifespan)

# Serve frontend
frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")


# ============================================================
# WebSocket (auth via first message)
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()

    # Wait for auth token as first message
    try:
        first_msg = await asyncio.wait_for(ws.receive_text(), timeout=10)
        try:
            msg = json.loads(first_msg)
            token = msg.get("token", "")
        except json.JSONDecodeError:
            token = first_msg

        payload = decode_jwt(token)
        if not payload:
            await ws.send_text(json.dumps({"event": "auth_error", "data": {"message": "Invalid token"}}))
            await ws.close()
            return
    except asyncio.TimeoutError:
        # Allow unauthenticated connections for backwards compatibility
        pass

    ws_clients.append(ws)
    m_ws_connections.set(len(ws_clients))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        ws_clients.remove(ws)
        m_ws_connections.set(len(ws_clients))


async def broadcast(event: str, data: dict):
    message = json.dumps({"event": event, "data": data})
    dead = []
    for ws in ws_clients:
        try:
            await ws.send_text(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        ws_clients.remove(ws)
    m_ws_connections.set(len(ws_clients))


async def broadcast_loop():
    """Push live data to frontend every 2 seconds"""
    while True:
        try:
            data = await get_dashboard_data()
            await broadcast("dashboard", data)
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
        await asyncio.sleep(2)


async def subscribe_events():
    """Subscribe to Redis events and forward to WebSocket"""
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("arb.signal", "arb.execution", "arb.approved")

    async for message in pubsub.listen():
        if message["type"] != "message":
            continue
        try:
            channel = message["channel"]
            data = json.loads(message["data"])
            await broadcast(channel, data)
        except Exception:
            pass


async def collect_metrics_loop():
    """Collect metrics from all services every 30 seconds, store in Redis list"""
    import aiohttp
    SERVICE_METRICS = {
        "data_collector": "http://data_collector:9101/metrics",
        "spread_engine": "http://spread_engine:9102/metrics",
        "risk_manager": "http://risk_manager:9103/metrics",
        "executor": "http://executor:9104/metrics",
        "watchdog": "http://watchdog:9105/metrics",
    }
    # Detect if running locally
    if not os.environ.get("SERVICE_NAME"):
        SERVICE_METRICS = {k: v.replace("data_collector:", "localhost:").replace("spread_engine:", "localhost:").replace("risk_manager:", "localhost:").replace("executor:", "localhost:").replace("watchdog:", "localhost:") for k, v in SERVICE_METRICS.items()}

    while True:
        try:
            snapshot = {"ts": int(time.time())}
            async with aiohttp.ClientSession() as session:
                for svc, url in SERVICE_METRICS.items():
                    try:
                        async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                            text = await resp.text()
                            # Parse prometheus text format
                            for line in text.strip().split("\n"):
                                if line.startswith("#") or not line.strip():
                                    continue
                                parts = line.split(" ", 1)
                                if len(parts) == 2:
                                    snapshot[f"{svc}.{parts[0]}"] = float(parts[1])
                    except Exception:
                        pass

            await redis_client.lpush("q:metrics:history", json.dumps(snapshot))
            await redis_client.ltrim("q:metrics:history", 0, 2879)  # 24 hours * 2/min = 2880
        except Exception as e:
            logger.error(f"Metrics collect error: {e}")
        await asyncio.sleep(30)


# ============================================================
# Dashboard Data
# ============================================================

async def get_dashboard_data() -> dict:
    """Aggregate all dashboard data from Redis"""
    # Account balance
    balance_data = await redis_client.get("q:account:balance")
    balance = json.loads(balance_data) if balance_data else {"total_balance": settings.risk.paper_initial_balance, "available_balance": settings.risk.paper_initial_balance}

    # Active positions with live PnL
    positions_data = await redis_client.get("q:active_positions")
    positions = json.loads(positions_data) if positions_data else {}

    for pair_id, pos in positions.items():
        sym_a = pos.get("leg_a_symbol", "")
        sym_b = pos.get("leg_b_symbol", "")
        price_a_raw = await redis_client.get(f"q:price:{sym_a}")
        price_b_raw = await redis_client.get(f"q:price:{sym_b}")
        if price_a_raw and price_b_raw:
            cur_a = json.loads(price_a_raw).get("price", 0)
            cur_b = json.loads(price_b_raw).get("price", 0)
            pnl_a = (cur_a - pos["leg_a_entry"]) * pos["leg_a_qty"] if pos["leg_a_side"] == "LONG" else (pos["leg_a_entry"] - cur_a) * pos["leg_a_qty"]
            pnl_b = (cur_b - pos["leg_b_entry"]) * pos["leg_b_qty"] if pos["leg_b_side"] == "LONG" else (pos["leg_b_entry"] - cur_b) * pos["leg_b_qty"]
            pos["live_pnl"] = round(pnl_a + pnl_b, 6)
            pos["cur_price_a"] = cur_a
            pos["cur_price_b"] = cur_b

    # Daily PnL
    daily_pnl = float(await redis_client.get("q:risk:daily_pnl") or "0")

    # Drawdown
    dd = float(await redis_client.get("q:risk:drawdown_pct") or "0")

    # Kill switch
    ks_data = await redis_client.get("q:control:kill_switch")
    kill_switch = json.loads(ks_data) if ks_data else {"active": False}

    # Service health
    watchdog_data = await redis_client.get("q:watchdog:status")
    services = json.loads(watchdog_data) if watchdog_data else {}

    # Recent trades
    trades_raw = await redis_client.lrange("q:trades:history", 0, 19)
    trades = [json.loads(t) for t in trades_raw]

    # Recent signals
    signals_raw = await redis_client.lrange("q:signals:history", 0, 19)
    signals = [json.loads(s) for s in signals_raw]

    # Prices - all symbols from active pairs
    all_symbols = set(["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"])
    screened_raw_for_prices = await redis_client.get("q:config:screened_pairs")
    if screened_raw_for_prices:
        for pp in json.loads(screened_raw_for_prices):
            a, b = pp.split("/")
            all_symbols.add(a)
            all_symbols.add(b)

    prices = {}
    for symbol in all_symbols:
        p = await redis_client.get(f"q:price:{symbol}")
        if p:
            prices[symbol] = json.loads(p).get("price", 0)

    # Get all pairs: config (static) + screened (dynamic)
    all_pairs = list(settings.arbitrage.pairs)
    screened_raw = await redis_client.get("q:config:screened_pairs")
    screened_pairs = json.loads(screened_raw) if screened_raw else []
    for p in screened_pairs:
        if p not in all_pairs:
            all_pairs.append(p)

    # Spread data for each pair (with z-score from spread engine)
    spreads = {}
    for pair in all_pairs:
        sym_a, sym_b = pair.split("/")
        if sym_a in prices and sym_b in prices and prices[sym_b] > 0:
            spread_data = {"ratio": round(prices[sym_a] / prices[sym_b], 4), "price_a": prices[sym_a], "price_b": prices[sym_b]}
            zscore_data = await redis_client.get(f"q:spread:{pair}")
            if zscore_data:
                zd = json.loads(zscore_data)
                spread_data["zscore"] = zd.get("zscore", 0)
                spread_data["half_life"] = zd.get("half_life", 0)
            else:
                spread_data["zscore"] = 0
            spread_data["is_default"] = pair in settings.arbitrage.pairs
            spreads[pair] = spread_data

    # Funding rates
    funding = {}
    for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]:
        fr = await redis_client.get(f"q:funding:{symbol}")
        if fr:
            funding[symbol] = json.loads(fr).get("funding_rate", 0)

    # Rejections (last 20)
    rejections_raw = await redis_client.lrange("q:risk:rejections", 0, 19)
    rejections = [json.loads(r) for r in rejections_raw]

    # Spread engine warmup info
    engine_start = await redis_client.get("q:spread_engine:start_time")
    engine_start_ts = float(engine_start) if engine_start else None

    # Regime detection
    regime_raw = await redis_client.get("q:regime")
    regime = json.loads(regime_raw) if regime_raw else {"regime": "NORMAL", "volatility": 0, "percentile": 50, "allow_new_positions": True, "warming_up": True}

    # Portfolio correlation matrix
    corr_raw = await redis_client.get("q:portfolio:correlation")
    correlation_matrix = json.loads(corr_raw) if corr_raw else {}

    return {
        "balance": balance,
        "positions": positions,
        "daily_pnl": round(daily_pnl, 4),
        "drawdown_pct": round(dd, 2),
        "kill_switch": kill_switch,
        "services": services,
        "trades": trades[:10],
        "signals": signals[:10],
        "prices": prices,
        "spreads": spreads,
        "funding": funding,
        "rejections": rejections,
        "trading_mode": settings.trading_mode.value,
        "engine_start_time": engine_start_ts,
        "regime": regime,
        "correlation_matrix": correlation_matrix,
        "timestamp": time.time(),
    }


# ============================================================
# Auth Endpoints (no auth required)
# ============================================================

class LoginRequest(BaseModel):
    username: str
    password: str

class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


@app.get("/")
async def root():
    return FileResponse(os.path.join(frontend_dir, "index.html"))


@app.post("/api/v1/auth/login")
async def api_login(req: LoginRequest):
    stored = await redis_client.hget("q:panel:users", req.username)
    if not stored:
        raise HTTPException(401, "Invalid username or password")
    if not verify_password(req.password, stored):
        raise HTTPException(401, "Invalid username or password")
    token = create_jwt({"sub": req.username})
    return {"token": token, "username": req.username}


@app.post("/api/v1/auth/change-password")
async def api_change_password(req: ChangePasswordRequest, user: dict = Depends(get_current_user)):
    username = user["sub"]
    stored = await redis_client.hget("q:panel:users", username)
    if not stored or not verify_password(req.old_password, stored):
        raise HTTPException(400, "Current password is incorrect")
    if len(req.new_password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    new_hash = hash_password(req.new_password)
    await redis_client.hset("q:panel:users", username, new_hash)
    return {"status": "ok", "message": "Password changed"}


# ============================================================
# Protected REST API
# ============================================================

@app.get("/api/v1/dashboard")
async def api_dashboard(user: dict = Depends(get_current_user)):
    m_api_requests.inc()
    return await get_dashboard_data()


@app.get("/api/v1/trades")
async def api_trades(limit: int = Query(50, le=200), user: dict = Depends(get_current_user)):
    trades_raw = await redis_client.lrange("q:trades:history", 0, limit - 1)
    return [json.loads(t) for t in trades_raw]


@app.get("/api/v1/signals")
async def api_signals(limit: int = Query(50, le=200), user: dict = Depends(get_current_user)):
    signals_raw = await redis_client.lrange("q:signals:history", 0, limit - 1)
    return [json.loads(s) for s in signals_raw]


@app.get("/api/v1/positions")
async def api_positions(user: dict = Depends(get_current_user)):
    data = await redis_client.get("q:active_positions")
    return json.loads(data) if data else {}


@app.get("/api/v1/risk")
async def api_risk(user: dict = Depends(get_current_user)):
    dd = float(await redis_client.get("q:risk:drawdown_pct") or "0")
    daily_pnl = float(await redis_client.get("q:risk:daily_pnl") or "0")
    peak = float(await redis_client.get("q:risk:equity_peak") or str(settings.risk.paper_initial_balance))
    ks_data = await redis_client.get("q:control:kill_switch")
    kill_switch = json.loads(ks_data) if ks_data else {"active": False}

    rejections_raw = await redis_client.lrange("q:risk:rejections", 0, 19)
    rejections = [json.loads(r) for r in rejections_raw]

    return {
        "drawdown_pct": round(dd, 2),
        "daily_pnl": round(daily_pnl, 4),
        "equity_peak": round(peak, 2),
        "kill_switch": kill_switch,
        "rejections": rejections,
        "limits": {
            "max_leverage": settings.risk.max_total_leverage,
            "max_positions": settings.risk.max_open_positions,
            "max_daily_loss_pct": settings.risk.max_daily_loss_percent,
            "max_drawdown_pct": settings.risk.max_drawdown_percent,
            "max_leg_size_usd": settings.risk.max_leg_size_usd,
        },
    }


@app.get("/api/v1/config")
async def api_config(user: dict = Depends(get_current_user)):
    return {
        "trading_mode": settings.trading_mode.value,
        "pairs": settings.arbitrage.pairs,
        "zscore_entry": settings.arbitrage.zscore_entry_threshold,
        "zscore_exit": settings.arbitrage.zscore_exit_threshold,
        "kelly_fraction": settings.arbitrage.kelly_fraction,
        "mc_simulations": settings.arbitrage.mc_simulations,
        "stoikov_gamma": settings.arbitrage.stoikov_gamma,
        "risk": {
            "max_leverage": settings.risk.max_total_leverage,
            "max_positions": settings.risk.max_open_positions,
            "max_daily_loss_pct": settings.risk.max_daily_loss_percent,
            "maker_fee": settings.risk.maker_fee,
            "taker_fee": settings.risk.taker_fee,
            "default_leverage": settings.risk.default_leverage,
            "paper_initial_balance": settings.risk.paper_initial_balance,
            "max_leg_size_usd": settings.risk.max_leg_size_usd,
        },
    }


# ============================================================
# Settings Endpoints (3.1 + 3.3 + 3.4)
# ============================================================

CONFIGURABLE_PARAMS = {
    "zscore_entry": {"type": "float", "min": 0.5, "max": 5.0},
    "zscore_exit": {"type": "float", "min": 0.1, "max": 2.0},
    "half_life_max": {"type": "int", "min": 10, "max": 500},
    "max_open_positions": {"type": "int", "min": 2, "max": 20},
    "default_leverage": {"type": "int", "min": 1, "max": 10},
    "kelly_fraction": {"type": "float", "min": 0.05, "max": 1.0},
    "paper_initial_balance": {"type": "float", "min": 50.0, "max": 10000.0},
    "max_leg_size_usd": {"type": "float", "min": 5.0, "max": 5000.0},
    "kalman_enabled": {"type": "boolean"},
    "regime_detection_enabled": {"type": "boolean"},
    "funding_arb_enabled": {"type": "boolean"},
    "momentum_enabled": {"type": "boolean"},
    "multi_tf_enabled": {"type": "boolean"},
    "portfolio_corr_enabled": {"type": "boolean"},
    "warmup_seconds": {"type": "int", "min": 0, "max": 28800},
}

# Type converters for validation
_TYPE_MAP = {"float": float, "int": int, "boolean": bool}


@app.get("/api/v1/settings")
async def api_get_settings(user: dict = Depends(get_current_user)):
    """Get current config + Redis overrides"""
    overrides_raw = await redis_client.get("q:config:overrides")
    overrides = json.loads(overrides_raw) if overrides_raw else {}

    current = {
        "zscore_entry": settings.arbitrage.zscore_entry_threshold,
        "zscore_exit": settings.arbitrage.zscore_exit_threshold,
        "half_life_max": settings.arbitrage.half_life_max,
        "max_open_positions": settings.risk.max_open_positions,
        "default_leverage": settings.risk.default_leverage,
        "kelly_fraction": settings.arbitrage.kelly_fraction,
        "paper_initial_balance": settings.risk.paper_initial_balance,
        "max_leg_size_usd": settings.risk.max_leg_size_usd,
        "kalman_enabled": settings.arbitrage.kalman_enabled,
        "regime_detection_enabled": settings.arbitrage.regime_detection_enabled,
        "funding_arb_enabled": settings.arbitrage.funding_arb_enabled,
        "momentum_enabled": settings.arbitrage.momentum_enabled,
        "multi_tf_enabled": settings.arbitrage.multi_tf_enabled,
        "portfolio_corr_enabled": settings.arbitrage.portfolio_corr_enabled,
        "warmup_seconds": 7200,
    }

    # Apply overrides on top
    for key, value in overrides.items():
        if key in current:
            current[key] = value

    return {"config": current, "overrides": overrides, "params": CONFIGURABLE_PARAMS}


@app.post("/api/v1/settings")
async def api_save_settings(request: Request, user: dict = Depends(get_current_user)):
    """Save config overrides to Redis"""
    body = await request.json()
    validated = {}

    for key, value in body.items():
        if key not in CONFIGURABLE_PARAMS:
            continue
        spec = CONFIGURABLE_PARAMS[key]
        converter = _TYPE_MAP.get(spec["type"], str)
        try:
            typed_val = converter(value)
            if spec["type"] in ("int", "float"):
                if typed_val < spec.get("min", float("-inf")) or typed_val > spec.get("max", float("inf")):
                    raise HTTPException(400, f"{key}: value must be between {spec['min']}-{spec['max']}")
            validated[key] = typed_val
        except (ValueError, TypeError):
            raise HTTPException(400, f"{key}: invalid value")

    if validated:
        await redis_client.set("q:config:overrides", json.dumps(validated))

    return {"status": "ok", "overrides": validated}


# API Key Management (3.3)
@app.get("/api/v1/settings/api-keys")
async def api_get_keys(user: dict = Depends(get_current_user)):
    """Get masked API keys"""
    keys_raw = await redis_client.get("q:config:api_keys")
    result = {"binance": {"has_key": False}, "bybit": {"has_key": False}}

    if keys_raw:
        try:
            from cryptography.fernet import Fernet
            fernet_key = base64.urlsafe_b64encode(hashlib.sha256(settings.panel_secret_key.encode()).digest())
            f = Fernet(fernet_key)
            keys = json.loads(f.decrypt(keys_raw.encode()))
            for exchange in ["binance", "bybit"]:
                ak = keys.get(f"{exchange}_api_key", "")
                if ak:
                    result[exchange] = {
                        "has_key": True,
                        "api_key_masked": ak[:4] + "****" + ak[-4:] if len(ak) > 8 else "****",
                    }
        except Exception as e:
            logger.error(f"API key decrypt failed: {e}")

    # Check env fallback
    if not result["binance"]["has_key"] and settings.binance.api_key:
        ak = settings.binance.api_key
        result["binance"] = {"has_key": True, "api_key_masked": ak[:4] + "****" + ak[-4:] if len(ak) > 8 else "****", "source": "env"}

    return result


@app.post("/api/v1/settings/api-keys")
async def api_save_keys(request: Request, user: dict = Depends(get_current_user)):
    """Save encrypted API keys"""
    body = await request.json()

    # Load existing keys
    keys_raw = await redis_client.get("q:config:api_keys")
    from cryptography.fernet import Fernet
    fernet_key = base64.urlsafe_b64encode(hashlib.sha256(settings.panel_secret_key.encode()).digest())
    f = Fernet(fernet_key)

    existing = {}
    if keys_raw:
        try:
            existing = json.loads(f.decrypt(keys_raw.encode()))
        except Exception:
            pass

    # Update only provided keys
    for key in ["binance_api_key", "binance_secret_key", "bybit_api_key", "bybit_secret_key"]:
        if key in body and body[key]:
            existing[key] = body[key]

    encrypted = f.encrypt(json.dumps(existing).encode()).decode()
    await redis_client.set("q:config:api_keys", encrypted)

    # Notify services
    await redis_client.publish("q:config:api_keys_updated", "1")

    return {"status": "ok", "message": "API keys saved"}


@app.delete("/api/v1/settings/api-keys")
async def api_delete_keys(exchange: str = "binance", user: dict = Depends(get_current_user)):
    """Delete API keys for an exchange"""
    keys_raw = await redis_client.get("q:config:api_keys")
    if not keys_raw:
        return {"status": "ok"}

    from cryptography.fernet import Fernet
    fernet_key = base64.urlsafe_b64encode(hashlib.sha256(settings.panel_secret_key.encode()).digest())
    f = Fernet(fernet_key)

    try:
        keys = json.loads(f.decrypt(keys_raw.encode()))
        keys.pop(f"{exchange}_api_key", None)
        keys.pop(f"{exchange}_secret_key", None)
        encrypted = f.encrypt(json.dumps(keys).encode()).decode()
        await redis_client.set("q:config:api_keys", encrypted)
    except Exception:
        pass

    await redis_client.publish("q:config:api_keys_updated", "1")
    return {"status": "ok", "message": f"{exchange} API keys deleted, env fallback active"}


# Telegram Settings (3.4)
@app.get("/api/v1/settings/telegram")
async def api_get_telegram(user: dict = Depends(get_current_user)):
    tg_raw = await redis_client.get("q:config:telegram")
    if tg_raw:
        config = json.loads(tg_raw)
        # Mask token
        if config.get("bot_token"):
            t = config["bot_token"]
            config["bot_token_masked"] = t[:6] + "****" + t[-4:] if len(t) > 10 else "****"
        return config
    return {
        "enabled": settings.monitoring.telegram_enabled,
        "bot_token_masked": "",
        "chat_id": settings.monitoring.telegram_chat_id or "",
    }


@app.post("/api/v1/settings/telegram")
async def api_save_telegram(request: Request, user: dict = Depends(get_current_user)):
    body = await request.json()
    config = {
        "enabled": body.get("enabled", False),
        "bot_token": body.get("bot_token", ""),
        "chat_id": body.get("chat_id", ""),
    }
    await redis_client.set("q:config:telegram", json.dumps(config))
    return {"status": "ok"}


@app.post("/api/v1/settings/telegram/test")
async def api_test_telegram(user: dict = Depends(get_current_user)):
    """Send a test Telegram message"""
    tg_raw = await redis_client.get("q:config:telegram")
    if not tg_raw:
        raise HTTPException(400, "Telegram not configured")
    config = json.loads(tg_raw)
    if not config.get("bot_token") or not config.get("chat_id"):
        raise HTTPException(400, "Bot token and chat ID required")

    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://api.telegram.org/bot{config['bot_token']}/sendMessage"
            resp = await session.post(url, json={
                "chat_id": config["chat_id"],
                "text": "H4wkQuant test message - Telegram connection successful!",
                "parse_mode": "HTML",
            })
            result = await resp.json()
            if result.get("ok"):
                return {"status": "ok", "message": "Test message sent"}
            else:
                return {"status": "error", "message": result.get("description", "Unknown error")}
    except Exception as e:
        raise HTTPException(500, f"Telegram error: {e}")


# ============================================================
# PostgreSQL Status (3.5)
# ============================================================

@app.get("/api/v1/db-status")
async def api_db_status(user: dict = Depends(get_current_user)):
    """Check PostgreSQL connection and table row counts"""
    try:
        from shared.database.connection import get_engine
        from sqlalchemy import text
        engine = get_engine()
        async with engine.connect() as conn:
            # Check connection
            await conn.execute(text("SELECT 1"))

            # Get table counts
            tables = {}
            for table in ["arb_trades", "account_snapshots", "spread_snapshots", "daily_stats"]:
                try:
                    result = await conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    tables[table] = result.scalar()
                except Exception:
                    tables[table] = -1

            return {
                "status": "connected",
                "host": settings.database.postgres_host,
                "database": settings.database.postgres_db,
                "tables": tables,
            }
    except Exception as e:
        return {
            "status": "disconnected",
            "error": str(e),
            "host": settings.database.postgres_host,
            "database": settings.database.postgres_db,
            "tables": {},
        }


# ============================================================
# Metrics Endpoint (3.6)
# ============================================================

@app.get("/api/v1/metrics")
async def api_metrics():
    """Prometheus-format metrics"""
    return PlainTextResponse(metrics.render(), media_type="text/plain")


@app.get("/api/v1/metrics/history")
async def api_metrics_history(hours: int = Query(6, le=24), user: dict = Depends(get_current_user)):
    """Get metrics history for charts (last N hours)"""
    count = min(hours * 120, 2880)  # 2 per minute
    raw = await redis_client.lrange("q:metrics:history", 0, count - 1)
    return [json.loads(r) for r in raw]


# ============================================================
# Control Endpoints
# ============================================================

@app.post("/api/v1/kill-switch")
async def api_kill_switch(active: bool = True, reason: str = "manual", user: dict = Depends(get_current_user)):
    await redis_client.set("q:control:kill_switch", json.dumps({
        "active": active,
        "reason": reason,
        "time": time.time(),
    }))
    return {"status": "ok", "kill_switch": active}


@app.post("/api/v1/close-all")
async def api_close_all(user: dict = Depends(get_current_user)):
    """Emergency: close all positions"""
    positions = await redis_client.get("q:active_positions")
    if not positions:
        return {"status": "no_positions"}

    for pair_id in json.loads(positions).keys():
        close_signal = {
            "pair_id": pair_id,
            "action": "close",
            "strategy": "manual",
            "zscore": 0,
            "edge_net": 0,
            "confidence": 1.0,
            "leg_a": {"side": "CLOSE"},
            "leg_b": {"side": "CLOSE"},
            "metadata": {"exit_reason": "manual_close_all"},
        }
        await redis_client.publish("arb.approved", json.dumps(close_signal))

    return {"status": "close_all_sent"}


# ============================================================
# Script Runner Endpoints
# ============================================================

_running_tasks: Dict[str, dict] = {}


@app.post("/api/v1/run/screen")
async def api_run_screen(user: dict = Depends(get_current_user)):
    return await _run_script("screen", "python", "-u", "scripts/pair_screener.py")


@app.post("/api/v1/run/backtest")
async def api_run_backtest(user: dict = Depends(get_current_user)):
    return await _run_script("backtest", "python", "-u", "scripts/backtest_enhanced.py")


@app.post("/api/v1/run/preflight")
async def api_run_preflight(user: dict = Depends(get_current_user)):
    return await _run_script("preflight", "python", "-u", "scripts/preflight_check.py")


@app.get("/api/v1/run/status")
async def api_run_status(user: dict = Depends(get_current_user)):
    result = {}
    for name, info in _running_tasks.items():
        proc = info.get("process")
        if proc and proc.returncode is None:
            result[name] = {"status": "running", "started": info.get("started")}
        else:
            result[name] = {
                "status": "finished",
                "exit_code": proc.returncode if proc else -1,
                "output": info.get("output", "")[-2000:],
                "started": info.get("started"),
            }
    return result


async def _run_script(name: str, *cmd):
    if name in _running_tasks:
        proc = _running_tasks[name].get("process")
        if proc and proc.returncode is None:
            return {"status": "already_running", "name": name}

    env = {**os.environ, "PYTHONPATH": "/app"}
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        env=env,
        cwd="/app",
    )
    _running_tasks[name] = {"process": proc, "started": time.time(), "output": ""}

    async def _collect():
        output = []
        async for line in proc.stdout:
            output.append(line.decode(errors="replace"))
        _running_tasks[name]["output"] = "".join(output)

    asyncio.create_task(_collect())
    return {"status": "started", "name": name}


@app.get("/api/v1/run/output/{name}")
async def api_run_output(name: str, user: dict = Depends(get_current_user)):
    if name not in _running_tasks:
        return {"status": "not_found"}
    info = _running_tasks[name]
    proc = info.get("process")
    return {
        "name": name,
        "status": "running" if proc and proc.returncode is None else "finished",
        "exit_code": proc.returncode if proc else -1,
        "output": info.get("output", "")[-5000:],
    }


@app.delete("/api/v1/screened-pair")
async def api_remove_screened_pair(pair: str, user: dict = Depends(get_current_user)):
    raw = await redis_client.get("q:config:screened_pairs")
    pairs = json.loads(raw) if raw else []
    if pair in pairs:
        pairs.remove(pair)
        if pairs:
            await redis_client.set("q:config:screened_pairs", json.dumps(pairs))
        else:
            await redis_client.delete("q:config:screened_pairs")
        await redis_client.publish("q:pairs:command", "clear")
    return {"status": "ok", "remaining": pairs}


@app.delete("/api/v1/screened-pairs")
async def api_clear_screened_pairs(user: dict = Depends(get_current_user)):
    await redis_client.delete("q:config:screened_pairs")
    await redis_client.delete("q:config:screened_details")
    await redis_client.publish("q:pairs:command", "clear")
    return {"status": "ok", "message": "Scan results cleared"}


@app.post("/api/v1/trading-mode")
async def api_trading_mode(mode: str = "paper", user: dict = Depends(get_current_user)):
    if mode not in ("paper", "live"):
        raise HTTPException(400, "Mode must be 'paper' or 'live'")
    await redis_client.set("q:control:trading_mode", mode)
    return {"status": "ok", "trading_mode": mode, "message": f"Mode set to {mode}. Restart services."}


@app.get("/api/v1/stats")
async def api_stats(user: dict = Depends(get_current_user)):
    trades_raw = await redis_client.lrange("q:trades:history", 0, -1)
    trades = [json.loads(t) for t in trades_raw]

    if not trades:
        return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, "total_pnl": 0}

    wins = [t for t in trades if (t.get("combined_pnl") or 0) > 0]
    losses = [t for t in trades if (t.get("combined_pnl") or 0) <= 0]
    total_pnl = sum(t.get("combined_pnl", 0) for t in trades)
    total_commission = sum(t.get("commission", 0) for t in trades)

    by_strategy = {}
    for t in trades:
        s = t.get("strategy", "unknown")
        by_strategy.setdefault(s, {"count": 0, "pnl": 0, "wins": 0})
        by_strategy[s]["count"] += 1
        by_strategy[s]["pnl"] += t.get("combined_pnl", 0)
        if (t.get("combined_pnl") or 0) > 0:
            by_strategy[s]["wins"] += 1

    return {
        "total": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(trades) * 100, 1) if trades else 0,
        "total_pnl": round(total_pnl, 4),
        "total_commission": round(total_commission, 4),
        "net_pnl": round(total_pnl, 4),
        "avg_pnl": round(total_pnl / len(trades), 4) if trades else 0,
        "by_strategy": by_strategy,
    }


@app.get("/api/v1/engine-status")
async def api_engine_status(user: dict = Depends(get_current_user)):
    all_pairs = list(settings.arbitrage.pairs)
    screened_raw = await redis_client.get("q:config:screened_pairs")
    if screened_raw:
        for p in json.loads(screened_raw):
            if p not in all_pairs:
                all_pairs.append(p)

    result = []
    for pair in all_pairs:
        spread_raw = await redis_client.get(f"q:spread:{pair}")
        if spread_raw:
            sd = json.loads(spread_raw)
            result.append({
                "pair": pair,
                "ticks": sd.get("ticks", 0),
                "warmup_remaining": sd.get("warmup_remaining", 0),
                "is_ready": sd.get("is_ready", False),
                "zscore": sd.get("zscore"),
                "half_life": sd.get("half_life"),
                "coint_pvalue": sd.get("coint_pvalue"),
                "is_default": pair in settings.arbitrage.pairs,
                "tf_5m_coint": sd.get("tf_5m_coint"),
                "tf_15m_coint": sd.get("tf_15m_coint"),
            })
        else:
            result.append({
                "pair": pair,
                "ticks": 0,
                "warmup_remaining": 7200,
                "is_ready": False,
                "zscore": None,
                "half_life": None,
                "coint_pvalue": None,
                "is_default": pair in settings.arbitrage.pairs,
                "tf_5m_coint": None,
                "tf_15m_coint": None,
            })

    return result


@app.get("/api/v1/montecarlo")
async def api_montecarlo(user: dict = Depends(get_current_user)):
    trades_raw = await redis_client.lrange("q:trades:history", 0, -1)
    if len(trades_raw) < 5:
        return {"status": "insufficient", "trades": len(trades_raw), "min_required": 5}

    trades = [json.loads(t) for t in trades_raw]
    returns = []
    for t in trades:
        pnl = t.get("combined_pnl", 0)
        size = t.get("leg_a_qty", 0) * t.get("leg_a_entry", 0)
        if size > 0:
            returns.append(pnl / size)

    if len(returns) < 5:
        return {"status": "insufficient", "trades": len(returns), "min_required": 5}

    balance_data = await redis_client.get("q:account:balance")
    balance = json.loads(balance_data).get("total_balance", 174.0) if balance_data else 174.0

    mc = MonteCarloModel(n_simulations=settings.arbitrage.mc_simulations)
    result = mc.simulate(returns, initial_balance=balance)

    return {
        "status": "ok",
        "trades_used": len(returns),
        "simulations": result.n_simulations,
        "sharpe_ratio": result.sharpe_ratio,
        "probability_of_profit": round(result.probability_of_profit * 100, 1),
        "probability_of_ruin": round(result.probability_of_ruin * 100, 2),
        "mean_max_drawdown": round(result.mean_max_drawdown * 100, 2),
        "worst_max_drawdown": round(result.worst_max_drawdown * 100, 2),
        "mean_return": round(result.mean_return * 100, 2),
        "pct_5": round(result.pct_5 * 100, 2),
        "pct_95": round(result.pct_95 * 100, 2),
        "passes_sharpe": result.passes_sharpe,
        "passes_drawdown": result.passes_drawdown,
        "passes_ruin": result.passes_ruin,
        "is_valid": result.is_valid,
    }


# Portfolio correlation endpoint
@app.get("/api/v1/portfolio/correlation")
async def api_portfolio_correlation(user: dict = Depends(get_current_user)):
    corr_raw = await redis_client.get("q:portfolio:correlation")
    if corr_raw:
        return json.loads(corr_raw)
    return {}


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8180)


# ============================================================
# Auto Scan (Backend-side, works without panel open)
# ============================================================

_auto_scan_task = None

@app.post("/api/v1/autoscan/start")
async def api_start_autoscan(body: dict = {}, user: dict = Depends(get_current_user)):
    global _auto_scan_task
    interval = int(body.get("interval_minutes", 30))
    if interval < 5:
        interval = 5
    if interval > 360:
        interval = 360

    # Save config to Redis so it persists
    await redis_client.set("q:config:autoscan", json.dumps({"enabled": True, "interval_minutes": interval}))

    # Cancel existing task
    if _auto_scan_task and not _auto_scan_task.done():
        _auto_scan_task.cancel()

    _auto_scan_task = asyncio.create_task(_autoscan_loop(interval))
    logger.info(f"Auto scan started: every {interval} minutes")
    return {"status": "started", "interval_minutes": interval}


@app.post("/api/v1/autoscan/stop")
async def api_stop_autoscan(user: dict = Depends(get_current_user)):
    global _auto_scan_task
    await redis_client.set("q:config:autoscan", json.dumps({"enabled": False}))
    if _auto_scan_task and not _auto_scan_task.done():
        _auto_scan_task.cancel()
        _auto_scan_task = None
    logger.info("Auto scan stopped")
    return {"status": "stopped"}


@app.get("/api/v1/autoscan/status")
async def api_autoscan_status():
    raw = await redis_client.get("q:config:autoscan")
    if raw:
        cfg = json.loads(raw)
        running = _auto_scan_task is not None and not _auto_scan_task.done() if _auto_scan_task else False
        cfg["running"] = running
        return cfg
    return {"enabled": False, "running": False}


async def _autoscan_loop(interval_minutes: int):
    """Background loop: run pair screener every N minutes"""
    import subprocess
    while True:
        try:
            logger.info(f"Auto scan: running pair screener...")
            proc = await asyncio.create_subprocess_exec(
                "python", "-u", "scripts/pair_screener.py",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=300)
            if proc.returncode == 0:
                logger.info(f"Auto scan completed successfully")
            else:
                logger.error(f"Auto scan failed: {stderr.decode()[-200:]}")
        except asyncio.CancelledError:
            logger.info("Auto scan loop cancelled")
            return
        except Exception as e:
            logger.error(f"Auto scan error: {e}")

        await asyncio.sleep(interval_minutes * 60)


# Startup: restore auto scan from Redis
@app.on_event("startup")
async def _restore_autoscan():
    global _auto_scan_task
    try:
        raw = await redis_client.get("q:config:autoscan")
        if raw:
            cfg = json.loads(raw)
            if cfg.get("enabled"):
                interval = cfg.get("interval_minutes", 30)
                _auto_scan_task = asyncio.create_task(_autoscan_loop(interval))
                logger.info(f"Auto scan restored from Redis: every {interval} minutes")
    except Exception as e:
        logger.debug(f"Auto scan restore error: {e}")
