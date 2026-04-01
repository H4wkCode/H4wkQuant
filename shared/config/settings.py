"""
H4wkQuant - Configuration
Mathematical arbitrage trading system configuration
"""
import os
import json
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import List, Optional, Literal
from enum import Enum


class TradingMode(str, Enum):
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# =============================================================================
# BINANCE CONFIGURATION
# =============================================================================
class BinanceConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BINANCE_")

    api_key: str = Field(default="")
    secret_key: str = Field(default="")

    testnet: bool = Field(default=False)
    futures_testnet: bool = Field(default=False)

    # Rate Limiting
    rate_limit_requests_per_minute: int = 1200
    rate_limit_weight_per_minute: int = 6000

    # WebSocket
    ws_reconnect_interval: int = 86400
    ws_ping_interval: int = 20
    ws_pong_timeout: int = 10

    @property
    def rest_url(self) -> str:
        return "https://testnet.binancefuture.com" if self.futures_testnet else "https://fapi.binance.com"

    @property
    def ws_url(self) -> str:
        return "wss://stream.binancefuture.com" if self.futures_testnet else "wss://fstream.binance.com"


# =============================================================================
# ARBITRAGE STRATEGY CONFIGURATION
# =============================================================================
class ArbitrageConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ARB_")

    # Pairs Trading
    pairs: List[str] = ["BTCUSDT/ETHUSDT", "SOLUSDT/ETHUSDT", "BNBUSDT/ETHUSDT"]
    zscore_entry_threshold: float = 3.0  # was 2.5 - wait for stronger signal
    zscore_exit_threshold: float = 0.8
    lookback_window: int = 1440  # 8 hours of 1m candles
    half_life_max: int = 60  # max mean reversion half-life (minutes)
    cointegration_pvalue: float = 0.05  # was 0.03 - fewer cointegration breaks

    # Funding Rate Arb
    funding_arb_enabled: bool = True
    funding_rate_threshold: float = 0.0005  # 0.1% per 8h
    funding_min_annualized: float = 0.05  # 10% annualized minimum
    funding_min_hold_minutes: int = 60  # minimum hold before close (1 funding period)

    # Momentum Divergence
    momentum_enabled: bool = False  # disabled - too risky for small capital
    divergence_lag_max_seconds: int = 300  # 5 minutes (was 30s)
    divergence_min_move_pct: float = 1.0  # BTC must move >1.0% (was 0.5%)
    momentum_max_position_pct: float = 0.25  # max 25% of equity

    # Dynamic Entry/Exit Thresholds
    dynamic_threshold_enabled: bool = True
    dynamic_entry_min: float = 3.0  # was 2.5 - stronger signal
    dynamic_entry_max: float = 4.5  # was 4.0 - high threshold for slow pairs
    dynamic_exit_min: float = 0.5  # was 0.8 - closer exit = more profit
    dynamic_exit_max: float = 1.2  # was 1.5

    # Kelly Criterion
    kelly_fraction: float = 0.25  # Quarter Kelly (conservative)
    kelly_min_trades: int = 20  # Min history for Kelly calculation

    # Stoikov Market Making
    stoikov_gamma: float = 0.1  # Risk aversion parameter
    stoikov_k: float = 1.5  # Order book depth parameter
    stoikov_max_inventory: float = 3.0  # Max inventory imbalance

    # Monte Carlo
    mc_simulations: int = 1000
    mc_min_sharpe: float = 1.0
    mc_max_drawdown: float = 0.05  # 5%
    mc_max_ruin_prob: float = 0.01  # 1%

    # Kalman Filter (half-life smoothing)
    kalman_enabled: bool = False
    kalman_process_variance: float = 0.5
    kalman_measurement_variance: float = 5.0

    # Multi-Timeframe (5m/15m cointegration check)
    multi_tf_enabled: bool = False  # Reduces signal count, keep disabled for small capital

    # Portfolio Correlation Check
    portfolio_corr_enabled: bool = False  # Unnecessary with 3 pairs, enable when capital grows

    # Regime Detection
    regime_detection_enabled: bool = True
    regime_vol_window: int = 60
    regime_history_window: int = 1440
    regime_high_percentile: float = 75.0
    regime_extreme_percentile: float = 90.0


# =============================================================================
# RISK MANAGEMENT - HARD LIMITS
# =============================================================================
class RiskConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RISK_")

    # Portfolio Level
    max_total_leverage: float = 3.0
    max_open_positions: int = 6  # 3 arb pairs = 6 legs
    max_daily_loss_percent: float = 2.0
    max_weekly_loss_percent: float = 5.0
    max_drawdown_percent: float = 10.0

    # Position Level
    max_risk_per_trade_percent: float = 1.0
    min_leg_size_usd: float = 20.0  # was 5 - minimum to cover commissions
    max_leg_size_usd: float = 300.0  # $500 * 3x / 3 pairs / 2 legs ~= $250
    default_leverage: int = 3

    # Paper Trading Virtual Balance
    paper_initial_balance: float = 500.0  # Virtual starting balance (adjustable from panel)

    # Spread/Fee Awareness
    maker_fee: float = 0.0002  # 0.02% maker
    taker_fee: float = 0.0004  # 0.04% taker
    max_slippage_percent: float = 0.05

    # Kill Switch
    kill_switch_enabled: bool = True
    kill_switch_auto_reset: bool = False


# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
class DatabaseConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_")

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "h4wk"
    postgres_password: str = Field(default_factory=lambda: os.environ.get("DB_POSTGRES_PASSWORD", "changeme"))
    postgres_db: str = "h4wk_quant"

    timescale_enabled: bool = True

    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 1  # DB 1 for H4wkQuant (DB 0 = H4wkTrading)
    redis_password: Optional[str] = None

    redis_sentinel_hosts: Optional[str] = None
    redis_sentinel_master: str = "h4wkmaster"

    db_pool_size: int = 10
    db_max_overflow: int = 5

    @property
    def postgres_url(self) -> str:
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def redis_url(self) -> str:
        # Check REDIS_URL env var first (set by Docker Compose)
        env_url = os.environ.get("REDIS_URL")
        if env_url:
            return env_url
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"


# =============================================================================
# MONITORING
# =============================================================================
class MonitoringConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MONITOR_")

    log_level: LogLevel = LogLevel.INFO

    telegram_enabled: bool = False
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None


# =============================================================================
# BYBIT CONFIGURATION
# =============================================================================
class BybitConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BYBIT_")

    api_key: str = Field(default="")
    secret_key: str = Field(default="")
    testnet: bool = Field(default=False)
    enabled: bool = Field(default=False)

    @property
    def rest_url(self) -> str:
        return "https://api-testnet.bybit.com" if self.testnet else "https://api.bybit.com"

    @property
    def ws_url(self) -> str:
        return "wss://stream-testnet.bybit.com/v5/public/linear" if self.testnet else "wss://stream.bybit.com/v5/public/linear"


# =============================================================================
# MAIN SETTINGS
# =============================================================================
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    service_name: str = "h4wk-quant"
    service_version: str = "1.0.0"
    environment: Literal["development", "staging", "production"] = "development"

    binance: BinanceConfig = Field(default_factory=BinanceConfig)
    arbitrage: ArbitrageConfig = Field(default_factory=ArbitrageConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    bybit: BybitConfig = Field(default_factory=BybitConfig)

    trading_mode: TradingMode = TradingMode.PAPER
    panel_host: str = "0.0.0.0"
    panel_port: int = 8180
    panel_secret_key: str = Field(default_factory=lambda: os.environ.get("PANEL_SECRET_KEY", os.urandom(32).hex()))


settings = Settings()


async def load_api_keys_from_redis(redis_client):
    """Load encrypted API keys from Redis and apply to settings."""
    import hashlib
    import base64
    try:
        keys_raw = await redis_client.get("q:config:api_keys")
        if not keys_raw:
            return
        from cryptography.fernet import Fernet
        fernet_key = base64.urlsafe_b64encode(hashlib.sha256(settings.panel_secret_key.encode()).digest())
        f = Fernet(fernet_key)
        keys = json.loads(f.decrypt(keys_raw.encode()))

        if keys.get("binance_api_key"):
            settings.binance.api_key = keys["binance_api_key"]
        if keys.get("binance_secret_key"):
            settings.binance.secret_key = keys["binance_secret_key"]
        if keys.get("bybit_api_key"):
            settings.bybit.api_key = keys["bybit_api_key"]
        if keys.get("bybit_secret_key"):
            settings.bybit.secret_key = keys["bybit_secret_key"]
    except Exception as e:
        pass  # Fail silently, env fallback


async def get_overrides(redis_client) -> dict:
    """Read config overrides from Redis."""
    try:
        raw = await redis_client.get("q:config:overrides")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return {}


async def apply_overrides(redis_client):
    """Apply Redis config overrides to settings singleton."""
    overrides = await get_overrides(redis_client)
    if not overrides:
        return

    arb = settings.arbitrage
    risk = settings.risk
    field_map = {
        "zscore_entry": ("arbitrage", "zscore_entry_threshold"),
        "zscore_exit": ("arbitrage", "zscore_exit_threshold"),
        "half_life_max": ("arbitrage", "half_life_max"),
        "max_open_positions": ("risk", "max_open_positions"),
        "default_leverage": ("risk", "default_leverage"),
        "kelly_fraction": ("arbitrage", "kelly_fraction"),
        "kalman_enabled": ("arbitrage", "kalman_enabled"),
        "regime_detection_enabled": ("arbitrage", "regime_detection_enabled"),
        "funding_arb_enabled": ("arbitrage", "funding_arb_enabled"),
        "momentum_enabled": ("arbitrage", "momentum_enabled"),
        "multi_tf_enabled": ("arbitrage", "multi_tf_enabled"),
        "portfolio_corr_enabled": ("arbitrage", "portfolio_corr_enabled"),
        "paper_initial_balance": ("risk", "paper_initial_balance"),
        "max_leg_size_usd": ("risk", "max_leg_size_usd"),
        "warmup_seconds": (None, None),  # handled specially
    }

    for key, value in overrides.items():
        if key in field_map:
            section, attr = field_map[key]
            if section == "arbitrage":
                setattr(arb, attr, type(getattr(arb, attr))(value))
            elif section == "risk":
                setattr(risk, attr, type(getattr(risk, attr))(value))


def setup_service_logging(service_name: str = None):
    """Service log file configuration"""
    from pathlib import Path
    from loguru import logger

    service_name = service_name or os.environ.get("SERVICE_NAME", "unknown")

    if os.environ.get("SERVICE_NAME"):
        log_dir = Path("/app/logs")
    else:
        log_dir = Path(__file__).parent.parent.parent / "logs"

    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_dir / f"{service_name}.log"),
        rotation="10 MB",
        retention="3 days",
        encoding="utf-8",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}",
    )
