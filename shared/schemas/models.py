"""
H4wkQuant - Shared Pydantic Models
Inter-service data contracts for arbitrage trading
"""
from typing import Optional, List, Dict
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


# ============================================================
# Enums
# ============================================================

class TradeSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TradingMode(str, Enum):
    LIVE = "live"
    PAPER = "paper"


class StrategyType(str, Enum):
    STAT_ARB = "stat_arb"
    FUNDING_ARB = "funding_arb"
    MOMENTUM_DIV = "momentum_div"
    CROSS_EXCHANGE = "cross_exchange"


class ArbSignalAction(str, Enum):
    OPEN = "open"
    CLOSE = "close"
    HOLD = "hold"


# ============================================================
# Spread / Z-Score Models
# ============================================================

class SpreadData(BaseModel):
    """Real-time spread data between two assets"""
    pair_id: str  # e.g. "BTCUSDT/ETHUSDT"
    symbol_a: str
    symbol_b: str
    price_a: float
    price_b: float
    ratio: float
    spread: float
    zscore: float
    mean: float
    std: float
    half_life: float = 0.0
    cointegration_pvalue: float = 1.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ArbSignal(BaseModel):
    """Spread Engine -> Risk Manager"""
    pair_id: str
    strategy: StrategyType
    action: ArbSignalAction
    zscore: float
    edge_net: float  # Net expected value after costs
    confidence: float = Field(ge=0.0, le=1.0)
    leg_a: dict  # {"symbol": "BTCUSDT", "side": "LONG", "weight": 1.0}
    leg_b: dict  # {"symbol": "ETHUSDT", "side": "SHORT", "weight": 0.85}
    kelly_size: float = 0.0  # Kelly-optimal fraction of capital
    position_size_usd: float = 0.0
    metadata: Optional[dict] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# Execution Models
# ============================================================

class LegOrder(BaseModel):
    """Single leg of an arbitrage trade"""
    symbol: str
    side: TradeSide
    size_usd: float
    leverage: int = Field(ge=1, le=20, default=3)
    order_type: OrderType = OrderType.LIMIT
    price: float = 0.0  # Stoikov-adjusted price for limit orders
    is_entry: bool = True
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ArbOrder(BaseModel):
    """Risk Manager -> Executor: Two-legged arbitrage order"""
    pair_id: str
    strategy: StrategyType
    leg_a: LegOrder
    leg_b: LegOrder
    signal_data: Optional[dict] = None
    trading_mode: TradingMode = TradingMode.PAPER
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# Position Model
# ============================================================

class Position(BaseModel):
    """Open position"""
    symbol: str
    side: TradeSide
    size: float
    entry_price: float
    mark_price: float = 0.0
    unrealized_pnl: float = 0.0
    leverage: int = 1
    liquidation_price: float = 0.0
    margin_type: str = "ISOLATED"
    notional_value: float = 0.0
    update_time: Optional[datetime] = None


class ArbPosition(BaseModel):
    """Combined arbitrage position (2 legs)"""
    pair_id: str
    strategy: StrategyType
    leg_a: Position
    leg_b: Position
    entry_zscore: float = 0.0
    current_zscore: float = 0.0
    combined_pnl: float = 0.0
    entry_time: datetime = Field(default_factory=datetime.utcnow)


# ============================================================
# Account Model
# ============================================================

class AccountBalance(BaseModel):
    total_balance: float
    available_balance: float
    unrealized_pnl: float = 0.0
    realized_pnl_today: float = 0.0
    margin_balance: float = 0.0
    update_time: Optional[datetime] = None


# ============================================================
# Trade Record
# ============================================================

class ArbTradeRecord(BaseModel):
    """Completed arbitrage trade"""
    id: str
    pair_id: str
    strategy: StrategyType
    leg_a_symbol: str
    leg_a_side: TradeSide
    leg_a_entry: float
    leg_a_exit: float
    leg_b_symbol: str
    leg_b_side: TradeSide
    leg_b_entry: float
    leg_b_exit: float
    quantity_a: float
    quantity_b: float
    leverage: int = 1
    combined_pnl: float = 0.0
    total_commission: float = 0.0
    entry_zscore: float = 0.0
    exit_zscore: float = 0.0
    exit_reason: Optional[str] = None
    trading_mode: TradingMode = TradingMode.PAPER
    entry_time: datetime = Field(default_factory=datetime.utcnow)
    exit_time: Optional[datetime] = None


# ============================================================
# Risk Assessment
# ============================================================

class RiskAssessment(BaseModel):
    result: str  # approved, rejected
    reason: str
    max_leverage: int = 3
    position_size_usd: float = 0.0
    risk_amount_usd: float = 0.0
    warnings: List[str] = Field(default_factory=list)


# ============================================================
# Bayesian Model Output
# ============================================================

class BayesianEstimate(BaseModel):
    """Output from the Bayesian fair value model"""
    symbol: str
    fair_price: float
    current_price: float
    mispricing_pct: float  # (fair - current) / current * 100
    orderbook_imbalance: float  # -1 to 1
    volume_signal: float  # -1 to 1
    funding_signal: float  # -1 to 1
    oi_signal: float  # -1 to 1
    posterior_long: float  # P(price goes up)
    posterior_short: float  # P(price goes down)
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
