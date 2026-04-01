"""
H4wkQuant - Database Models
SQLAlchemy ORM models for arbitrage trading
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, BigInteger, Float, String, Text, Boolean,
    DateTime, Index,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class ArbTrade(Base):
    """Arbitrage trade records - paired legs"""
    __tablename__ = "arb_trades"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    pair_id = Column(String(40), nullable=False, index=True)  # BTCUSDT/ETHUSDT
    strategy = Column(String(20), nullable=False)  # stat_arb, funding_arb, momentum_div

    # Leg A
    leg_a_symbol = Column(String(20), nullable=False)
    leg_a_side = Column(String(10), nullable=False)
    leg_a_entry_price = Column(Float, nullable=False)
    leg_a_exit_price = Column(Float, nullable=True)
    leg_a_quantity = Column(Float, nullable=False)
    leg_a_order_id = Column(String(100), nullable=True)

    # Leg B
    leg_b_symbol = Column(String(20), nullable=False)
    leg_b_side = Column(String(10), nullable=False)
    leg_b_entry_price = Column(Float, nullable=False)
    leg_b_exit_price = Column(Float, nullable=True)
    leg_b_quantity = Column(Float, nullable=False)
    leg_b_order_id = Column(String(100), nullable=True)

    leverage = Column(Integer, default=3)
    entry_zscore = Column(Float, nullable=True)
    exit_zscore = Column(Float, nullable=True)
    combined_pnl = Column(Float, nullable=True)
    total_commission = Column(Float, default=0)
    exit_reason = Column(String(50), nullable=True)
    trading_mode = Column(String(10), default="paper")
    entry_time = Column(DateTime, nullable=False, default=datetime.utcnow)
    exit_time = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("idx_arb_trades_pair_time", "pair_id", "entry_time"),
        Index("idx_arb_trades_strategy", "strategy"),
    )


class SpreadHistory(Base):
    """Spread/Z-score time series - TimescaleDB hypertable"""
    __tablename__ = "spread_history"

    time = Column(DateTime, primary_key=True, nullable=False)
    pair_id = Column(String(40), primary_key=True, nullable=False)
    ratio = Column(Float, nullable=False)
    spread = Column(Float, nullable=False)
    zscore = Column(Float, nullable=False)
    mean = Column(Float, nullable=True)
    std = Column(Float, nullable=True)
    half_life = Column(Float, nullable=True)

    __table_args__ = (
        Index("idx_spread_pair_time", "pair_id", "time"),
    )


class AccountSnapshot(Base):
    """Account balance snapshots"""
    __tablename__ = "quant_account_snapshots"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    time = Column(DateTime, nullable=False, default=datetime.utcnow)
    total_balance = Column(Float, nullable=False)
    available_balance = Column(Float, nullable=False)
    unrealized_pnl = Column(Float, default=0)
    realized_pnl_today = Column(Float, default=0)
    open_positions = Column(Integer, default=0)
    total_leverage = Column(Float, default=0)

    __table_args__ = (
        Index("idx_quant_snapshots_time", "time"),
    )


class DailyStat(Base):
    """Daily PnL statistics"""
    __tablename__ = "quant_daily_stats"

    date = Column(DateTime, primary_key=True, nullable=False)
    realized_pnl = Column(Float, default=0)
    total_commission = Column(Float, default=0)
    trade_count = Column(Integer, default=0)
    win_count = Column(Integer, default=0)
    loss_count = Column(Integer, default=0)
    balance_start = Column(Float, default=0)
    balance_end = Column(Float, default=0)
    best_trade_pnl = Column(Float, default=0)
    worst_trade_pnl = Column(Float, default=0)
    sharpe_ratio = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)
