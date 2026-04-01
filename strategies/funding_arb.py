"""
H4wkQuant - Funding Rate Arbitrage
Collect funding payments from extreme funding rates.

When funding > 0.1% per 8h:
- Short the asset (receive funding)
- Hedge with correlated asset or spot position

Market neutral: profit from funding, not price movement.
"""
import numpy as np
from typing import Optional, Dict, List
from loguru import logger

from strategies.base import BaseStrategy
from models.edge import EdgeModel
from models.kelly import KellyModel
from shared.schemas.models import ArbSignal, ArbSignalAction, StrategyType
from shared.config.settings import settings


class FundingArbStrategy(BaseStrategy):
    """
    Funding Rate Arbitrage.

    Entry:
    - |funding_rate| > threshold (0.1% per 8h)
    - Annualized funding > 10%
    - Short the highly-funded side, hedge with correlated pair

    Exit:
    - Funding normalizes (< 0.03%)
    - Next funding period collected

    Pairs: Each symbol paired with its hedge (e.g., BTCUSDT hedged with ETHUSDT)
    """

    def __init__(self):
        super().__init__("funding_arb")

        arb = settings.arbitrage
        self.enabled = arb.funding_arb_enabled
        self.funding_threshold = arb.funding_rate_threshold
        self.min_annualized = arb.funding_min_annualized
        self.min_hold_minutes = arb.funding_min_hold_minutes

        # Hedge pairs: {symbol: hedge_symbol}
        self.hedge_map = {
            "BTCUSDT": "ETHUSDT",
            "ETHUSDT": "BTCUSDT",
            "SOLUSDT": "ETHUSDT",
            "BNBUSDT": "ETHUSDT",
        }

        self.edge_model = EdgeModel(
            maker_fee=settings.risk.maker_fee,
            taker_fee=settings.risk.taker_fee,
        )
        self.kelly_model = KellyModel(fraction=arb.kelly_fraction)
        self.trade_returns: List[float] = []

    def get_pairs(self) -> List[str]:
        return [f"{s}/{h}" for s, h in self.hedge_map.items()]

    async def evaluate(self, market_data: Dict) -> Optional[ArbSignal]:
        """
        Check if funding rate is exploitable.

        market_data:
            symbol: str
            funding_rate: float (per 8h)
            next_funding_time: int (unix ms)
            price: float
            hedge_symbol: str (optional, auto from hedge_map)
            hedge_price: float
            equity: float
        """
        if not self.enabled:
            return None

        symbol = market_data["symbol"]
        funding_rate = market_data["funding_rate"]
        price = market_data["price"]
        equity = market_data.get("equity", 174.0)

        # Check threshold
        if abs(funding_rate) < self.funding_threshold:
            return None

        # Annualized check: funding * 3 * 365
        annualized = abs(funding_rate) * 3 * 365
        if annualized < self.min_annualized:
            return None

        # Determine direction
        hedge_symbol = market_data.get("hedge_symbol", self.hedge_map.get(symbol))
        if not hedge_symbol:
            return None

        hedge_price = market_data.get("hedge_price", 0)
        if hedge_price <= 0:
            return None

        pair_id = f"{symbol}/{hedge_symbol}"

        # If funding is positive -> shorts receive payment -> short the asset
        # If funding is negative -> longs receive payment -> long the asset
        if funding_rate > 0:
            main_side = "SHORT"
            hedge_side = "LONG"
        else:
            main_side = "LONG"
            hedge_side = "SHORT"

        # Edge: funding income vs trading costs
        notional = min(equity * settings.risk.default_leverage / 4, settings.risk.max_leg_size_usd)

        # Gross edge = 1 funding period payment
        gross_edge = abs(funding_rate) * notional
        commission_cost = 4 * settings.risk.maker_fee * notional  # 2 legs, entry + exit
        net_edge = gross_edge - commission_cost

        if net_edge <= 0:
            return None

        # Kelly
        kelly_result = self.kelly_model.calculate(equity, self.trade_returns)
        position_size = min(kelly_result.position_size_usd, notional)

        confidence = min(annualized / 0.5, 1.0)  # Higher annualized = higher confidence

        return ArbSignal(
            pair_id=pair_id,
            strategy=StrategyType.FUNDING_ARB,
            action=ArbSignalAction.OPEN,
            zscore=0.0,
            edge_net=net_edge,
            confidence=confidence,
            leg_a={"symbol": symbol, "side": main_side, "weight": 1.0},
            leg_b={"symbol": hedge_symbol, "side": hedge_side, "weight": 1.0},
            kelly_size=kelly_result.fractional_kelly,
            position_size_usd=round(position_size, 2),
            metadata={
                "funding_rate": funding_rate,
                "annualized": round(annualized, 4),
                "gross_edge": round(gross_edge, 6),
            },
        )

    async def should_close(self, position_data: Dict, market_data: Dict) -> Optional[ArbSignal]:
        """Close after funding collected or funding normalized (with min hold)"""
        import time
        symbol = position_data.get("leg_a_symbol", "")
        funding_rate = market_data.get("funding_rate", 0)
        funding_collected = market_data.get("funding_collected", False)

        # Minimum hold time check
        entry_time = position_data.get("entry_time", 0)
        elapsed_minutes = (time.time() - entry_time) / 60 if entry_time else 0

        exit_reason = None

        # Collected funding -> exit (always allowed)
        if funding_collected:
            exit_reason = "funding_collected"

        # Don't exit before min hold unless funding collected
        elif elapsed_minutes < self.min_hold_minutes:
            return None

        # Funding normalized (only after min hold)
        elif abs(funding_rate) < self.funding_threshold * 0.3:
            exit_reason = "funding_normalized"

        if not exit_reason:
            return None

        hedge_symbol = self.hedge_map.get(symbol, "")
        pair_id = f"{symbol}/{hedge_symbol}"

        return ArbSignal(
            pair_id=pair_id,
            strategy=StrategyType.FUNDING_ARB,
            action=ArbSignalAction.CLOSE,
            zscore=0.0,
            edge_net=0.0,
            confidence=1.0,
            leg_a={"symbol": symbol, "side": "CLOSE"},
            leg_b={"symbol": hedge_symbol, "side": "CLOSE"},
            metadata={"exit_reason": exit_reason},
        )

    def record_trade_result(self, return_pct: float):
        self.trade_returns.append(return_pct)
