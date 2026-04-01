"""
H4wkQuant - Cross-Pair Momentum Divergence
When BTC moves sharply, altcoins lag by seconds.
Catch the lag, close within seconds.

Short-term alpha from cross-asset momentum propagation.
"""
import time
import numpy as np
from typing import Optional, Dict, List
from collections import deque
from loguru import logger

from strategies.base import BaseStrategy
from models.edge import EdgeModel
from shared.schemas.models import ArbSignal, ArbSignalAction, StrategyType
from shared.config.settings import settings


class MomentumDivStrategy(BaseStrategy):
    """
    Cross-Pair Momentum Divergence.

    Logic:
    1. BTC makes a sharp move (>0.5% in last N seconds)
    2. An altcoin hasn't moved proportionally yet
    3. Enter the altcoin in BTC's direction
    4. Exit within 30 seconds or at target

    This is the fastest strategy - relies on speed.
    """

    def __init__(self):
        super().__init__("momentum_div")

        arb = settings.arbitrage
        self.enabled = arb.momentum_enabled
        self.lag_max_seconds = arb.divergence_lag_max_seconds
        self.min_move_pct = arb.divergence_min_move_pct
        self.max_position_pct = arb.momentum_max_position_pct

        # BTC as the leader
        self.leader = "BTCUSDT"
        self.followers = ["ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT"]

        # Recent price ticks (timestamp, price)
        self.leader_ticks: deque = deque(maxlen=300)  # 5 min of ticks
        self.follower_ticks: Dict[str, deque] = {
            s: deque(maxlen=300) for s in self.followers
        }

        # Correlation coefficients (pre-calculated)
        self.betas: Dict[str, float] = {
            "ETHUSDT": 0.85,
            "SOLUSDT": 1.2,
            "BNBUSDT": 0.7,
            "DOGEUSDT": 1.5,
        }

        self.edge_model = EdgeModel(
            maker_fee=settings.risk.maker_fee,
            taker_fee=settings.risk.taker_fee,
        )

    def get_pairs(self) -> List[str]:
        return [f"{self.leader}/{f}" for f in self.followers]

    def update_tick(self, symbol: str, price: float, timestamp: float = None):
        """Feed price tick"""
        ts = timestamp or time.time()
        if symbol == self.leader:
            self.leader_ticks.append((ts, price))
        elif symbol in self.follower_ticks:
            self.follower_ticks[symbol].append((ts, price))

    async def evaluate(self, market_data: Dict) -> Optional[ArbSignal]:
        """
        Check for momentum divergence.

        market_data:
            symbol: str (follower)
            price: float
            btc_price: float
            equity: float
        """
        if not self.enabled:
            return None

        symbol = market_data.get("symbol", "")
        if symbol not in self.followers:
            return None

        if len(self.leader_ticks) < 10:
            return None

        follower_ticks = self.follower_ticks.get(symbol)
        if not follower_ticks or len(follower_ticks) < 5:
            return None

        now = time.time()

        # BTC move in last N seconds
        btc_move = self._calculate_move(self.leader_ticks, self.lag_max_seconds)
        if btc_move is None or abs(btc_move) < self.min_move_pct:
            return None

        # Follower move in same period
        follower_move = self._calculate_move(follower_ticks, self.lag_max_seconds)
        if follower_move is None:
            return None

        # Expected follower move based on beta
        beta = self.betas.get(symbol, 1.0)
        expected_move = btc_move * beta

        # Divergence: how much the follower underreacted
        divergence = expected_move - follower_move

        # Need significant divergence
        if abs(divergence) < self.min_move_pct * 0.5:
            return None

        # Direction: trade in the direction BTC moved
        pair_id = f"{self.leader}/{symbol}"
        current_price = market_data["price"]
        equity = market_data.get("equity", 174.0)

        # Small notional for this fast strategy
        notional = min(equity * self.max_position_pct, settings.risk.max_leg_size_usd * 0.5)

        # Edge: expected catch-up minus costs (use taker for speed)
        edge = self.edge_model.calculate(
            zscore=abs(divergence) / self.min_move_pct,
            spread_std=abs(divergence) / 100,  # Convert pct to fraction
            notional_per_leg=notional,
            use_limit_orders=False,  # Taker for speed
        )

        if not edge.is_profitable:
            return None

        if divergence > 0:
            # BTC went up, follower lagging -> long follower
            side = "LONG"
        else:
            # BTC went down, follower lagging -> short follower
            side = "SHORT"

        return ArbSignal(
            pair_id=pair_id,
            strategy=StrategyType.MOMENTUM_DIV,
            action=ArbSignalAction.OPEN,
            zscore=divergence / self.min_move_pct,
            edge_net=edge.net_edge,
            confidence=min(abs(divergence) / (self.min_move_pct * 2), 1.0),
            leg_a={"symbol": symbol, "side": side, "weight": 1.0},
            leg_b={"symbol": self.leader, "side": "NONE", "weight": 0},  # No hedge leg
            position_size_usd=round(notional, 2),
            metadata={
                "btc_move": round(btc_move, 4),
                "follower_move": round(follower_move, 4),
                "divergence": round(divergence, 4),
                "beta": beta,
            },
        )

    async def should_close(self, position_data: Dict, market_data: Dict) -> Optional[ArbSignal]:
        """Close after time limit or when divergence closes"""
        entry_time = position_data.get("entry_time", 0)
        elapsed = time.time() - entry_time

        symbol = position_data.get("leg_a_symbol", "")
        pair_id = f"{self.leader}/{symbol}"

        # Time-based exit (max 30 seconds)
        if elapsed > self.lag_max_seconds:
            return ArbSignal(
                pair_id=pair_id,
                strategy=StrategyType.MOMENTUM_DIV,
                action=ArbSignalAction.CLOSE,
                zscore=0.0, edge_net=0.0, confidence=1.0,
                leg_a={"symbol": symbol, "side": "CLOSE"},
                leg_b={"symbol": self.leader, "side": "NONE"},
                metadata={"exit_reason": "timeout"},
            )

        return None

    def _calculate_move(self, ticks: deque, seconds: int) -> Optional[float]:
        """Calculate price move % over last N seconds"""
        now = time.time()
        cutoff = now - seconds

        recent = [(ts, p) for ts, p in ticks if ts >= cutoff]
        if len(recent) < 2:
            return None

        start_price = recent[0][1]
        end_price = recent[-1][1]

        if start_price <= 0:
            return None

        return (end_price - start_price) / start_price * 100
