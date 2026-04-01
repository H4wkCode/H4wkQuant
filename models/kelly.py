"""
H4wkQuant - Kelly Criterion
f* = (b*p - q) / b

Optimal position sizing for arbitrage trades.
Uses fractional Kelly (0.25) for conservative sizing.
"""
import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class KellyResult:
    full_kelly: float  # Optimal fraction (full Kelly)
    fractional_kelly: float  # Adjusted fraction (conservative)
    position_size_usd: float  # Dollar size for given equity
    win_probability: float
    avg_win: float
    avg_loss: float
    edge: float  # Expected value per dollar risked
    enough_data: bool


class KellyModel:
    """
    Kelly Criterion for position sizing.

    f* = (b*p - q) / b

    Where:
        p = win probability
        q = loss probability (1-p)
        b = avg_win / avg_loss (odds ratio)

    We use 0.25 Kelly (quarter Kelly) for safety:
    - Full Kelly maximizes growth but has high volatility
    - Half Kelly: 75% of growth, 50% of variance
    - Quarter Kelly: ~50% of growth, 25% of variance

    For arb trading with $174 capital, quarter Kelly is appropriate.
    """

    def __init__(
        self,
        fraction: float = 0.25,
        min_trades: int = 20,
        max_fraction: float = 0.20,  # was 0.15 - sufficient position size with $500
        min_fraction: float = 0.05,  # was 0.02 - minimum 5% = $25 (with $500 balance)
    ):
        self.fraction = fraction
        self.min_trades = min_trades
        self.max_fraction = max_fraction
        self.min_fraction = min_fraction

    def calculate(
        self,
        equity: float,
        trade_history: List[float] = None,
        win_prob: float = None,
        avg_win: float = None,
        avg_loss: float = None,
    ) -> KellyResult:
        """
        Calculate Kelly-optimal position size.

        Args:
            equity: Current account equity in USD
            trade_history: List of PnL values from past trades (preferred)
            win_prob: Override win probability (0-1)
            avg_win: Override average win amount
            avg_loss: Override average loss amount (positive number)
        """
        enough_data = True

        if trade_history and len(trade_history) >= self.min_trades:
            # Calculate from actual trade history
            wins = [t for t in trade_history if t > 0]
            losses = [t for t in trade_history if t <= 0]

            if not wins or not losses:
                return self._default_result(equity)

            p = len(wins) / len(trade_history)
            w = np.mean(wins)
            l_val = abs(np.mean(losses))
        elif win_prob is not None and avg_win is not None and avg_loss is not None:
            # Use provided parameters
            p = win_prob
            w = avg_win
            l_val = abs(avg_loss)
            enough_data = False
        else:
            return self._default_result(equity)

        q = 1.0 - p

        if l_val < 1e-10:
            return self._default_result(equity)

        # Odds ratio
        b = w / l_val

        # Kelly formula: f* = (b*p - q) / b
        full_kelly = (b * p - q) / b

        # If Kelly is negative, we have negative edge -> don't trade
        if full_kelly <= 0:
            return KellyResult(
                full_kelly=full_kelly,
                fractional_kelly=0.0,
                position_size_usd=0.0,
                win_probability=p,
                avg_win=w,
                avg_loss=l_val,
                edge=b * p - q,
                enough_data=enough_data,
            )

        # Apply fraction
        frac_kelly = full_kelly * self.fraction

        # Clamp to safety bounds
        frac_kelly = np.clip(frac_kelly, self.min_fraction, self.max_fraction)

        # Dollar position size
        position_size = equity * frac_kelly

        return KellyResult(
            full_kelly=full_kelly,
            fractional_kelly=frac_kelly,
            position_size_usd=round(position_size, 2),
            win_probability=p,
            avg_win=w,
            avg_loss=l_val,
            edge=b * p - q,
            enough_data=enough_data,
        )

    def _default_result(self, equity: float) -> KellyResult:
        """Default conservative sizing when insufficient data"""
        min_position = 5.0
        default_frac = max(self.min_fraction, min_position / equity if equity > 0 else self.min_fraction)
        default_frac = min(default_frac, self.max_fraction)
        return KellyResult(
            full_kelly=0.0,
            fractional_kelly=default_frac,
            position_size_usd=round(max(equity * default_frac, min_position), 2),
            win_probability=0.5,
            avg_win=0.0,
            avg_loss=0.0,
            edge=0.0,
            enough_data=False,
        )

    def kelly_from_edge(self, edge_ratio: float, equity: float) -> float:
        """
        Simplified Kelly from edge ratio.
        When we know the edge/cost ratio directly.

        f* = edge / variance (simplified)
        For arb: f* ~ edge_ratio / (1 + edge_ratio)
        """
        if edge_ratio <= 0:
            return 0.0

        kelly = edge_ratio / (1 + edge_ratio)
        frac = kelly * self.fraction
        frac = np.clip(frac, self.min_fraction, self.max_fraction)

        return round(equity * frac, 2)
