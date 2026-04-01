"""
H4wkQuant - Edge Calculator
EV_net = q - p - c

Calculates net expected value after ALL costs:
- Commission (maker/taker fees, both legs)
- Slippage (estimated from orderbook depth)
- Funding cost (if holding through funding period)
"""
from dataclasses import dataclass
from loguru import logger


@dataclass
class EdgeResult:
    gross_edge: float  # Raw spread profit potential
    commission_cost: float  # Total fees for both legs (entry + exit)
    slippage_cost: float  # Estimated slippage
    funding_cost: float  # Estimated funding cost if applicable
    total_cost: float  # Sum of all costs
    net_edge: float  # gross_edge - total_cost
    edge_ratio: float  # net_edge / total_cost (how many costs we earn)
    is_profitable: bool  # net_edge > 0
    breakeven_zscore: float  # Z-score needed for breakeven


class EdgeModel:
    """
    Edge calculator for arbitrage trades.

    For a stat arb with z-score entry at Z and exit at 0:
        gross_edge = Z * sigma * notional_value (approximately)

    Round-trip costs (2 legs, entry + exit = 4 orders):
        - Limit orders: 4 * maker_fee * notional
        - Market orders: 4 * taker_fee * notional
        - Slippage: estimated from spread + depth

    Trade only when net_edge > 0.
    """

    def __init__(
        self,
        maker_fee: float = 0.0002,  # 0.02% Binance Futures maker
        taker_fee: float = 0.0004,  # 0.04% Binance Futures taker
        default_slippage_bps: float = 1.0,  # 0.01% default slippage per leg
    ):
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee
        self.default_slippage_bps = default_slippage_bps / 10000

    def calculate(
        self,
        zscore: float,
        spread_std: float,
        notional_per_leg: float,
        use_limit_orders: bool = True,
        holding_periods_8h: float = 0.0,
        funding_rate_a: float = 0.0,
        funding_rate_b: float = 0.0,
        slippage_bps: float = None,
    ) -> EdgeResult:
        """
        Calculate net edge for an arbitrage trade.

        Args:
            zscore: Current z-score of spread (entry signal)
            spread_std: Standard deviation of spread (in log terms)
            notional_per_leg: USD value per leg
            use_limit_orders: True for maker fees, False for taker
            holding_periods_8h: Expected holding time in 8h periods
            funding_rate_a: Funding rate for asset A
            funding_rate_b: Funding rate for asset B
            slippage_bps: Override slippage estimate (basis points)
        """
        # Gross edge: when z-score reverts from Z to 0
        # The profit in spread terms = Z * sigma
        # In dollar terms (approximately): Z * sigma * notional
        # Since spread is in log terms: profit_pct = abs(zscore) * spread_std
        gross_edge_pct = abs(zscore) * spread_std
        gross_edge = gross_edge_pct * notional_per_leg * 2  # 2 legs profit

        # Commission: 2 legs * 2 sides (entry + exit) = 4 orders
        fee_rate = self.maker_fee if use_limit_orders else self.taker_fee
        commission_cost = 4 * fee_rate * notional_per_leg

        # Slippage: per leg, entry + exit = 4 slippage events
        slip = (slippage_bps / 10000) if slippage_bps else self.default_slippage_bps
        slippage_cost = 4 * slip * notional_per_leg

        # Funding cost: holding through funding periods
        funding_cost = 0.0
        if holding_periods_8h > 0:
            # We pay/receive funding on both legs
            # For market-neutral: net funding = |funding_a| + |funding_b| (worst case)
            funding_cost = holding_periods_8h * (
                abs(funding_rate_a) + abs(funding_rate_b)
            ) * notional_per_leg

        total_cost = commission_cost + slippage_cost + funding_cost
        net_edge = gross_edge - total_cost

        # Edge ratio: how many times the cost we earn
        edge_ratio = net_edge / total_cost if total_cost > 0 else float('inf')

        # Breakeven z-score: what Z do we need to cover costs?
        if spread_std > 0 and notional_per_leg > 0:
            breakeven_zscore = total_cost / (spread_std * notional_per_leg * 2)
        else:
            breakeven_zscore = float('inf')

        return EdgeResult(
            gross_edge=gross_edge,
            commission_cost=commission_cost,
            slippage_cost=slippage_cost,
            funding_cost=funding_cost,
            total_cost=total_cost,
            net_edge=net_edge,
            edge_ratio=edge_ratio,
            is_profitable=net_edge > 0,
            breakeven_zscore=breakeven_zscore,
        )

    def min_notional_for_profit(
        self,
        zscore: float,
        spread_std: float,
        use_limit_orders: bool = True,
    ) -> float:
        """
        Calculate minimum notional per leg to be profitable.
        With fixed costs (slippage), there's a minimum size threshold.
        """
        fee_rate = self.maker_fee if use_limit_orders else self.taker_fee
        gross_pct = abs(zscore) * spread_std

        # gross - 4*fee - 4*slip > 0
        # gross_pct * 2N > 4 * fee_rate * N + 4 * slip * N
        # This simplifies to: gross_pct * 2 > 4 * (fee_rate + slip)
        # So profitability is scale-independent for percentage-based costs!
        # The minimum is really just the exchange minimum ($5)
        min_profitable = 2 * (fee_rate + self.default_slippage_bps) / (gross_pct + 1e-10)

        return max(min_profitable, 5.0)  # At least $5 (Binance min)
