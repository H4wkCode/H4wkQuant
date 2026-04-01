"""
H4wkQuant - Stoikov Market Making Model
r = s - q * gamma * sigma^2 * (T - t)

Inventory-aware limit order placement for efficient execution.
Based on Avellaneda-Stoikov (2008) model.
"""
import numpy as np
from typing import Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class StoikovQuote:
    bid_price: float
    ask_price: float
    spread: float
    mid_price: float
    inventory_skew: float  # How much inventory affects the quote
    reservation_price: float  # Inventory-adjusted fair price


class StoikovModel:
    """
    Avellaneda-Stoikov optimal market making / execution model.

    Reservation price:
        r = s - q * gamma * sigma^2 * (T - t)

    Optimal spread:
        delta = gamma * sigma^2 * (T - t) + (2/gamma) * ln(1 + gamma/k)

    Where:
        s = mid price
        q = inventory (positive = long, negative = short)
        gamma = risk aversion parameter (higher = more risk averse)
        sigma = volatility (per unit time)
        T - t = time remaining
        k = order book depth parameter

    For arbitrage execution:
    - We use this to place limit orders that minimize slippage
    - Inventory awareness prevents accumulation on one side
    - The spread widens when we have inventory to offload
    """

    def __init__(
        self,
        gamma: float = 0.1,  # Risk aversion
        k: float = 1.5,  # Order book depth
        dt: float = 1.0,  # Time step (normalized)
    ):
        self.gamma = gamma
        self.k = k
        self.dt = dt

    def quote(
        self,
        mid_price: float,
        volatility: float,
        inventory: float = 0.0,
        time_remaining: float = 1.0,
    ) -> StoikovQuote:
        """
        Calculate optimal bid/ask prices.

        Args:
            mid_price: Current mid price
            volatility: Price volatility (std dev per time unit, e.g. 1-minute returns std * price)
            inventory: Current position (+ = long, - = short)
            time_remaining: Normalized time remaining (0 to 1)
        """
        # Reservation price: inventory-adjusted fair value
        # r = s - q * gamma * sigma^2 * (T - t)
        sigma_sq = volatility ** 2
        reservation_price = mid_price - inventory * self.gamma * sigma_sq * time_remaining

        # Optimal spread
        # delta = gamma * sigma^2 * (T-t) + (2/gamma) * ln(1 + gamma/k)
        spread = (
            self.gamma * sigma_sq * time_remaining +
            (2.0 / self.gamma) * np.log(1 + self.gamma / self.k)
        )

        # Half-spread
        half_spread = spread / 2.0

        # Quote prices
        bid_price = reservation_price - half_spread
        ask_price = reservation_price + half_spread

        # Inventory skew: how much we adjust from mid
        inventory_skew = mid_price - reservation_price

        return StoikovQuote(
            bid_price=round(bid_price, 8),
            ask_price=round(ask_price, 8),
            spread=round(spread, 8),
            mid_price=mid_price,
            inventory_skew=round(inventory_skew, 8),
            reservation_price=round(reservation_price, 8),
        )

    def execution_price(
        self,
        mid_price: float,
        side: str,
        volatility: float,
        urgency: float = 0.5,
        inventory: float = 0.0,
    ) -> float:
        """
        Get optimal limit order price for execution.

        For arb entry/exit, we want aggressive enough to fill,
        but not so aggressive we give up edge.

        Args:
            mid_price: Current mid price
            side: "BUY" or "SELL"
            volatility: Price volatility
            urgency: 0 = patient (wide), 1 = aggressive (tight to mid)
            inventory: Current position
        """
        quote = self.quote(mid_price, volatility, inventory)

        if side == "BUY":
            # Interpolate between our bid and mid based on urgency
            price = quote.bid_price + urgency * (mid_price - quote.bid_price)
        else:
            # Interpolate between our ask and mid based on urgency
            price = quote.ask_price - urgency * (quote.ask_price - mid_price)

        return round(price, 8)

    def should_cancel_and_replace(
        self,
        current_order_price: float,
        new_optimal_price: float,
        min_price_change_pct: float = 0.01,
    ) -> bool:
        """
        Determine if an existing order should be cancelled and replaced.
        Avoid excessive cancellations.
        """
        if current_order_price == 0:
            return True

        change_pct = abs(new_optimal_price - current_order_price) / current_order_price * 100
        return change_pct > min_price_change_pct
