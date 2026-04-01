"""
H4wkQuant - Bayesian Fair Value Model
P(H|D) = P(D|H) * P(H) / P(D)

Combines orderbook imbalance, volume, funding rate, and OI
to estimate fair value and detect mispricing.
"""
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class BayesianResult:
    fair_price: float
    current_price: float
    mispricing_pct: float
    orderbook_imbalance: float  # -1 (sell pressure) to 1 (buy pressure)
    volume_signal: float  # -1 to 1
    funding_signal: float  # -1 to 1
    oi_signal: float  # -1 to 1
    posterior_long: float  # P(price up)
    posterior_short: float  # P(price down)
    confidence: float  # 0 to 1


class BayesianModel:
    """
    Bayesian fair value estimator using multiple evidence sources.

    Prior: 50/50 (no directional bias - market neutral)
    Evidence:
        1. Orderbook imbalance → P(D|H) for buy/sell pressure
        2. Volume anomaly → confirms or denies move
        3. Funding rate → crowded trade signal
        4. OI change → new money entering/exiting

    Output: Posterior probability of price move direction + fair value estimate
    """

    def __init__(
        self,
        ob_weight: float = 0.35,
        volume_weight: float = 0.25,
        funding_weight: float = 0.25,
        oi_weight: float = 0.15,
    ):
        self.ob_weight = ob_weight
        self.volume_weight = volume_weight
        self.funding_weight = funding_weight
        self.oi_weight = oi_weight

    def estimate(
        self,
        current_price: float,
        orderbook: Dict,
        volume_ratio: float = 1.0,
        funding_rate: float = 0.0,
        oi_change_pct: float = 0.0,
    ) -> BayesianResult:
        """
        Estimate fair value and directional probabilities.

        Args:
            current_price: Current market price
            orderbook: {"bids": [[price, qty], ...], "asks": [[price, qty], ...]}
            volume_ratio: current_volume / avg_volume (>1 = above average)
            funding_rate: Current funding rate (positive = longs pay shorts)
            oi_change_pct: OI change % in last period
        """
        # 1. Orderbook imbalance
        ob_imbalance = self._orderbook_imbalance(orderbook)
        ob_fair_price = self._orderbook_fair_price(orderbook, current_price)

        # 2. Volume signal
        volume_signal = self._volume_signal(volume_ratio)

        # 3. Funding rate signal (contrarian: high funding = crowded, expect reversion)
        funding_signal = self._funding_signal(funding_rate)

        # 4. OI signal
        oi_signal = self._oi_signal(oi_change_pct)

        # Bayesian update: combine signals
        # Each signal contributes P(D|H_long) and P(D|H_short)
        prior_long = 0.5
        prior_short = 0.5

        # Likelihood for each evidence
        l_long, l_short = self._combined_likelihood(
            ob_imbalance, volume_signal, funding_signal, oi_signal
        )

        # Posterior
        evidence = l_long * prior_long + l_short * prior_short
        if evidence < 1e-10:
            posterior_long = 0.5
            posterior_short = 0.5
        else:
            posterior_long = (l_long * prior_long) / evidence
            posterior_short = (l_short * prior_short) / evidence

        # Fair price: weighted average of current and OB-implied
        fair_price = 0.7 * current_price + 0.3 * ob_fair_price

        mispricing_pct = (fair_price - current_price) / current_price * 100

        # Confidence: how far from 50/50
        confidence = abs(posterior_long - posterior_short)

        return BayesianResult(
            fair_price=fair_price,
            current_price=current_price,
            mispricing_pct=mispricing_pct,
            orderbook_imbalance=ob_imbalance,
            volume_signal=volume_signal,
            funding_signal=funding_signal,
            oi_signal=oi_signal,
            posterior_long=posterior_long,
            posterior_short=posterior_short,
            confidence=confidence,
        )

    def _orderbook_imbalance(self, orderbook: Dict) -> float:
        """
        Calculate bid/ask imbalance from orderbook.
        Returns: -1 (all asks, sell pressure) to 1 (all bids, buy pressure)
        """
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if not bids or not asks:
            return 0.0

        # Sum top N levels (quantity-weighted)
        n_levels = min(10, len(bids), len(asks))
        bid_volume = sum(float(b[1]) for b in bids[:n_levels])
        ask_volume = sum(float(a[1]) for a in asks[:n_levels])

        total = bid_volume + ask_volume
        if total < 1e-10:
            return 0.0

        return (bid_volume - ask_volume) / total

    def _orderbook_fair_price(self, orderbook: Dict, current_price: float) -> float:
        """Volume-weighted mid price from orderbook"""
        bids = orderbook.get("bids", [])
        asks = orderbook.get("asks", [])

        if not bids or not asks:
            return current_price

        n = min(5, len(bids), len(asks))

        bid_vwap_num = sum(float(b[0]) * float(b[1]) for b in bids[:n])
        bid_vwap_den = sum(float(b[1]) for b in bids[:n])
        ask_vwap_num = sum(float(a[0]) * float(a[1]) for a in asks[:n])
        ask_vwap_den = sum(float(a[1]) for a in asks[:n])

        if bid_vwap_den < 1e-10 or ask_vwap_den < 1e-10:
            return current_price

        bid_vwap = bid_vwap_num / bid_vwap_den
        ask_vwap = ask_vwap_num / ask_vwap_den

        # Weight by volume
        total_vol = bid_vwap_den + ask_vwap_den
        fair = (bid_vwap * bid_vwap_den + ask_vwap * ask_vwap_den) / total_vol

        return fair

    def _volume_signal(self, volume_ratio: float) -> float:
        """
        Volume ratio signal: high volume = confirmation of move
        ratio > 2 = strong signal, ratio < 0.5 = weak/fade
        Returns -1 to 1 (negative means fade/revert)
        """
        if volume_ratio > 3.0:
            return 0.8  # Very high volume = strong directional move
        elif volume_ratio > 1.5:
            return 0.4
        elif volume_ratio > 0.8:
            return 0.0  # Normal
        else:
            return -0.3  # Low volume = potential fade

    def _funding_signal(self, funding_rate: float) -> float:
        """
        Funding rate contrarian signal.
        High positive funding = longs are crowded -> bearish signal
        High negative funding = shorts are crowded -> bullish signal
        Returns: -1 to 1 (positive = bullish for price)
        """
        # Normalize: typical funding is -0.01% to 0.05%
        # Extreme: > 0.1% or < -0.05%
        if abs(funding_rate) < 0.0001:
            return 0.0

        # Contrarian: high funding -> expect price to move against the crowd
        signal = -np.clip(funding_rate * 1000, -1.0, 1.0)
        return float(signal)

    def _oi_signal(self, oi_change_pct: float) -> float:
        """
        Open Interest change signal.
        Rising OI + price move = trend confirmation
        Falling OI = position unwinding
        """
        return float(np.clip(oi_change_pct / 5.0, -1.0, 1.0))

    def _combined_likelihood(
        self, ob: float, vol: float, funding: float, oi: float
    ) -> tuple:
        """
        Combine evidence into likelihood P(D|H_long) and P(D|H_short).
        Each signal votes for long or short.
        """
        # Weighted composite signal (-1 to 1)
        composite = (
            self.ob_weight * ob +
            self.volume_weight * vol +
            self.funding_weight * funding +
            self.oi_weight * oi
        )

        # Convert to probability-like likelihoods
        # sigmoid-ish mapping: composite -> [0.2, 0.8]
        l_long = 0.5 + 0.3 * np.clip(composite, -1, 1)
        l_short = 1.0 - l_long

        return float(l_long), float(l_short)
