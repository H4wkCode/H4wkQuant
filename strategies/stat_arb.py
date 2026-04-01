"""
H4wkQuant - Statistical Arbitrage (Pairs Trading)
Main strategy: mean reversion on cointegrated pairs

BTC/ETH, SOL/ETH, BNB/ETH ratio tracking
Z-score > 2 entry, Z-score < 0.5 exit
Market neutral - no directional exposure
"""
import numpy as np
from typing import Optional, Dict, List
from collections import deque
from loguru import logger

from strategies.base import BaseStrategy
from models.spread import SpreadModel
from models.bayesian import BayesianModel
from models.edge import EdgeModel
from models.kelly import KellyModel
from shared.schemas.models import ArbSignal, ArbSignalAction, StrategyType
from shared.config.settings import settings


class StatArbStrategy(BaseStrategy):
    """
    Statistical Arbitrage via Pairs Trading.

    Pipeline:
    1. SpreadModel: Calculate z-score of price ratio
    2. Check cointegration (must pass ADF test)
    3. BayesianModel: Enhance signal with orderbook/funding/OI
    4. EdgeModel: Verify positive expected value after costs
    5. KellyModel: Optimal position sizing

    Entry: |z-score| > entry_threshold AND cointegrated AND edge > 0
    Exit:  |z-score| < exit_threshold OR z-score flips sign
    Stop:  |z-score| > 4 (spread blowing up = broken relationship)
    """

    def __init__(self, pairs: List[str] = None):
        super().__init__("stat_arb")

        arb = settings.arbitrage
        self.pairs = pairs or arb.pairs
        self.entry_threshold = arb.zscore_entry_threshold
        self.exit_threshold = arb.zscore_exit_threshold
        self.max_half_life = arb.half_life_max
        self.coint_pvalue = arb.cointegration_pvalue

        # Dynamic thresholds
        self.dynamic_enabled = arb.dynamic_threshold_enabled
        self.dyn_entry_min = arb.dynamic_entry_min
        self.dyn_entry_max = arb.dynamic_entry_max
        self.dyn_exit_min = arb.dynamic_exit_min
        self.dyn_exit_max = arb.dynamic_exit_max

        # Models
        self.spread_model = SpreadModel(
            lookback=arb.lookback_window,
            min_lookback=120,
            use_kalman=arb.kalman_enabled,
            kalman_process_variance=arb.kalman_process_variance,
            kalman_measurement_variance=arb.kalman_measurement_variance,
        )
        self.bayesian_model = BayesianModel()
        self.edge_model = EdgeModel(
            maker_fee=settings.risk.maker_fee,
            taker_fee=settings.risk.taker_fee,
        )
        self.kelly_model = KellyModel(fraction=arb.kelly_fraction)

        # Price history per pair
        self.price_history: Dict[str, Dict[str, deque]] = {}
        for pair in self.pairs:
            sym_a, sym_b = pair.split("/")
            self.price_history[pair] = {
                "a": deque(maxlen=arb.lookback_window),
                "b": deque(maxlen=arb.lookback_window),
            }

        # Trade history for Kelly
        self.trade_returns: List[float] = []

    def get_pairs(self) -> List[str]:
        return self.pairs

    def update_prices(self, pair_id: str, price_a: float, price_b: float):
        """Feed new price tick to history"""
        if pair_id in self.price_history:
            self.price_history[pair_id]["a"].append(price_a)
            self.price_history[pair_id]["b"].append(price_b)

    async def evaluate(self, market_data: Dict) -> Optional[ArbSignal]:
        """
        Evaluate a pair for entry signal.

        market_data expected keys:
            pair_id: str
            price_a: float
            price_b: float
            orderbook_a: dict (optional)
            orderbook_b: dict (optional)
            funding_rate_a: float (optional)
            funding_rate_b: float (optional)
            oi_change_a: float (optional)
            equity: float
        """
        pair_id = market_data["pair_id"]
        if pair_id not in self.price_history:
            return None

        # Update prices
        self.update_prices(pair_id, market_data["price_a"], market_data["price_b"])

        history = self.price_history[pair_id]
        if len(history["a"]) < self.spread_model.min_lookback:
            return None

        prices_a = np.array(history["a"])
        prices_b = np.array(history["b"])

        # 1. Spread calculation
        spread_result = self.spread_model.compute(prices_a, prices_b, pair_id=pair_id)

        # Dynamic thresholds based on half-life
        entry_thresh, _ = self._dynamic_thresholds(spread_result.half_life)

        # Check z-score threshold (must be between entry and blowup)
        if abs(spread_result.zscore) < entry_thresh:
            return None

        # Don't enter if z-score already blown up - would close immediately
        if abs(spread_result.zscore) > 8.0:
            logger.debug(f"{pair_id}: Z-score too high for entry ({spread_result.zscore:.2f} > 8.0)")
            return None

        # Check cointegration
        if not spread_result.is_cointegrated:
            logger.debug(f"{pair_id}: Not cointegrated (p={spread_result.coint_pvalue:.3f})")
            return None

        # Check half-life (must revert fast enough)
        if spread_result.half_life > self.max_half_life:
            logger.debug(f"{pair_id}: Half-life too long ({spread_result.half_life:.0f}m)")
            return None

        # 2. Bayesian enhancement (optional - use if orderbook data available)
        bayesian_confidence = 0.5
        if "orderbook_a" in market_data:
            bayes_a = self.bayesian_model.estimate(
                market_data["price_a"],
                market_data["orderbook_a"],
                funding_rate=market_data.get("funding_rate_a", 0),
                oi_change_pct=market_data.get("oi_change_a", 0),
            )
            bayesian_confidence = bayes_a.confidence

        # 3. Edge calculation
        equity = market_data.get("equity", settings.risk.paper_initial_balance)
        notional = min(equity * settings.risk.default_leverage / 3, settings.risk.max_leg_size_usd)

        edge_result = self.edge_model.calculate(
            zscore=spread_result.zscore,
            spread_std=spread_result.std,
            notional_per_leg=notional,
            use_limit_orders=True,
            funding_rate_a=market_data.get("funding_rate_a", 0),
            funding_rate_b=market_data.get("funding_rate_b", 0),
        )

        if not edge_result.is_profitable or edge_result.edge_ratio < 1.5:  # was 1.0 - higher profit expectation
            logger.debug(f"{pair_id}: Edge insufficient (net={edge_result.net_edge:.6f}, ratio={edge_result.edge_ratio:.2f})")
            return None

        # 4. Kelly sizing
        kelly_result = self.kelly_model.calculate(equity, self.trade_returns)
        position_size = min(kelly_result.position_size_usd, notional)

        # Determine direction
        sym_a, sym_b = pair_id.split("/")
        if spread_result.zscore > 0:
            # Spread is above mean -> short A, long B (expect reversion down)
            leg_a = {"symbol": sym_a, "side": "SHORT", "weight": 1.0}
            leg_b = {"symbol": sym_b, "side": "LONG", "weight": spread_result.hedge_ratio}
        else:
            # Spread is below mean -> long A, short B (expect reversion up)
            leg_a = {"symbol": sym_a, "side": "LONG", "weight": 1.0}
            leg_b = {"symbol": sym_b, "side": "SHORT", "weight": spread_result.hedge_ratio}

        # Combined confidence
        confidence = min(
            abs(spread_result.zscore) / 4.0,  # Z-score contribution
            1.0
        ) * 0.6 + bayesian_confidence * 0.4

        return ArbSignal(
            pair_id=pair_id,
            strategy=StrategyType.STAT_ARB,
            action=ArbSignalAction.OPEN,
            zscore=spread_result.zscore,
            edge_net=edge_result.net_edge,
            confidence=min(confidence, 1.0),
            leg_a=leg_a,
            leg_b=leg_b,
            kelly_size=kelly_result.fractional_kelly,
            position_size_usd=round(position_size, 2),
            metadata={
                "half_life": spread_result.half_life,
                "coint_pvalue": spread_result.coint_pvalue,
                "hedge_ratio": spread_result.hedge_ratio,
                "edge_ratio": edge_result.edge_ratio,
                "breakeven_zscore": edge_result.breakeven_zscore,
                "entry_threshold": round(entry_thresh, 2),
                "exit_threshold": round(self._dynamic_thresholds(spread_result.half_life)[1], 2),
            },
        )

    async def should_close(self, position_data: Dict, market_data: Dict) -> Optional[ArbSignal]:
        """
        Check if existing position should be closed.

        Close conditions:
        1. Z-score reverted to near 0 (profit target) - after min hold
        2. Z-score exceeded 4 (stop loss - relationship broken)
        3. Cointegration broke down - after min hold
        """
        pair_id = position_data["pair_id"]
        entry_zscore = position_data["entry_zscore"]

        history = self.price_history[pair_id]
        if len(history["a"]) < self.spread_model.min_lookback:
            return None

        prices_a = np.array(history["a"])
        prices_b = np.array(history["b"])
        spread_result = self.spread_model.compute(prices_a, prices_b, pair_id=pair_id)

        sym_a, sym_b = pair_id.split("/")
        exit_reason = None

        # Minimum hold time for non-emergency exits
        import time
        entry_time = position_data.get("entry_time", 0)
        hold_seconds = time.time() - entry_time if entry_time else 999
        min_hold = 900  # 15 minutes minimum hold (was 300s/5min - too short for reversion)

        # Dynamic exit threshold
        _, exit_thresh = self._dynamic_thresholds(spread_result.half_life)

        # 1. Z-score blowup (stop loss) - NO min hold, always honor
        if abs(spread_result.zscore) > 8.0:  # was 6.5 - wider movement range
            exit_reason = "zscore_blowup"

        # Below here: only after minimum hold time
        elif hold_seconds < min_hold:
            return None

        # 2. Mean reversion target hit (profit)
        elif abs(spread_result.zscore) < exit_thresh:
            exit_reason = "zscore_revert"

        # 3. Z-score flipped sign (crossed mean) - only close if profitable
        elif np.sign(spread_result.zscore) != np.sign(entry_zscore):
            # Z-score crossed zero: this is actually a profit signal for stat arb
            exit_reason = "zscore_cross"

        # 4. Cointegration broke
        elif not spread_result.is_cointegrated:
            exit_reason = "coint_break"

        if exit_reason is None:
            return None

        return ArbSignal(
            pair_id=pair_id,
            strategy=StrategyType.STAT_ARB,
            action=ArbSignalAction.CLOSE,
            zscore=spread_result.zscore,
            edge_net=0.0,
            confidence=1.0,
            leg_a={"symbol": sym_a, "side": "CLOSE"},
            leg_b={"symbol": sym_b, "side": "CLOSE"},
            metadata={"exit_reason": exit_reason},
        )

    def _dynamic_thresholds(self, half_life: float) -> tuple:
        """
        Compute dynamic entry/exit thresholds based on half-life.

        Short half-life → pair reverts fast → tighter entry OK, tighter exit
        Long half-life → pair reverts slow → need wider entry for safety

        Returns (entry_threshold, exit_threshold)
        """
        if not self.dynamic_enabled:
            return self.entry_threshold, self.exit_threshold

        # Normalize half-life to [0, 1] range within [1, max_half_life]
        hl_ratio = min(max((half_life - 1.0) / max(self.max_half_life - 1, 1), 0.0), 1.0)

        # Short HL → lower entry (1.5), Long HL → higher entry (3.0)
        entry = self.dyn_entry_min + hl_ratio * (self.dyn_entry_max - self.dyn_entry_min)

        # Short HL → tighter exit (0.3), Long HL → wider exit (0.8)
        exit_ = self.dyn_exit_min + hl_ratio * (self.dyn_exit_max - self.dyn_exit_min)

        return entry, exit_

    def record_trade_result(self, return_pct: float):
        """Record completed trade return for Kelly updates"""
        self.trade_returns.append(return_pct)
