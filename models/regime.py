"""
H4wkQuant - Market Regime Detection
Classifies market regime based on BTC rolling volatility percentile.
Blocks new position opens during HIGH/EXTREME regimes.
"""
import time
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from collections import deque
from loguru import logger


class MarketRegime(str, Enum):
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


@dataclass
class RegimeResult:
    regime: MarketRegime
    current_volatility: float
    percentile: float
    allow_new_positions: bool
    vol_window_size: int
    history_window_size: int


class RegimeDetector:
    """
    Detects market regime from BTC price ticks.

    1. Compute rolling volatility (std of log-returns) over vol_window
    2. Compare current vol to historical percentile over history_window
    3. Classify: LOW (<25th) / NORMAL (25-75th) / HIGH (75-90th) / EXTREME (>90th)
    """

    def __init__(
        self,
        vol_window: int = 60,
        history_window: int = 1440,
        high_percentile: float = 75.0,
        extreme_percentile: float = 90.0,
    ):
        self.vol_window = vol_window
        self.history_window = history_window
        self.high_percentile = high_percentile
        self.extreme_percentile = extreme_percentile

        # Store recent prices for volatility calculation
        self.prices: deque = deque(maxlen=max(vol_window + 1, 100))
        # Store rolling volatility history for percentile
        self.vol_history: deque = deque(maxlen=history_window)
        # Current regime
        self.current_regime: MarketRegime = MarketRegime.NORMAL
        self._last_result: RegimeResult | None = None
        self._regime_changed_at: float = 0.0  # timestamp of last regime change
        self._min_regime_hold: float = 60.0  # minimum seconds before regime can change

    def update(self, price: float) -> RegimeResult | None:
        """
        Feed a new BTC price tick. Returns RegimeResult when enough data,
        or None if still warming up.
        """
        self.prices.append(price)

        if len(self.prices) < self.vol_window + 1:
            return None

        # Calculate rolling volatility (std of log-returns)
        prices_arr = np.array(list(self.prices))
        log_returns = np.diff(np.log(prices_arr[-self.vol_window - 1:]))
        current_vol = float(np.std(log_returns))

        self.vol_history.append(current_vol)

        if len(self.vol_history) < 60:
            # Need minimum history for percentile
            return None

        # Calculate percentile of current vol in history
        vol_arr = np.array(self.vol_history)
        percentile = float(np.sum(vol_arr <= current_vol) / len(vol_arr) * 100)

        # Classify regime with 5% hysteresis to prevent oscillation
        hysteresis = 5.0
        if self.current_regime == MarketRegime.EXTREME:
            if percentile < self.extreme_percentile - hysteresis:
                regime = MarketRegime.HIGH if percentile >= self.high_percentile - hysteresis else MarketRegime.NORMAL
            else:
                regime = MarketRegime.EXTREME
        elif self.current_regime == MarketRegime.HIGH:
            if percentile >= self.extreme_percentile:
                regime = MarketRegime.EXTREME
            elif percentile < self.high_percentile - hysteresis:
                regime = MarketRegime.NORMAL
            else:
                regime = MarketRegime.HIGH
        elif self.current_regime == MarketRegime.LOW:
            if percentile > 25.0 + hysteresis:
                regime = MarketRegime.NORMAL
            else:
                regime = MarketRegime.LOW
        else:  # NORMAL
            if percentile >= self.extreme_percentile:
                regime = MarketRegime.EXTREME
            elif percentile >= self.high_percentile:
                regime = MarketRegime.HIGH
            elif percentile < 25.0:
                regime = MarketRegime.LOW
            else:
                regime = MarketRegime.NORMAL

        # Minimum hold time: don't change regime too frequently
        now = time.time()
        if regime != self.current_regime:
            if now - self._regime_changed_at >= self._min_regime_hold:
                logger.info(f"Regime change: {self.current_regime.value} → {regime.value} (vol={current_vol:.6f}, pct={percentile:.1f}%)")
                self.current_regime = regime
                self._regime_changed_at = now
            else:
                regime = self.current_regime  # keep current regime

        allow = regime in (MarketRegime.LOW, MarketRegime.NORMAL)

        self._last_result = RegimeResult(
            regime=regime,
            current_volatility=current_vol,
            percentile=percentile,
            allow_new_positions=allow,
            vol_window_size=len(self.prices),
            history_window_size=len(self.vol_history),
        )
        return self._last_result

    @property
    def last_result(self) -> RegimeResult | None:
        return self._last_result

    def to_dict(self) -> dict:
        """Serialize current state for Redis caching."""
        if self._last_result is None:
            return {
                "regime": MarketRegime.NORMAL.value,
                "volatility": 0.0,
                "percentile": 50.0,
                "allow_new_positions": True,
                "warming_up": True,
            }
        r = self._last_result
        return {
            "regime": r.regime.value,
            "volatility": round(r.current_volatility, 8),
            "percentile": round(r.percentile, 1),
            "allow_new_positions": r.allow_new_positions,
            "warming_up": False,
        }
