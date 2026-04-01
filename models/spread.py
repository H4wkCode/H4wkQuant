"""
H4wkQuant - Spread Model
Z-score calculation, cointegration test, half-life estimation
Core of statistical arbitrage
"""
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from loguru import logger

# Try to import statsmodels for proper ADF test
try:
    from statsmodels.tsa.stattools import adfuller
    _HAS_STATSMODELS = True
except (ImportError, TypeError):
    _HAS_STATSMODELS = False
    logger.warning("statsmodels not available, using simplified ADF test")

from models.kalman import KalmanFilter


@dataclass
class SpreadResult:
    ratio: float
    spread: float
    zscore: float
    mean: float
    std: float
    half_life: float
    is_cointegrated: bool
    coint_pvalue: float
    hedge_ratio: float


class SpreadModel:
    """
    Computes spread between two price series using log-ratio.
    Z-score = (spread - mean) / std

    Uses OLS hedge ratio for proper pair construction:
        spread = log(A) - beta * log(B)

    Cointegration: Engle-Granger (ADF on spread residuals)
    Half-life: Ornstein-Uhlenbeck mean reversion speed
    """

    def __init__(self, lookback: int = 480, min_lookback: int = 60,
                 use_kalman: bool = False,
                 kalman_process_variance: float = 0.5,
                 kalman_measurement_variance: float = 5.0):
        self.lookback = lookback
        self.min_lookback = min_lookback
        self.use_kalman = use_kalman
        self._kalman_q = kalman_process_variance
        self._kalman_r = kalman_measurement_variance
        self._kalman_filters: Dict[str, KalmanFilter] = {}

    def _get_kalman(self, pair_id: str) -> KalmanFilter:
        if pair_id not in self._kalman_filters:
            self._kalman_filters[pair_id] = KalmanFilter(
                process_variance=self._kalman_q,
                measurement_variance=self._kalman_r,
            )
        return self._kalman_filters[pair_id]

    def compute(
        self, prices_a: np.ndarray, prices_b: np.ndarray,
        pair_id: str = ""
    ) -> SpreadResult:
        """
        Compute spread, z-score, cointegration, and half-life.

        Args:
            prices_a: Price series of asset A (e.g. BTC)
            prices_b: Price series of asset B (e.g. ETH)
            pair_id: Pair identifier for per-pair Kalman state

        Returns:
            SpreadResult with all metrics
        """
        assert len(prices_a) == len(prices_b), "Price series must have equal length"
        assert len(prices_a) >= self.min_lookback, f"Need >= {self.min_lookback} data points"

        # Use last N points
        n = min(len(prices_a), self.lookback)
        a = prices_a[-n:]
        b = prices_b[-n:]

        # Log prices
        log_a = np.log(a)
        log_b = np.log(b)

        # OLS hedge ratio: log_a = alpha + beta * log_b + epsilon
        hedge_ratio = self._ols_hedge_ratio(log_a, log_b)

        # Spread = log(A) - beta * log(B)
        spread_series = log_a - hedge_ratio * log_b

        # Z-score
        mean = np.mean(spread_series)
        std = np.std(spread_series, ddof=1)
        current_spread = spread_series[-1]

        if std < 1e-10:
            zscore = 0.0
        else:
            zscore = (current_spread - mean) / std

        # Current ratio (for display)
        ratio = a[-1] / b[-1]

        # Cointegration test (ADF on spread)
        is_coint, pvalue = self._adf_test(spread_series)

        # Half-life of mean reversion
        half_life = self._half_life(spread_series)

        # Apply Kalman smoothing to half-life if enabled
        if self.use_kalman and pair_id and half_life != float('inf'):
            kalman = self._get_kalman(pair_id)
            kr = kalman.update(half_life)
            half_life = max(kr.filtered_value, 1.0)

        return SpreadResult(
            ratio=ratio,
            spread=current_spread,
            zscore=zscore,
            mean=mean,
            std=std,
            half_life=half_life,
            is_cointegrated=is_coint,
            coint_pvalue=pvalue,
            hedge_ratio=hedge_ratio,
        )

    def _ols_hedge_ratio(self, y: np.ndarray, x: np.ndarray) -> float:
        """Simple OLS: y = alpha + beta*x"""
        x_with_const = np.column_stack([np.ones(len(x)), x])
        # Normal equation: beta = (X'X)^-1 X'y
        try:
            coeffs = np.linalg.lstsq(x_with_const, y, rcond=None)[0]
            return coeffs[1]  # hedge ratio (beta)
        except np.linalg.LinAlgError:
            return 1.0  # fallback to 1:1

    def _adf_test(self, spread: np.ndarray, significance: float = 0.05) -> Tuple[bool, float]:
        """
        Augmented Dickey-Fuller test for stationarity.
        Uses statsmodels if available, otherwise falls back to simplified version.
        """
        if _HAS_STATSMODELS:
            return self._adf_test_statsmodels(spread, significance)
        return self._adf_test_simple(spread, significance)

    def _adf_test_statsmodels(self, spread: np.ndarray, significance: float = 0.05) -> Tuple[bool, float]:
        """
        Proper ADF test via statsmodels.adfuller.
        Returns continuous p-value (0.0001 - 1.0).
        """
        n = len(spread)
        if n < 20:
            return False, 1.0

        try:
            maxlag = min(int(np.sqrt(n)), 15)
            result = adfuller(spread, maxlag=maxlag, autolag='AIC', regression='c')
            t_stat = result[0]
            pvalue = result[1]
            # Clamp p-value to valid range
            pvalue = max(0.0001, min(1.0, pvalue))
            return pvalue < significance, pvalue
        except Exception as e:
            logger.debug(f"statsmodels ADF failed, falling back to simple: {e}")
            return self._adf_test_simple(spread, significance)

    def _adf_test_simple(self, spread: np.ndarray, significance: float = 0.05) -> Tuple[bool, float]:
        """
        Simplified ADF test without statsmodels dependency.
        Tests: delta_y = alpha + gamma*y_{t-1} + epsilon
        If gamma < 0 and significant -> stationary (cointegrated)
        """
        n = len(spread)
        if n < 20:
            return False, 1.0

        # First difference
        dy = np.diff(spread)
        y_lag = spread[:-1]

        # OLS: dy = alpha + gamma * y_lag
        x = np.column_stack([np.ones(len(y_lag)), y_lag])
        try:
            coeffs, residuals, _, _ = np.linalg.lstsq(x, dy, rcond=None)
        except np.linalg.LinAlgError:
            return False, 1.0

        gamma = coeffs[1]

        if gamma >= 0:
            return False, 1.0

        # Standard error of gamma
        residuals_vec = dy - x @ coeffs
        mse = np.sum(residuals_vec ** 2) / (n - 3)
        try:
            var_gamma = mse * np.linalg.inv(x.T @ x)[1, 1]
            se_gamma = np.sqrt(max(var_gamma, 1e-20))
        except np.linalg.LinAlgError:
            return False, 1.0

        # ADF t-statistic
        t_stat = gamma / se_gamma

        # Critical values (approximate, for n>100)
        # 1%: -3.43, 5%: -2.86, 10%: -2.57
        critical_5pct = -2.86

        # Approximate p-value using critical values
        if t_stat < -3.43:
            pvalue = 0.01
        elif t_stat < -2.86:
            pvalue = 0.05
        elif t_stat < -2.57:
            pvalue = 0.10
        else:
            pvalue = 0.50

        return t_stat < critical_5pct, pvalue

    def _half_life(self, spread: np.ndarray) -> float:
        """
        Ornstein-Uhlenbeck half-life estimation.
        dx = theta * (mu - x) * dt + sigma * dW
        half_life = -ln(2) / ln(1 + theta)

        Uses OLS: delta_spread = alpha + beta * spread_lag
        half_life = -ln(2) / beta
        """
        n = len(spread)
        if n < 20:
            return float('inf')

        dy = np.diff(spread)
        y_lag = spread[:-1]

        x = np.column_stack([np.ones(len(y_lag)), y_lag])
        try:
            coeffs = np.linalg.lstsq(x, dy, rcond=None)[0]
        except np.linalg.LinAlgError:
            return float('inf')

        beta = coeffs[1]

        if beta >= 0:
            return float('inf')  # Not mean reverting

        half_life = -np.log(2) / beta

        return max(half_life, 1.0)  # At least 1 period
