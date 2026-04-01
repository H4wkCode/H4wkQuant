"""
H4wkQuant - Portfolio Construction & Correlation Manager
Prevents opening highly correlated pairs simultaneously.
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque
from loguru import logger


class PortfolioOptimizer:
    """
    Tracks per-pair return series and rolling correlation matrix.
    Rejects new pairs if |correlation| > threshold with existing positions.
    """

    def __init__(self, max_correlation: float = 0.7, lookback: int = 120):
        self.max_correlation = max_correlation
        self.lookback = lookback

        # pair_id -> deque of spread returns
        self.spread_returns: Dict[str, deque] = {}

        # Cache correlation matrix
        self._corr_matrix: Dict[str, Dict[str, float]] = {}

    def update_spread(self, pair_id: str, spread_value: float):
        """Feed current spread value for a pair."""
        if pair_id not in self.spread_returns:
            self.spread_returns[pair_id] = deque(maxlen=self.lookback)
        self.spread_returns[pair_id].append(spread_value)

    def check_correlation(self, new_pair: str, existing_pairs: List[str]) -> Tuple[bool, Dict[str, float]]:
        """
        Check if new_pair is too correlated with existing positions.

        Returns:
            (allowed: bool, correlations: {pair: corr_value})
        """
        if not existing_pairs:
            return True, {}

        new_returns = self.spread_returns.get(new_pair)
        if not new_returns or len(new_returns) < 30:
            # Not enough data - allow by default
            return True, {}

        new_arr = np.array(new_returns)
        correlations = {}

        for pair in existing_pairs:
            pair_returns = self.spread_returns.get(pair)
            if not pair_returns or len(pair_returns) < 30:
                continue

            pair_arr = np.array(pair_returns)
            min_len = min(len(new_arr), len(pair_arr))
            if min_len < 20:
                continue

            try:
                corr = np.corrcoef(new_arr[-min_len:], pair_arr[-min_len:])[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                correlations[pair] = round(corr, 4)
            except Exception:
                correlations[pair] = 0.0

        # Check if any correlation exceeds threshold
        blocked = any(abs(c) > self.max_correlation for c in correlations.values())
        if blocked:
            max_corr_pair = max(correlations, key=lambda p: abs(correlations[p]))
            logger.info(
                f"Portfolio: {new_pair} blocked - corr={correlations[max_corr_pair]:.3f} with {max_corr_pair}"
            )

        return not blocked, correlations

    def compute_correlation_matrix(self, pairs: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Compute full correlation matrix for given pairs (or all tracked pairs)."""
        pairs = pairs or list(self.spread_returns.keys())
        matrix = {}

        for i, pair_a in enumerate(pairs):
            matrix[pair_a] = {}
            returns_a = self.spread_returns.get(pair_a)
            if not returns_a or len(returns_a) < 20:
                for pair_b in pairs:
                    matrix[pair_a][pair_b] = 1.0 if pair_a == pair_b else 0.0
                continue

            arr_a = np.array(returns_a)
            for pair_b in pairs:
                if pair_a == pair_b:
                    matrix[pair_a][pair_b] = 1.0
                    continue

                returns_b = self.spread_returns.get(pair_b)
                if not returns_b or len(returns_b) < 20:
                    matrix[pair_a][pair_b] = 0.0
                    continue

                arr_b = np.array(returns_b)
                min_len = min(len(arr_a), len(arr_b))
                try:
                    corr = np.corrcoef(arr_a[-min_len:], arr_b[-min_len:])[0, 1]
                    matrix[pair_a][pair_b] = float(round(corr, 4)) if not np.isnan(corr) else 0.0
                except Exception:
                    matrix[pair_a][pair_b] = 0.0

        self._corr_matrix = matrix
        return matrix

    def get_cached_matrix(self) -> Dict[str, Dict[str, float]]:
        return self._corr_matrix
