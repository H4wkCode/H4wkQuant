"""
H4wkQuant - Multi-Timeframe Candle Aggregator
Aggregates tick data into OHLC candles (5m, 15m).
"""
import time
from typing import Dict, List, Optional
from collections import deque
from dataclasses import dataclass

import numpy as np
from loguru import logger


@dataclass
class Candle:
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    timestamp: float = 0.0


class CandleAggregator:
    """Aggregates tick prices into OHLC candles for multiple timeframes."""

    def __init__(self, timeframes: List[int] = None, max_candles: int = 200):
        """
        timeframes: list of intervals in seconds (e.g. [300, 900] for 5m, 15m)
        """
        self.timeframes = timeframes or [300, 900]  # 5m, 15m
        self.max_candles = max_candles

        # Per-symbol, per-timeframe storage
        # {symbol: {tf_seconds: {"candles": deque, "current": Candle, "bucket_start": float}}}
        self._data: Dict[str, Dict[int, dict]] = {}

    def _ensure_symbol(self, symbol: str):
        if symbol not in self._data:
            self._data[symbol] = {}
            for tf in self.timeframes:
                self._data[symbol][tf] = {
                    "candles": deque(maxlen=self.max_candles),
                    "current": None,
                    "bucket_start": 0,
                }

    def update(self, symbol: str, price: float, volume: float = 0.0, timestamp: float = None):
        """Feed a tick price. Returns list of (timeframe, candle) for any completed candles."""
        ts = timestamp or time.time()
        self._ensure_symbol(symbol)
        completed = []

        for tf in self.timeframes:
            state = self._data[symbol][tf]
            bucket = int(ts // tf) * tf

            if state["current"] is None:
                # First tick
                state["current"] = Candle(open=price, high=price, low=price, close=price, volume=volume, timestamp=bucket)
                state["bucket_start"] = bucket
            elif bucket > state["bucket_start"]:
                # New bucket - finalize current candle
                state["candles"].append(state["current"])
                completed.append((tf, state["current"]))
                state["current"] = Candle(open=price, high=price, low=price, close=price, volume=volume, timestamp=bucket)
                state["bucket_start"] = bucket
            else:
                # Update current candle
                c = state["current"]
                c.high = max(c.high, price)
                c.low = min(c.low, price)
                c.close = price
                c.volume += volume

        return completed

    def get_closes(self, symbol: str, timeframe: int) -> Optional[np.ndarray]:
        """Get array of close prices for a symbol and timeframe."""
        if symbol not in self._data or timeframe not in self._data[symbol]:
            return None
        candles = self._data[symbol][timeframe]["candles"]
        if len(candles) < 10:
            return None
        return np.array([c.close for c in candles])

    def get_candle_count(self, symbol: str, timeframe: int) -> int:
        if symbol not in self._data or timeframe not in self._data[symbol]:
            return 0
        return len(self._data[symbol][timeframe]["candles"])


class MultiTimeframeAnalyzer:
    """Checks cointegration across multiple timeframes for a pair."""

    def __init__(self, timeframes: List[int] = None, min_agreement: int = 2):
        self.timeframes = timeframes or [60, 300, 900]  # 1m, 5m, 15m
        self.min_agreement = min_agreement
        self.aggregator = CandleAggregator(timeframes=[300, 900])  # 1m handled by main history

    def update_tick(self, symbol: str, price: float, volume: float = 0.0):
        """Feed tick to aggregator."""
        self.aggregator.update(symbol, price, volume)

    def check_cointegration(self, pair_id: str, prices_a_1m: np.ndarray, prices_b_1m: np.ndarray,
                             spread_model) -> dict:
        """
        Check cointegration across timeframes.
        Returns dict with per-TF results and overall agreement.
        """
        sym_a, sym_b = pair_id.split("/")
        results = {}

        # 1m - use provided price history
        if len(prices_a_1m) >= 60:
            try:
                r = spread_model.compute(prices_a_1m, prices_b_1m, pair_id=pair_id)
                results["1m"] = {
                    "cointegrated": r.is_cointegrated,
                    "pvalue": round(r.coint_pvalue, 4),
                    "zscore": round(r.zscore, 4),
                }
            except Exception:
                results["1m"] = {"cointegrated": False, "pvalue": 1.0, "zscore": 0}
        else:
            results["1m"] = {"cointegrated": False, "pvalue": 1.0, "zscore": 0}

        # 5m and 15m - use aggregated candles
        for tf in [300, 900]:
            tf_label = "5m" if tf == 300 else "15m"
            closes_a = self.aggregator.get_closes(sym_a, tf)
            closes_b = self.aggregator.get_closes(sym_b, tf)

            if closes_a is not None and closes_b is not None:
                min_len = min(len(closes_a), len(closes_b))
                if min_len >= 20:
                    try:
                        r = spread_model.compute(closes_a[-min_len:], closes_b[-min_len:], pair_id=f"{pair_id}_{tf_label}")
                        results[tf_label] = {
                            "cointegrated": r.is_cointegrated,
                            "pvalue": round(r.coint_pvalue, 4),
                            "zscore": round(r.zscore, 4),
                        }
                        continue
                    except Exception:
                        pass
            results[tf_label] = {"cointegrated": False, "pvalue": 1.0, "zscore": 0}

        # Count agreements
        coint_count = sum(1 for r in results.values() if r["cointegrated"])
        results["agreement"] = coint_count
        results["passes"] = coint_count >= self.min_agreement

        return results
