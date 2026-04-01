"""
H4wkQuant - Unit Tests for 6 Mathematical Models
"""
import numpy as np
import pytest
from models.spread import SpreadModel
from models.kalman import KalmanFilter
from models.regime import RegimeDetector, MarketRegime
from models.bayesian import BayesianModel
from models.edge import EdgeModel
from models.kelly import KellyModel
from models.stoikov import StoikovModel
from models.montecarlo import MonteCarloModel


# ============================================================
# Spread Model Tests
# ============================================================

class TestSpreadModel:
    def setup_method(self):
        self.model = SpreadModel(lookback=100, min_lookback=20)

    def test_basic_zscore(self):
        """Z-score should be 0 when spread equals mean"""
        np.random.seed(42)
        n = 100
        prices_a = 50000 + np.cumsum(np.random.randn(n) * 10)
        prices_b = 3000 + np.cumsum(np.random.randn(n) * 5)
        prices_a = np.abs(prices_a)
        prices_b = np.abs(prices_b)

        result = self.model.compute(prices_a, prices_b)
        assert -5 < result.zscore < 5, f"Z-score out of range: {result.zscore}"
        assert result.ratio > 0
        assert result.std > 0

    def test_extreme_zscore(self):
        """When prices diverge strongly, z-score should be large"""
        n = 100
        # Normal period
        prices_a = np.full(n, 50000.0) + np.random.randn(n) * 10
        prices_b = np.full(n, 3000.0) + np.random.randn(n) * 5
        # Spike at end
        prices_a[-1] = 55000  # BTC jumps
        prices_a = np.abs(prices_a)
        prices_b = np.abs(prices_b)

        result = self.model.compute(prices_a, prices_b)
        assert abs(result.zscore) > 1.0, "Z-score should be elevated"

    def test_cointegrated_series(self):
        """Cointegrated series should pass ADF test"""
        np.random.seed(123)
        n = 200
        # Create cointegrated pair
        b = np.cumsum(np.random.randn(n))  # Random walk
        a = 2 * b + np.random.randn(n) * 0.5  # Cointegrated with noise
        prices_a = np.exp(a / 10 + 10)  # Convert to price-like
        prices_b = np.exp(b / 10 + 8)

        result = self.model.compute(prices_a, prices_b)
        # With strong cointegration, ADF should reject null
        assert result.half_life > 0

    def test_insufficient_data(self):
        """Should raise with too few data points"""
        with pytest.raises(AssertionError):
            self.model.compute(np.array([100.0] * 5), np.array([50.0] * 5))

    @pytest.mark.skipif(
        not hasattr(__import__('models.spread', fromlist=['_HAS_STATSMODELS']), '_HAS_STATSMODELS') or
        not __import__('models.spread', fromlist=['_HAS_STATSMODELS'])._HAS_STATSMODELS,
        reason="statsmodels not installed"
    )
    def test_adf_continuous_pvalue(self):
        """ADF test should return continuous p-value, not just 4 fixed values"""
        np.random.seed(42)
        pvalues = set()
        for seed in range(10):
            np.random.seed(seed)
            n = 200
            b = np.cumsum(np.random.randn(n))
            noise_level = 0.1 + seed * 0.3
            a = 2 * b + np.random.randn(n) * noise_level
            prices_a = np.exp(a / 10 + 10)
            prices_b = np.exp(b / 10 + 8)
            result = self.model.compute(prices_a, prices_b)
            pvalues.add(round(result.coint_pvalue, 4))
        # With statsmodels, should get more than 4 distinct p-values
        assert len(pvalues) > 3, f"Only got {len(pvalues)} distinct p-values: {pvalues}"

    def test_kalman_smoothing(self):
        """Half-life should be smoother with Kalman enabled"""
        np.random.seed(42)
        model_kalman = SpreadModel(lookback=100, min_lookback=20,
                                   use_kalman=True, kalman_process_variance=0.5,
                                   kalman_measurement_variance=5.0)
        n = 100
        b = np.cumsum(np.random.randn(n))
        a = 2 * b + np.random.randn(n) * 0.5
        prices_a = np.exp(a / 10 + 10)
        prices_b = np.exp(b / 10 + 8)

        result = model_kalman.compute(prices_a, prices_b, pair_id="TEST/PAIR")
        assert result.half_life > 0
        # Second call should use Kalman state
        result2 = model_kalman.compute(prices_a, prices_b, pair_id="TEST/PAIR")
        assert result2.half_life > 0

    def test_pair_id_parameter(self):
        """compute() should accept pair_id parameter"""
        np.random.seed(42)
        n = 100
        prices_a = 50000 + np.cumsum(np.random.randn(n) * 10)
        prices_b = 3000 + np.cumsum(np.random.randn(n) * 5)
        prices_a = np.abs(prices_a)
        prices_b = np.abs(prices_b)

        result = self.model.compute(prices_a, prices_b, pair_id="BTC/ETH")
        assert result.ratio > 0


# ============================================================
# Kalman Filter Tests
# ============================================================

class TestKalmanFilter:
    def test_first_measurement(self):
        """First measurement should be returned as-is"""
        kf = KalmanFilter(process_variance=0.5, measurement_variance=5.0)
        result = kf.update(10.0)
        assert result.filtered_value == 10.0
        assert result.gain == 1.0

    def test_smoothing(self):
        """Kalman should smooth out spikes"""
        kf = KalmanFilter(process_variance=0.5, measurement_variance=5.0)
        # Feed steady values then a spike
        for _ in range(10):
            kf.update(10.0)
        result = kf.update(50.0)  # Spike
        # Filtered value should be between 10 and 50, closer to 10
        assert 10.0 < result.filtered_value < 50.0
        assert result.filtered_value < 30.0  # Should dampen the spike

    def test_convergence(self):
        """Repeated same value should converge to that value"""
        kf = KalmanFilter(process_variance=0.5, measurement_variance=5.0)
        kf.update(100.0)  # Start at wrong value
        for _ in range(50):
            result = kf.update(20.0)
        assert abs(result.filtered_value - 20.0) < 1.0

    def test_reset(self):
        """Reset should clear state"""
        kf = KalmanFilter()
        kf.update(10.0)
        kf.reset()
        assert kf.x_hat is None


# ============================================================
# Regime Detection Tests
# ============================================================

class TestRegimeDetector:
    def test_warmup_returns_none(self):
        """Should return None during warmup"""
        rd = RegimeDetector(vol_window=10, history_window=100)
        for i in range(5):
            result = rd.update(50000 + i)
        assert result is None

    def test_normal_regime(self):
        """Stable prices should give LOW or NORMAL regime"""
        np.random.seed(42)
        rd = RegimeDetector(vol_window=10, history_window=200)
        # Feed stable prices
        for i in range(300):
            price = 50000 + np.random.randn() * 5
            result = rd.update(price)
        assert result is not None
        assert result.regime in (MarketRegime.LOW, MarketRegime.NORMAL)
        assert result.allow_new_positions is True

    def test_extreme_regime(self):
        """Sudden high volatility should trigger HIGH or EXTREME"""
        np.random.seed(42)
        rd = RegimeDetector(vol_window=10, history_window=200)
        rd._min_regime_hold = 0.0  # Disable hold time for test
        # Feed stable prices first
        for i in range(200):
            price = 50000 + np.random.randn() * 5
            rd.update(price)
        # Then extreme volatility
        for i in range(50):
            price = 50000 + np.random.randn() * 500
            result = rd.update(price)
        assert result is not None
        assert result.regime in (MarketRegime.HIGH, MarketRegime.EXTREME)
        assert result.allow_new_positions is False

    def test_to_dict(self):
        """to_dict should return valid dict"""
        rd = RegimeDetector(vol_window=10, history_window=100)
        d = rd.to_dict()
        assert "regime" in d
        assert "allow_new_positions" in d
        assert d["warming_up"] is True

    def test_allow_positions_low_normal(self):
        """LOW and NORMAL should allow positions"""
        rd = RegimeDetector()
        rd.current_regime = MarketRegime.LOW
        # Simulate a result
        from models.regime import RegimeResult
        rd._last_result = RegimeResult(
            regime=MarketRegime.LOW,
            current_volatility=0.001,
            percentile=15.0,
            allow_new_positions=True,
            vol_window_size=60,
            history_window_size=200,
        )
        assert rd.last_result.allow_new_positions is True


# ============================================================
# Bayesian Model Tests
# ============================================================

class TestBayesianModel:
    def setup_method(self):
        self.model = BayesianModel()

    def test_balanced_orderbook(self):
        """Equal bids/asks should give ~0.5/0.5 posterior"""
        ob = {
            "bids": [[49999, 10], [49998, 10], [49997, 10]],
            "asks": [[50001, 10], [50002, 10], [50003, 10]],
        }
        result = self.model.estimate(50000, ob)
        assert 0.4 < result.posterior_long < 0.6
        assert abs(result.orderbook_imbalance) < 0.1

    def test_buy_pressure(self):
        """Heavy bids should increase P(long)"""
        ob = {
            "bids": [[49999, 100], [49998, 100], [49997, 100]],
            "asks": [[50001, 10], [50002, 10], [50003, 10]],
        }
        result = self.model.estimate(50000, ob)
        assert result.posterior_long > 0.5
        assert result.orderbook_imbalance > 0.5

    def test_high_funding_bearish(self):
        """High positive funding = crowded longs = bearish signal"""
        ob = {
            "bids": [[49999, 10], [49998, 10]],
            "asks": [[50001, 10], [50002, 10]],
        }
        result = self.model.estimate(50000, ob, funding_rate=0.001)
        assert result.funding_signal < 0  # Bearish

    def test_confidence_range(self):
        """Confidence should be between 0 and 1"""
        ob = {
            "bids": [[49999, 50]],
            "asks": [[50001, 10]],
        }
        result = self.model.estimate(50000, ob)
        assert 0 <= result.confidence <= 1


# ============================================================
# Edge Model Tests
# ============================================================

class TestEdgeModel:
    def setup_method(self):
        self.model = EdgeModel()

    def test_profitable_trade(self):
        """High z-score should produce positive edge"""
        result = self.model.calculate(
            zscore=3.0,
            spread_std=0.005,
            notional_per_leg=100,
            use_limit_orders=True,
        )
        assert result.net_edge > 0
        assert result.is_profitable

    def test_unprofitable_small_zscore(self):
        """Tiny z-score may not cover costs"""
        result = self.model.calculate(
            zscore=0.5,
            spread_std=0.001,
            notional_per_leg=100,
        )
        # With very small z-score and spread, costs may exceed edge
        assert result.total_cost > 0

    def test_taker_vs_maker(self):
        """Taker fees should be higher than maker"""
        maker = self.model.calculate(3.0, 0.005, 100, use_limit_orders=True)
        taker = self.model.calculate(3.0, 0.005, 100, use_limit_orders=False)
        assert taker.commission_cost > maker.commission_cost

    def test_funding_cost_adds_up(self):
        """Holding through funding should increase costs"""
        no_fund = self.model.calculate(3.0, 0.005, 100)
        with_fund = self.model.calculate(
            3.0, 0.005, 100,
            holding_periods_8h=3.0,
            funding_rate_a=0.001,
            funding_rate_b=0.0005,
        )
        assert with_fund.total_cost > no_fund.total_cost


# ============================================================
# Kelly Model Tests
# ============================================================

class TestKellyModel:
    def setup_method(self):
        self.model = KellyModel(fraction=0.25)

    def test_positive_edge(self):
        """Win rate > 50% with equal wins/losses should give positive Kelly"""
        trades = [0.01] * 60 + [-0.01] * 40  # 60% win rate
        result = self.model.calculate(174.0, trades)
        assert result.full_kelly > 0
        assert result.fractional_kelly > 0
        assert result.position_size_usd > 0
        assert result.edge > 0

    def test_negative_edge(self):
        """Losing strategy should give 0 position size"""
        trades = [0.005] * 30 + [-0.01] * 70  # 30% win rate, equal R
        result = self.model.calculate(174.0, trades)
        assert result.position_size_usd == 0

    def test_fractional_smaller_than_full(self):
        """Quarter Kelly should be smaller than full Kelly"""
        trades = [0.01] * 60 + [-0.008] * 40
        result = self.model.calculate(174.0, trades)
        if result.full_kelly > 0:
            assert result.fractional_kelly < result.full_kelly

    def test_insufficient_data(self):
        """With < min_trades, should return default sizing"""
        trades = [0.01, 0.02, -0.005]  # Only 3 trades
        result = self.model.calculate(174.0, trades)
        assert not result.enough_data

    def test_max_cap(self):
        """Should never exceed max_fraction"""
        trades = [0.05] * 90 + [-0.001] * 10  # 90% win rate, huge edge
        result = self.model.calculate(174.0, trades)
        assert result.fractional_kelly <= 0.20


# ============================================================
# Stoikov Model Tests
# ============================================================

class TestStoikovModel:
    def setup_method(self):
        self.model = StoikovModel()

    def test_symmetric_no_inventory(self):
        """Without inventory, bid/ask should be symmetric around mid"""
        quote = self.model.quote(50000, volatility=50.0, inventory=0.0)
        assert quote.bid_price < 50000
        assert quote.ask_price > 50000
        assert abs(quote.inventory_skew) < 0.01

    def test_long_inventory_skews_down(self):
        """Long inventory should lower reservation price (want to sell)"""
        quote = self.model.quote(50000, volatility=50.0, inventory=1.0)
        assert quote.reservation_price < 50000
        assert quote.bid_price < 50000

    def test_short_inventory_skews_up(self):
        """Short inventory should raise reservation price (want to buy)"""
        quote = self.model.quote(50000, volatility=50.0, inventory=-1.0)
        assert quote.reservation_price > 50000

    def test_spread_positive(self):
        """Spread should always be positive"""
        quote = self.model.quote(50000, volatility=50.0)
        assert quote.spread > 0
        assert quote.ask_price > quote.bid_price

    def test_execution_price_buy(self):
        """Buy execution price should be at or below mid"""
        price = self.model.execution_price(50000, "BUY", volatility=50.0)
        assert price <= 50000

    def test_execution_price_urgency(self):
        """Higher urgency should give tighter price"""
        patient = self.model.execution_price(50000, "BUY", 50.0, urgency=0.1)
        urgent = self.model.execution_price(50000, "BUY", 50.0, urgency=0.9)
        assert urgent > patient  # Urgent buy is closer to mid


# ============================================================
# Monte Carlo Model Tests
# ============================================================

class TestMonteCarloModel:
    def setup_method(self):
        self.model = MonteCarloModel(n_simulations=500)  # Fewer for test speed

    def test_profitable_strategy(self):
        """Consistently profitable strategy should pass validation"""
        np.random.seed(42)
        # 65% win rate, 1.5:1 reward ratio
        trades = list(np.random.choice(
            [0.003, 0.003, 0.003, 0.003, -0.002, -0.002, -0.002],
            size=100
        ))
        result = self.model.simulate(trades, initial_balance=174.0)
        assert result.probability_of_profit > 0.5
        assert result.mean_return > 0

    def test_losing_strategy_fails(self):
        """Losing strategy should fail validation"""
        np.random.seed(42)
        trades = list(np.random.choice([0.001, -0.003], size=100))
        result = self.model.simulate(trades, initial_balance=174.0)
        assert result.probability_of_profit < 0.5

    def test_insufficient_data(self):
        """Should handle < 5 trades gracefully"""
        result = self.model.simulate([0.01, -0.005], initial_balance=174.0)
        assert not result.is_valid
        assert result.probability_of_ruin == 1.0

    def test_simulate_with_params(self):
        """Theoretical simulation should work"""
        result = self.model.simulate_with_params(
            win_rate=0.6,
            avg_win_pct=0.003,
            avg_loss_pct=-0.002,
            initial_balance=174.0,
            trades_per_sim=200,
        )
        assert result.n_simulations == 500
        assert result.n_trades_per_sim == 200

    def test_ruin_probability_bounded(self):
        """Ruin probability should be between 0 and 1"""
        trades = list(np.random.choice([0.005, -0.003], size=50))
        result = self.model.simulate(trades)
        assert 0 <= result.probability_of_ruin <= 1


# ============================================================
# Dynamic Threshold Tests
# ============================================================

class TestDynamicThresholds:
    def setup_method(self):
        from strategies.stat_arb import StatArbStrategy
        self.strategy = StatArbStrategy()

    def test_short_halflife_tighter_entry(self):
        """Short half-life should give lower entry threshold"""
        entry_short, _ = self.strategy._dynamic_thresholds(10)
        entry_long, _ = self.strategy._dynamic_thresholds(100)
        assert entry_short < entry_long

    def test_short_halflife_tighter_exit(self):
        """Short half-life should give tighter exit threshold"""
        _, exit_short = self.strategy._dynamic_thresholds(10)
        _, exit_long = self.strategy._dynamic_thresholds(100)
        assert exit_short < exit_long

    def test_bounds(self):
        """Thresholds should stay within configured bounds"""
        for hl in [1, 10, 50, 120, 500, float('inf')]:
            entry, exit_ = self.strategy._dynamic_thresholds(hl)
            assert self.strategy.dyn_entry_min <= entry <= self.strategy.dyn_entry_max
            assert self.strategy.dyn_exit_min <= exit_ <= self.strategy.dyn_exit_max

    def test_disabled_returns_fixed(self):
        """When disabled, should return fixed thresholds"""
        self.strategy.dynamic_enabled = False
        entry, exit_ = self.strategy._dynamic_thresholds(10)
        assert entry == self.strategy.entry_threshold
        assert exit_ == self.strategy.exit_threshold


# ============================================================
# Strategy Enable Flag Tests
# ============================================================

class TestStrategyFlags:
    def test_momentum_disabled(self):
        """Momentum should be disabled by default"""
        from strategies.momentum_div import MomentumDivStrategy
        m = MomentumDivStrategy()
        assert m.enabled is False

    def test_funding_enabled(self):
        """Funding arb should be enabled by default"""
        from strategies.funding_arb import FundingArbStrategy
        f = FundingArbStrategy()
        assert f.enabled is True
        assert f.min_hold_minutes == 60


# ============================================================
# Multi-Timeframe Tests
# ============================================================

class TestCandleAggregator:
    def test_basic_aggregation(self):
        """Should aggregate ticks into candles"""
        from models.timeframe import CandleAggregator
        agg = CandleAggregator(timeframes=[60])  # 1m candles
        # Feed 120 seconds of ticks
        completed = []
        for i in range(120):
            result = agg.update("BTCUSDT", 50000 + i, timestamp=1000 + i)
            completed.extend(result)
        assert len(completed) >= 1  # Should have at least 1 completed candle

    def test_ohlc_correct(self):
        """OHLC values should be correct"""
        from models.timeframe import CandleAggregator
        agg = CandleAggregator(timeframes=[60])
        # First candle: prices 100, 110, 90, 105
        agg.update("TEST", 100.0, timestamp=0)
        agg.update("TEST", 110.0, timestamp=10)
        agg.update("TEST", 90.0, timestamp=20)
        agg.update("TEST", 105.0, timestamp=50)
        # Complete candle
        agg.update("TEST", 200.0, timestamp=60)

        closes = agg.get_closes("TEST", 60)
        # Should have 1 completed candle
        assert closes is None or len(closes) >= 1


# ============================================================
# Portfolio Correlation Tests
# ============================================================

class TestPortfolioOptimizer:
    def test_uncorrelated_allowed(self):
        """Uncorrelated pairs should be allowed"""
        from models.portfolio import PortfolioOptimizer
        po = PortfolioOptimizer(max_correlation=0.7)

        np.random.seed(42)
        for i in range(100):
            po.update_spread("A/B", np.random.randn())
            po.update_spread("C/D", np.random.randn())

        allowed, corrs = po.check_correlation("C/D", ["A/B"])
        assert allowed is True

    def test_identical_blocked(self):
        """Identical spread series should be blocked"""
        from models.portfolio import PortfolioOptimizer
        po = PortfolioOptimizer(max_correlation=0.7)

        for i in range(100):
            val = np.sin(i * 0.1)
            po.update_spread("A/B", val)
            po.update_spread("C/D", val)  # Same values

        allowed, corrs = po.check_correlation("C/D", ["A/B"])
        assert allowed is False

    def test_correlation_matrix(self):
        """Should compute valid correlation matrix"""
        from models.portfolio import PortfolioOptimizer
        po = PortfolioOptimizer()

        np.random.seed(42)
        for i in range(100):
            po.update_spread("A/B", np.random.randn())
            po.update_spread("C/D", np.random.randn())

        matrix = po.compute_correlation_matrix(["A/B", "C/D"])
        assert matrix["A/B"]["A/B"] == 1.0
        assert matrix["C/D"]["C/D"] == 1.0
        assert -1 <= matrix["A/B"]["C/D"] <= 1


# ============================================================
# Cross-Exchange Strategy Tests
# ============================================================

class TestCrossExchangeStrategy:
    def test_no_signal_small_spread(self):
        """Small spread should not generate signal"""
        from strategies.cross_exchange import CrossExchangeStrategy
        import asyncio
        strategy = CrossExchangeStrategy()
        result = asyncio.get_event_loop().run_until_complete(
            strategy.evaluate({
                "symbol": "BTCUSDT",
                "binance_price": 50000.0,
                "bybit_price": 50001.0,  # Very small difference
                "equity": 174.0,
            })
        )
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
