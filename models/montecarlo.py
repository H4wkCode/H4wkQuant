"""
H4wkQuant - Monte Carlo Simulator
W(t+1) = W(t) * (1 + r(t))

Validates strategies with 1000+ simulations before going live.
Checks Sharpe ratio, max drawdown, and probability of ruin.
"""
import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class MonteCarloResult:
    # Core metrics (across all simulations)
    mean_return: float  # Average total return
    median_return: float
    std_return: float

    # Risk metrics
    sharpe_ratio: float
    mean_max_drawdown: float
    worst_max_drawdown: float
    probability_of_ruin: float  # P(balance < ruin_threshold)
    probability_of_profit: float  # P(final_balance > initial)

    # Percentiles
    pct_5: float  # 5th percentile return (worst 5%)
    pct_25: float
    pct_75: float
    pct_95: float  # 95th percentile return (best 5%)

    # Strategy validation
    passes_sharpe: bool
    passes_drawdown: bool
    passes_ruin: bool
    is_valid: bool  # All checks pass

    n_simulations: int
    n_trades_per_sim: int


class MonteCarloModel:
    """
    Monte Carlo strategy validator.

    Takes historical trade returns and simulates N random paths
    by resampling (bootstrapping) from actual trade results.

    Validation criteria:
        - Sharpe > 1.0 (risk-adjusted return)
        - Max DD < 5% (drawdown control)
        - P(ruin) < 1% (survival probability)

    This runs BEFORE going live to validate the strategy works
    across many possible orderings of trades.
    """

    def __init__(
        self,
        n_simulations: int = 1000,
        min_sharpe: float = 1.0,
        max_drawdown: float = 0.05,  # 5%
        max_ruin_prob: float = 0.01,  # 1%
        ruin_threshold: float = 0.5,  # 50% of initial balance = ruin
    ):
        self.n_simulations = n_simulations
        self.min_sharpe = min_sharpe
        self.max_drawdown = max_drawdown
        self.max_ruin_prob = max_ruin_prob
        self.ruin_threshold = ruin_threshold

    def simulate(
        self,
        trade_returns: List[float],
        initial_balance: float = 174.0,
        trades_per_sim: int = None,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation by bootstrapping from actual trade returns.

        Args:
            trade_returns: List of percentage returns per trade (e.g. [0.005, -0.002, ...])
            initial_balance: Starting capital
            trades_per_sim: Number of trades per simulation (default: len(trade_returns) * 2)
        """
        returns = np.array(trade_returns)
        n_trades = trades_per_sim or len(returns) * 2

        if len(returns) < 5:
            logger.warning("Insufficient trade history for Monte Carlo (<5 trades)")
            return self._insufficient_data_result(n_trades)

        # Run simulations
        final_balances = np.zeros(self.n_simulations)
        max_drawdowns = np.zeros(self.n_simulations)
        all_returns = np.zeros(self.n_simulations)

        for i in range(self.n_simulations):
            # Bootstrap: random sample with replacement
            sampled_returns = np.random.choice(returns, size=n_trades, replace=True)

            # Simulate equity curve: W(t+1) = W(t) * (1 + r(t))
            equity = initial_balance
            peak = equity
            max_dd = 0.0

            for r in sampled_returns:
                equity *= (1 + r)
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd

            final_balances[i] = equity
            max_drawdowns[i] = max_dd
            all_returns[i] = (equity - initial_balance) / initial_balance

        # Calculate metrics
        mean_return = float(np.mean(all_returns))
        median_return = float(np.median(all_returns))
        std_return = float(np.std(all_returns))

        # Sharpe ratio (annualized assuming ~250 trading days, ~10 trades/day)
        # Simple: mean / std (already total period, not per-trade)
        per_trade_mean = float(np.mean(returns))
        per_trade_std = float(np.std(returns))
        if per_trade_std > 1e-10:
            # Annualize: assume 2500 trades/year (10/day * 250 days)
            sharpe_ratio = (per_trade_mean / per_trade_std) * np.sqrt(2500)
        else:
            sharpe_ratio = 0.0

        mean_max_dd = float(np.mean(max_drawdowns))
        worst_max_dd = float(np.max(max_drawdowns))

        # Ruin probability
        ruin_balance = initial_balance * self.ruin_threshold
        n_ruin = np.sum(final_balances < ruin_balance)
        prob_ruin = float(n_ruin / self.n_simulations)

        # Profit probability
        prob_profit = float(np.sum(final_balances > initial_balance) / self.n_simulations)

        # Percentiles
        pct_5 = float(np.percentile(all_returns, 5))
        pct_25 = float(np.percentile(all_returns, 25))
        pct_75 = float(np.percentile(all_returns, 75))
        pct_95 = float(np.percentile(all_returns, 95))

        # Validation
        passes_sharpe = sharpe_ratio >= self.min_sharpe
        passes_dd = mean_max_dd <= self.max_drawdown
        passes_ruin = prob_ruin <= self.max_ruin_prob

        return MonteCarloResult(
            mean_return=mean_return,
            median_return=median_return,
            std_return=std_return,
            sharpe_ratio=round(sharpe_ratio, 2),
            mean_max_drawdown=round(mean_max_dd, 4),
            worst_max_drawdown=round(worst_max_dd, 4),
            probability_of_ruin=round(prob_ruin, 4),
            probability_of_profit=round(prob_profit, 4),
            pct_5=round(pct_5, 4),
            pct_25=round(pct_25, 4),
            pct_75=round(pct_75, 4),
            pct_95=round(pct_95, 4),
            passes_sharpe=passes_sharpe,
            passes_drawdown=passes_dd,
            passes_ruin=passes_ruin,
            is_valid=passes_sharpe and passes_dd and passes_ruin,
            n_simulations=self.n_simulations,
            n_trades_per_sim=n_trades,
        )

    def simulate_with_params(
        self,
        win_rate: float,
        avg_win_pct: float,
        avg_loss_pct: float,
        initial_balance: float = 174.0,
        trades_per_sim: int = 500,
    ) -> MonteCarloResult:
        """
        Simulate from theoretical parameters (before having real trades).

        Args:
            win_rate: Expected win probability (0-1)
            avg_win_pct: Average win return (e.g. 0.003 for 0.3%)
            avg_loss_pct: Average loss return (e.g. -0.002 for -0.2%)
        """
        # Generate synthetic trade returns
        n_synthetic = 100
        n_wins = int(n_synthetic * win_rate)
        n_losses = n_synthetic - n_wins

        wins = np.random.normal(avg_win_pct, avg_win_pct * 0.3, n_wins)
        losses = np.random.normal(avg_loss_pct, abs(avg_loss_pct) * 0.3, n_losses)

        # Ensure wins are positive, losses are negative
        wins = np.abs(wins)
        losses = -np.abs(losses)

        trade_returns = list(wins) + list(losses)
        np.random.shuffle(trade_returns)

        return self.simulate(trade_returns, initial_balance, trades_per_sim)

    def _insufficient_data_result(self, n_trades: int) -> MonteCarloResult:
        return MonteCarloResult(
            mean_return=0, median_return=0, std_return=0,
            sharpe_ratio=0, mean_max_drawdown=0, worst_max_drawdown=0,
            probability_of_ruin=1.0, probability_of_profit=0,
            pct_5=0, pct_25=0, pct_75=0, pct_95=0,
            passes_sharpe=False, passes_drawdown=False, passes_ruin=False,
            is_valid=False, n_simulations=0, n_trades_per_sim=n_trades,
        )
