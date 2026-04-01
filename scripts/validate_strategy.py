"""
H4wkQuant - Strategy Validator
Monte Carlo validation with theoretical parameters.
Run before going live to verify strategy viability.
"""
import numpy as np
from loguru import logger

from models.montecarlo import MonteCarloModel
from models.kelly import KellyModel
from models.edge import EdgeModel


def validate_stat_arb():
    """Validate statistical arbitrage strategy"""
    logger.info("="*60)
    logger.info("STATISTICAL ARBITRAGE VALIDATION")
    logger.info("="*60)

    mc = MonteCarloModel(n_simulations=2000)

    # Conservative scenario
    logger.info("\n--- Conservative (win=55%, R=1.2:1) ---")
    result = mc.simulate_with_params(
        win_rate=0.55,
        avg_win_pct=0.003,   # 0.3% per trade
        avg_loss_pct=-0.0025,  # 0.25% per trade
        initial_balance=174.0,
        trades_per_sim=500,
    )
    _print_result(result)

    # Moderate scenario
    logger.info("\n--- Moderate (win=60%, R=1.5:1) ---")
    result = mc.simulate_with_params(
        win_rate=0.60,
        avg_win_pct=0.004,
        avg_loss_pct=-0.0027,
        initial_balance=174.0,
        trades_per_sim=500,
    )
    _print_result(result)

    # Optimistic scenario
    logger.info("\n--- Optimistic (win=65%, R=2:1) ---")
    result = mc.simulate_with_params(
        win_rate=0.65,
        avg_win_pct=0.005,
        avg_loss_pct=-0.0025,
        initial_balance=174.0,
        trades_per_sim=500,
    )
    _print_result(result)


def validate_edge_requirements():
    """Calculate minimum edge requirements"""
    logger.info("\n" + "="*60)
    logger.info("EDGE REQUIREMENTS")
    logger.info("="*60)

    edge = EdgeModel()

    for zscore in [1.5, 2.0, 2.5, 3.0]:
        for spread_std in [0.002, 0.005, 0.01]:
            result = edge.calculate(
                zscore=zscore,
                spread_std=spread_std,
                notional_per_leg=80,  # $80/leg with $174 * 3x / 3 pairs
                use_limit_orders=True,
            )
            status = "OK" if result.is_profitable else "NO"
            logger.info(
                f"  z={zscore:.1f} std={spread_std:.3f} -> "
                f"gross=${result.gross_edge:.4f} cost=${result.total_cost:.4f} "
                f"net=${result.net_edge:.4f} [{status}]"
            )


def validate_kelly_sizing():
    """Demonstrate Kelly sizing for different scenarios"""
    logger.info("\n" + "="*60)
    logger.info("KELLY SIZING ($174 equity)")
    logger.info("="*60)

    kelly = KellyModel(fraction=0.25)

    scenarios = [
        ("Conservative", 0.55, 0.003, 0.0025),
        ("Moderate", 0.60, 0.004, 0.0027),
        ("Aggressive", 0.70, 0.005, 0.002),
    ]

    for name, win_rate, avg_win, avg_loss in scenarios:
        result = kelly.calculate(
            equity=174.0,
            win_prob=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
        )
        logger.info(
            f"  {name}: full_kelly={result.full_kelly:.4f} "
            f"frac_kelly={result.fractional_kelly:.4f} "
            f"size=${result.position_size_usd:.2f} "
            f"edge={result.edge:.4f}"
        )


def _print_result(result):
    logger.info(f"  Sharpe: {result.sharpe_ratio} {'PASS' if result.passes_sharpe else 'FAIL'}")
    logger.info(f"  Mean DD: {result.mean_max_drawdown*100:.2f}% {'PASS' if result.passes_drawdown else 'FAIL'}")
    logger.info(f"  P(ruin): {result.probability_of_ruin*100:.2f}% {'PASS' if result.passes_ruin else 'FAIL'}")
    logger.info(f"  P(profit): {result.probability_of_profit*100:.1f}%")
    logger.info(f"  5th-95th pctile: {result.pct_5*100:.1f}% to {result.pct_95*100:.1f}%")
    logger.info(f"  STRATEGY VALID: {'YES' if result.is_valid else 'NO'}")


if __name__ == "__main__":
    validate_stat_arb()
    validate_edge_requirements()
    validate_kelly_sizing()
