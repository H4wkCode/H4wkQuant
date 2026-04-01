"""
H4wkQuant - Backtest Pairs
Tests statistical arbitrage on historical data.
Validates z-score signals, edge calculation, and Kelly sizing.
"""
import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from loguru import logger

from shared.clients.binance_rest import BinanceRestClient
from models.spread import SpreadModel
from models.edge import EdgeModel
from models.kelly import KellyModel
from models.montecarlo import MonteCarloModel


async def backtest_pair(symbol_a: str, symbol_b: str, days: int = 14):
    """Backtest stat arb on a pair using historical klines"""
    client = BinanceRestClient()

    logger.info(f"Fetching {days}d of 1m klines for {symbol_a} and {symbol_b}...")

    # Fetch historical 1m klines (max 1500 per request)
    klines_a = await client.get_klines(symbol_a, "1m", limit=500)
    klines_b = await client.get_klines(symbol_b, "1m", limit=500)

    await client.close()

    if not klines_a or not klines_b:
        logger.error("Failed to fetch klines")
        return

    # Extract close prices
    prices_a = np.array([float(k[4]) for k in klines_a])
    prices_b = np.array([float(k[4]) for k in klines_b])

    # Align lengths
    n = min(len(prices_a), len(prices_b))
    prices_a = prices_a[:n]
    prices_b = prices_b[:n]

    logger.info(f"Data: {n} candles")

    # Run spread model
    spread_model = SpreadModel(lookback=240, min_lookback=60)
    edge_model = EdgeModel()
    kelly_model = KellyModel(fraction=0.25)

    # Simulate trading
    lookback = 240
    entry_z = 2.0
    exit_z = 0.5
    notional = 80.0  # $80 per leg

    trades = []
    in_position = False
    entry_zscore = 0
    entry_idx = 0

    for i in range(lookback, n):
        window_a = prices_a[i - lookback:i + 1]
        window_b = prices_b[i - lookback:i + 1]

        result = spread_model.compute(window_a, window_b)

        if not in_position:
            # Entry check
            if abs(result.zscore) >= entry_z and result.is_cointegrated:
                edge = edge_model.calculate(result.zscore, result.std, notional)
                if edge.is_profitable:
                    in_position = True
                    entry_zscore = result.zscore
                    entry_idx = i
        else:
            # Exit check
            if abs(result.zscore) < exit_z or abs(result.zscore) > 4.0:
                # Calculate PnL (approximate)
                z_change = abs(entry_zscore) - abs(result.zscore)
                pnl_pct = z_change * result.std
                pnl = pnl_pct * notional * 2  # 2 legs

                # Subtract fees
                commission = 4 * 0.0002 * notional
                net_pnl = pnl - commission

                trades.append(net_pnl)
                in_position = False

                duration = i - entry_idx
                logger.info(
                    f"Trade: entry_z={entry_zscore:.2f} exit_z={result.zscore:.2f} "
                    f"PnL=${net_pnl:.4f} duration={duration}m"
                )

    if not trades:
        logger.warning("No trades generated!")
        return

    # Results
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    total_pnl = sum(trades)

    logger.info(f"\n{'='*50}")
    logger.info(f"BACKTEST RESULTS: {symbol_a}/{symbol_b}")
    logger.info(f"{'='*50}")
    logger.info(f"Total trades: {len(trades)}")
    logger.info(f"Wins: {len(wins)} | Losses: {len(losses)}")
    logger.info(f"Win rate: {len(wins)/len(trades)*100:.1f}%")
    logger.info(f"Total PnL: ${total_pnl:.4f}")
    logger.info(f"Avg PnL: ${total_pnl/len(trades):.4f}")
    logger.info(f"Best: ${max(trades):.4f} | Worst: ${min(trades):.4f}")

    # Kelly
    kelly = kelly_model.calculate(174.0, [t / notional for t in trades])
    logger.info(f"Kelly fraction: {kelly.fractional_kelly:.4f}")
    logger.info(f"Kelly position size: ${kelly.position_size_usd:.2f}")

    # Monte Carlo validation
    mc = MonteCarloModel(n_simulations=1000)
    mc_result = mc.simulate([t / notional for t in trades], initial_balance=174.0)
    logger.info(f"\nMonte Carlo ({mc_result.n_simulations} sims):")
    logger.info(f"  Sharpe: {mc_result.sharpe_ratio}")
    logger.info(f"  Mean DD: {mc_result.mean_max_drawdown*100:.2f}%")
    logger.info(f"  P(ruin): {mc_result.probability_of_ruin*100:.2f}%")
    logger.info(f"  P(profit): {mc_result.probability_of_profit*100:.1f}%")
    logger.info(f"  VALID: {'YES' if mc_result.is_valid else 'NO'}")


async def main():
    pairs = [
        ("BTCUSDT", "ETHUSDT"),
        ("SOLUSDT", "ETHUSDT"),
        ("BNBUSDT", "ETHUSDT"),
    ]

    for a, b in pairs:
        await backtest_pair(a, b)
        print()


if __name__ == "__main__":
    asyncio.run(main())
