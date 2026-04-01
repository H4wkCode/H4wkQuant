"""
H4wkQuant - Model Calibration
Calibrates Stoikov, Bayesian, and Spread parameters using live market data.
"""
import asyncio
import json
import numpy as np
from loguru import logger

from shared.clients.binance_rest import BinanceRestClient
from models.spread import SpreadModel
from models.stoikov import StoikovModel


async def calibrate_volatility(client: BinanceRestClient, symbol: str) -> float:
    """Estimate 1-minute realized volatility"""
    klines = await client.get_klines(symbol, "1m", limit=480)
    closes = [float(k[4]) for k in klines]
    returns = np.diff(np.log(closes))
    vol = np.std(returns) * np.mean(closes)
    logger.info(f"{symbol} 1m volatility: {vol:.4f} (${vol:.2f})")
    return vol


async def calibrate_spread(client: BinanceRestClient, sym_a: str, sym_b: str):
    """Calibrate spread parameters for a pair"""
    klines_a = await client.get_klines(sym_a, "1m", limit=480)
    klines_b = await client.get_klines(sym_b, "1m", limit=480)

    prices_a = np.array([float(k[4]) for k in klines_a])
    prices_b = np.array([float(k[4]) for k in klines_b])

    n = min(len(prices_a), len(prices_b))
    prices_a = prices_a[:n]
    prices_b = prices_b[:n]

    spread = SpreadModel(lookback=n, min_lookback=60)
    result = spread.compute(prices_a, prices_b)

    logger.info(f"\n{sym_a}/{sym_b} Spread Calibration:")
    logger.info(f"  Hedge ratio: {result.hedge_ratio:.6f}")
    logger.info(f"  Z-score: {result.zscore:.4f}")
    logger.info(f"  Mean: {result.mean:.6f}")
    logger.info(f"  Std: {result.std:.6f}")
    logger.info(f"  Half-life: {result.half_life:.1f} minutes")
    logger.info(f"  Cointegrated: {result.is_cointegrated} (p={result.coint_pvalue:.3f})")

    return result


async def calibrate_stoikov(vol: float):
    """Test Stoikov quotes with calibrated volatility"""
    stoikov = StoikovModel(gamma=0.1, k=1.5)
    mid = 50000.0

    quote = stoikov.quote(mid, vol)
    logger.info(f"\nStoikov Quotes (mid=${mid:.2f}, vol={vol:.2f}):")
    logger.info(f"  Bid: ${quote.bid_price:.2f}")
    logger.info(f"  Ask: ${quote.ask_price:.2f}")
    logger.info(f"  Spread: ${quote.spread:.2f} ({quote.spread/mid*100:.4f}%)")

    # With inventory
    for inv in [-1.0, 0.0, 1.0]:
        q = stoikov.quote(mid, vol, inventory=inv)
        logger.info(f"  Inventory={inv:+.0f}: bid=${q.bid_price:.2f} ask=${q.ask_price:.2f} skew=${q.inventory_skew:.2f}")


async def main():
    client = BinanceRestClient()

    try:
        # Volatility calibration
        vols = {}
        for symbol in ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]:
            vols[symbol] = await calibrate_volatility(client, symbol)

        # Spread calibration
        pairs = [
            ("BTCUSDT", "ETHUSDT"),
            ("SOLUSDT", "ETHUSDT"),
            ("BNBUSDT", "ETHUSDT"),
        ]
        for a, b in pairs:
            await calibrate_spread(client, a, b)

        # Stoikov calibration with BTC vol
        await calibrate_stoikov(vols.get("BTCUSDT", 50.0))

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
