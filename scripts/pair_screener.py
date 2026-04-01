"""
H4wkQuant - Dynamic Pair Screener
Scans 500+ Binance Futures perpetual pairs for cointegration.
Outputs best pairs for stat arb trading.
"""
import asyncio
import json
import itertools
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
from loguru import logger

from shared.clients.binance_rest import BinanceRestClient
from models.spread import SpreadModel
from shared.config.settings import settings


class PairScreener:
    def __init__(self, top_n_volume: int = 50, min_klines: int = 480):
        self.client = BinanceRestClient()
        self.spread_model = SpreadModel(lookback=min_klines, min_lookback=60)
        self.top_n_volume = top_n_volume
        self.min_klines = min_klines
        self.results: List[Dict] = []

    async def run(self):
        logger.info("Starting pair screening...")

        # 1. Get all USDT perpetuals
        symbols = await self._get_usdt_perpetuals()
        logger.info(f"Found {len(symbols)} USDT perpetual symbols")

        # 2. Get top N by 24h volume
        top_symbols = await self._filter_by_volume(symbols)
        logger.info(f"Top {len(top_symbols)} by volume: {[s['symbol'] for s in top_symbols[:10]]}...")

        # 3. Fetch klines for all top symbols
        klines_cache = {}
        for i, sym_info in enumerate(top_symbols):
            symbol = sym_info["symbol"]
            try:
                klines = await self.client.get_klines(symbol, "1m", limit=500)
                if klines and len(klines) >= self.min_klines // 2:
                    prices = np.array([float(k[4]) for k in klines])
                    klines_cache[symbol] = prices
                    logger.debug(f"[{i+1}/{len(top_symbols)}] {symbol}: {len(prices)} candles")
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")

        logger.info(f"Fetched klines for {len(klines_cache)} symbols")

        # 4. Test all combinations
        symbols_list = list(klines_cache.keys())
        total_pairs = len(symbols_list) * (len(symbols_list) - 1) // 2
        logger.info(f"Testing {total_pairs} pair combinations...")

        results = []
        tested = 0
        for sym_a, sym_b in itertools.combinations(symbols_list, 2):
            tested += 1
            if tested % 100 == 0:
                logger.info(f"Progress: {tested}/{total_pairs} pairs tested, {len(results)} valid")

            prices_a = klines_cache[sym_a]
            prices_b = klines_cache[sym_b]

            n = min(len(prices_a), len(prices_b))
            if n < 60:
                continue

            try:
                spread_result = self.spread_model.compute(prices_a[-n:], prices_b[-n:])
            except Exception:
                continue

            if not spread_result.is_cointegrated:
                continue

            if spread_result.half_life > 120 or spread_result.half_life < 1:
                continue

            score = (
                (1.0 - spread_result.coint_pvalue) * 40
                + max(0, 120 - spread_result.half_life) / 120 * 30
                + min(abs(spread_result.zscore), 4.0) / 4.0 * 20
                + (1.0 if spread_result.std > 0.001 else 0.0) * 10
            )

            results.append({
                "pair_id": f"{sym_a}/{sym_b}",
                "symbol_a": sym_a,
                "symbol_b": sym_b,
                "score": round(score, 2),
                "zscore": round(spread_result.zscore, 4),
                "half_life": round(spread_result.half_life, 1),
                "coint_pvalue": round(spread_result.coint_pvalue, 4),
                "hedge_ratio": round(spread_result.hedge_ratio, 6),
                "spread_std": round(spread_result.std, 8),
                "is_cointegrated": True,
            })

        results.sort(key=lambda x: x["score"], reverse=True)

        # Bug #2 fix: Max 2 pairs per coin to prevent single-coin monopoly
        filtered = []
        coin_count = {}
        for r in results:
            sym_a = r["symbol_a"]
            sym_b = r["symbol_b"]
            count_a = coin_count.get(sym_a, 0)
            count_b = coin_count.get(sym_b, 0)
            if count_a >= 2 or count_b >= 2:
                continue
            filtered.append(r)
            coin_count[sym_a] = count_a + 1
            coin_count[sym_b] = count_b + 1
            if len(filtered) >= 20:
                break

        self.results = filtered

        logger.info(f"\n{'='*70}")
        logger.info(f"PAIR SCREENING RESULTS - {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
        logger.info(f"{'='*70}")
        logger.info(f"Scanned: {len(symbols)} symbols, {total_pairs} combinations")
        logger.info(f"Cointegrated pairs found: {len(results)}")
        logger.info(f"\nTop {len(self.results)} pairs:")
        logger.info(f"{'Rank':<5} {'Pair':<25} {'Score':<8} {'Z-Score':<10} {'Half-Life':<12} {'P-Value':<10}")
        logger.info("-" * 70)

        for i, r in enumerate(self.results, 1):
            logger.info(
                f"{i:<5} {r['pair_id']:<25} {r['score']:<8.1f} "
                f"{r['zscore']:<10.3f} {r['half_life']:<12.1f} {r['coint_pvalue']:<10.4f}"
            )

        await self._save_results(results, len(symbols), total_pairs)
        await self.client.close()

        return self.results

    async def _get_usdt_perpetuals(self) -> List[str]:
        info = await self.client.get_exchange_info()
        symbols = []
        for s in info.get("symbols", []):
            if (
                s.get("status") == "TRADING"
                and s.get("contractType") == "PERPETUAL"
                and s["symbol"].endswith("USDT")
            ):
                symbols.append(s["symbol"])
        return symbols

    async def _filter_by_volume(self, symbols: List[str]) -> List[Dict]:
        tickers = await self.client.get_ticker_24h()
        if not isinstance(tickers, list):
            tickers = [tickers]

        sym_set = set(symbols)
        volume_data = []
        for t in tickers:
            sym = t.get("symbol", "")
            if sym in sym_set:
                volume_data.append({
                    "symbol": sym,
                    "volume_usd": float(t.get("quoteVolume", 0)),
                })

        volume_data.sort(key=lambda x: x["volume_usd"], reverse=True)
        return volume_data[: self.top_n_volume]

    async def _save_results(self, all_results: List[Dict], total_symbols: int, total_pairs: int):
        output_dir = Path(__file__).parent.parent / "data"
        output_dir.mkdir(parents=True, exist_ok=True)

        output = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_symbols_scanned": total_symbols,
            "total_pairs_tested": total_pairs,
            "cointegrated_found": len(all_results),
            "top_pairs": self.results,
            "all_cointegrated": all_results[:50],
        }

        output_path = output_dir / "screened_pairs.json"
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"\nResults saved to {output_path}")

        # Write to Redis for live system
        try:
            import redis.asyncio as aioredis
            import os
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/1")
            r = aioredis.from_url(redis_url, decode_responses=True)
            await r.ping()

            # Save screened pairs to Redis (top 10)
            screened = [p["pair_id"] for p in self.results[:10]]
            await r.set("q:config:screened_pairs", json.dumps(screened))

            # Save detailed info
            await r.set("q:config:screened_details", json.dumps(self.results[:10]))

            await r.close()
            logger.info(f"Wrote {len(screened)} pairs to Redis: {screened}")
        except Exception as e:
            logger.warning(f"Redis write failed (offline mode): {e}")


async def main():
    screener = PairScreener(top_n_volume=50, min_klines=480)
    results = await screener.run()

    if results:
        pairs = [r["pair_id"] for r in results[:6]]
        print(f"\nRecommended pairs for config:")
        print(f"  pairs: {json.dumps(pairs)}")


if __name__ == "__main__":
    asyncio.run(main())
