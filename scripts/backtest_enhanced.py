"""
H4wkQuant - Enhanced Backtest
Walk-forward backtest with extended data and equity curve tracking.
Supports dynamic pairs from pair screener.
"""
import asyncio
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from loguru import logger

from shared.clients.binance_rest import BinanceRestClient
from models.spread import SpreadModel
from models.edge import EdgeModel
from models.kelly import KellyModel
from models.montecarlo import MonteCarloModel
from shared.config.settings import settings


class EnhancedBacktest:
    def __init__(
        self,
        pairs: List[Tuple[str, str]] = None,
        days: int = 7,
        initial_equity: float = 174.0,
        train_pct: float = 0.7,
    ):
        self.client = BinanceRestClient()
        self.spread_model = SpreadModel(lookback=480, min_lookback=60)
        self.edge_model = EdgeModel(
            maker_fee=settings.risk.maker_fee,
            taker_fee=settings.risk.taker_fee,
        )
        self.kelly_model = KellyModel(fraction=settings.arbitrage.kelly_fraction)
        self.mc_model = MonteCarloModel(n_simulations=1000)

        self.pairs = pairs
        self.days = days
        self.initial_equity = initial_equity
        self.train_pct = train_pct

        self.entry_z = settings.arbitrage.zscore_entry_threshold
        self.exit_z = settings.arbitrage.zscore_exit_threshold
        self.max_half_life = settings.arbitrage.half_life_max
        self.max_leg_size = settings.risk.max_leg_size_usd
        self.leverage = settings.risk.default_leverage

    async def run(self):
        if not self.pairs:
            self.pairs = self._load_pairs()

        logger.info(f"Enhanced Backtest: {len(self.pairs)} pairs, {self.days} days")

        all_results = []
        for sym_a, sym_b in self.pairs:
            result = await self._backtest_pair(sym_a, sym_b)
            if result:
                all_results.append(result)

        await self.client.close()

        if not all_results:
            logger.warning("No backtest results generated!")
            return

        self._print_summary(all_results)
        self._save_results(all_results)

        return all_results

    def _load_pairs(self) -> List[Tuple[str, str]]:
        screened_path = Path(__file__).parent.parent / "data" / "screened_pairs.json"
        if screened_path.exists():
            with open(screened_path) as f:
                data = json.load(f)
            pairs = []
            for p in data.get("top_pairs", [])[:6]:
                pairs.append((p["symbol_a"], p["symbol_b"]))
            if pairs:
                logger.info(f"Loaded {len(pairs)} pairs from screener results")
                return pairs

        logger.info("Using default pairs")
        return [
            ("BTCUSDT", "ETHUSDT"),
            ("SOLUSDT", "ETHUSDT"),
            ("BNBUSDT", "ETHUSDT"),
        ]

    async def _fetch_klines_extended(self, symbol: str, days: int) -> np.ndarray:
        all_klines = []
        end_time = int(time.time() * 1000)
        start_time = end_time - (days * 24 * 60 * 60 * 1000)
        current_start = start_time

        while current_start < end_time:
            try:
                session = await self.client._get_session()
                url = f"{self.client.base_url}/fapi/v1/klines"
                params = {
                    "symbol": symbol,
                    "interval": "1m",
                    "startTime": current_start,
                    "limit": 1500,
                }
                async with session.get(url, params=params) as resp:
                    klines = await resp.json()

                if not klines or not isinstance(klines, list):
                    break

                all_klines.extend(klines)
                current_start = int(klines[-1][0]) + 60000

                if len(klines) < 1500:
                    break

                await asyncio.sleep(0.2)
            except Exception as e:
                logger.error(f"Kline fetch error for {symbol}: {e}")
                break

        if not all_klines:
            return np.array([])

        prices = np.array([float(k[4]) for k in all_klines])
        logger.info(f"  {symbol}: {len(prices)} candles ({len(prices)/1440:.1f} days)")
        return prices

    async def _backtest_pair(self, sym_a: str, sym_b: str) -> Optional[Dict]:
        pair_id = f"{sym_a}/{sym_b}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Backtesting: {pair_id}")

        prices_a = await self._fetch_klines_extended(sym_a, self.days)
        prices_b = await self._fetch_klines_extended(sym_b, self.days)

        if len(prices_a) == 0 or len(prices_b) == 0:
            logger.error(f"No data for {pair_id}")
            return None

        n = min(len(prices_a), len(prices_b))
        prices_a = prices_a[:n]
        prices_b = prices_b[:n]

        if n < 500:
            logger.warning(f"Not enough data: {n} candles")
            return None

        train_n = int(n * self.train_pct)
        test_start = train_n

        logger.info(f"  Train: {train_n} candles | Test: {n - train_n} candles")

        train_result = self.spread_model.compute(prices_a[:train_n], prices_b[:train_n])
        if not train_result.is_cointegrated:
            logger.warning(f"  Not cointegrated in training period")
            return None

        logger.info(
            f"  Training: z={train_result.zscore:.2f} HL={train_result.half_life:.0f}m "
            f"coint_p={train_result.coint_pvalue:.4f}"
        )

        lookback = 480
        trades = []
        equity = self.initial_equity
        equity_curve = [equity]
        in_position = False
        entry_zscore = 0
        entry_idx = 0
        entry_notional = 0

        for i in range(max(test_start, lookback), n):
            window_a = prices_a[i - lookback : i + 1]
            window_b = prices_b[i - lookback : i + 1]

            try:
                result = self.spread_model.compute(window_a, window_b)
            except Exception:
                continue

            if not in_position:
                if (
                    abs(result.zscore) >= self.entry_z
                    and result.is_cointegrated
                    and result.half_life <= self.max_half_life
                ):
                    notional = min(equity * self.leverage / 3, self.max_leg_size)
                    edge = self.edge_model.calculate(result.zscore, result.std, notional)
                    if edge.is_profitable and notional >= 5.0:
                        in_position = True
                        entry_zscore = result.zscore
                        entry_idx = i
                        entry_notional = notional
            else:
                should_exit = False
                exit_reason = ""

                if abs(result.zscore) < self.exit_z:
                    should_exit = True
                    exit_reason = "zscore_revert"
                elif abs(result.zscore) > 4.0:
                    should_exit = True
                    exit_reason = "zscore_blowup"
                elif np.sign(result.zscore) != np.sign(entry_zscore):
                    should_exit = True
                    exit_reason = "zscore_cross"
                elif not result.is_cointegrated:
                    should_exit = True
                    exit_reason = "coint_break"

                if should_exit:
                    z_change = abs(entry_zscore) - abs(result.zscore)
                    pnl_pct = z_change * result.std
                    gross_pnl = pnl_pct * entry_notional * 2

                    commission = 4 * settings.risk.maker_fee * entry_notional
                    net_pnl = gross_pnl - commission

                    equity += net_pnl
                    duration = i - entry_idx

                    trades.append({
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "entry_zscore": round(entry_zscore, 4),
                        "exit_zscore": round(result.zscore, 4),
                        "gross_pnl": round(gross_pnl, 6),
                        "commission": round(commission, 6),
                        "net_pnl": round(net_pnl, 6),
                        "duration_min": duration,
                        "exit_reason": exit_reason,
                        "notional": round(entry_notional, 2),
                        "equity_after": round(equity, 4),
                    })

                    in_position = False

            equity_curve.append(equity)

        if not trades:
            logger.warning(f"  No trades generated for {pair_id}")
            return None

        returns = [t["net_pnl"] / t["notional"] for t in trades]
        wins = [t for t in trades if t["net_pnl"] > 0]
        losses = [t for t in trades if t["net_pnl"] <= 0]
        total_pnl = sum(t["net_pnl"] for t in trades)

        equity_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (peak - equity_arr) / peak
        max_dd = float(np.max(drawdown))

        avg_holding = np.mean([t["duration_min"] for t in trades])
        win_rate = len(wins) / len(trades) * 100

        gross_wins = sum(t["net_pnl"] for t in wins) if wins else 0
        gross_losses = abs(sum(t["net_pnl"] for t in losses)) if losses else 1
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

        if len(returns) > 1 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252 * 24 * 60 / max(avg_holding, 1))
        else:
            sharpe = 0.0

        mc_result = self.mc_model.simulate(returns, initial_balance=self.initial_equity)

        logger.info(f"\n  RESULTS: {pair_id}")
        logger.info(f"  {'─'*40}")
        logger.info(f"  Trades: {len(trades)} (W:{len(wins)} L:{len(losses)})")
        logger.info(f"  Win Rate: {win_rate:.1f}%")
        logger.info(f"  Total PnL: ${total_pnl:.4f}")
        logger.info(f"  Profit Factor: {profit_factor:.2f}")
        logger.info(f"  Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"  Max Drawdown: {max_dd*100:.2f}%")
        logger.info(f"  Avg Holding: {avg_holding:.0f} min")
        logger.info(f"  Final Equity: ${equity:.2f}")
        logger.info(f"  MC Valid: {'YES' if mc_result.is_valid else 'NO'}")
        logger.info(f"  MC P(ruin): {mc_result.probability_of_ruin*100:.2f}%")

        return {
            "pair_id": pair_id,
            "symbol_a": sym_a,
            "symbol_b": sym_b,
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 1),
            "total_pnl": round(total_pnl, 4),
            "profit_factor": round(profit_factor, 2),
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "avg_holding_min": round(avg_holding, 0),
            "final_equity": round(equity, 2),
            "mc_valid": mc_result.is_valid,
            "mc_sharpe": mc_result.sharpe_ratio,
            "mc_prob_ruin": round(mc_result.probability_of_ruin, 4),
            "mc_prob_profit": round(mc_result.probability_of_profit, 4),
            "trades": trades,
            "equity_curve_sample": equity_curve[:: max(1, len(equity_curve) // 200)],
            "train_candles": train_n,
            "test_candles": n - train_n,
        }

    def _print_summary(self, results: List[Dict]):
        logger.info(f"\n{'='*70}")
        logger.info(f"ENHANCED BACKTEST SUMMARY")
        logger.info(f"{'='*70}")

        total_trades = sum(r["total_trades"] for r in results)
        total_pnl = sum(r["total_pnl"] for r in results)
        valid_count = sum(1 for r in results if r["mc_valid"])

        logger.info(f"Pairs tested: {len(results)}")
        logger.info(f"Total trades: {total_trades}")
        logger.info(f"Total PnL: ${total_pnl:.4f}")
        logger.info(f"MC Valid: {valid_count}/{len(results)}")

        logger.info(
            f"\n{'Pair':<25} {'Trades':<8} {'WR%':<8} {'PnL$':<12} "
            f"{'Sharpe':<8} {'MaxDD%':<8} {'MC':<5}"
        )
        logger.info("-" * 70)
        for r in sorted(results, key=lambda x: x["total_pnl"], reverse=True):
            mc = "OK" if r["mc_valid"] else "FAIL"
            logger.info(
                f"{r['pair_id']:<25} {r['total_trades']:<8} {r['win_rate']:<8.1f} "
                f"${r['total_pnl']:<11.4f} {r['sharpe_ratio']:<8.2f} "
                f"{r['max_drawdown_pct']:<8.2f} {mc:<5}"
            )

    def _save_results(self, results: List[Dict]):
        output_dir = Path(__file__).parent.parent / "data"
        output_dir.mkdir(parents=True, exist_ok=True)

        output = {
            "timestamp": datetime.utcnow().isoformat(),
            "config": {
                "days": self.days,
                "initial_equity": self.initial_equity,
                "train_pct": self.train_pct,
                "entry_z": self.entry_z,
                "exit_z": self.exit_z,
                "leverage": self.leverage,
            },
            "summary": {
                "pairs_tested": len(results),
                "total_trades": sum(r["total_trades"] for r in results),
                "total_pnl": round(sum(r["total_pnl"] for r in results), 4),
                "mc_valid_count": sum(1 for r in results if r["mc_valid"]),
            },
            "pairs": results,
        }

        path = output_dir / "backtest_results.json"
        with open(path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"\nDetailed results saved to {path}")


async def main():
    bt = EnhancedBacktest(days=7, initial_equity=174.0)
    await bt.run()


if __name__ == "__main__":
    asyncio.run(main())
