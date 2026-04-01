"""
H4wkQuant - Pre-Flight Check
Safety checks before going live. Run this before switching TRADING_MODE=live.
"""
import asyncio
import json
import os
from pathlib import Path
from loguru import logger
import redis.asyncio as aioredis

from shared.clients.binance_rest import BinanceRestClient
from shared.config.settings import settings
from models.montecarlo import MonteCarloModel


class PreflightCheck:
    def __init__(self):
        self.client = BinanceRestClient()
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def _ok(self, msg: str):
        self.passed += 1
        logger.info(f"  [PASS] {msg}")

    def _fail(self, msg: str):
        self.failed += 1
        logger.error(f"  [FAIL] {msg}")

    def _warn(self, msg: str):
        self.warnings += 1
        logger.warning(f"  [WARN] {msg}")

    async def run(self):
        logger.info("=" * 60)
        logger.info("H4wkQuant PRE-FLIGHT CHECK")
        logger.info("=" * 60)

        await self._check_api_keys()
        await self._check_account_balance()
        await self._check_exchange_access()
        await self._check_backtest_results()
        await self._check_monte_carlo()
        self._check_risk_config()
        self._check_pairs_config()

        await self.client.close()

        logger.info(f"\n{'=' * 60}")
        logger.info(f"RESULTS: {self.passed} passed, {self.failed} failed, {self.warnings} warnings")
        logger.info(f"{'=' * 60}")

        if self.failed > 0:
            logger.error("PRE-FLIGHT FAILED - DO NOT GO LIVE")
            logger.error("Fix all failures before switching to TRADING_MODE=live")
        elif self.warnings > 0:
            logger.warning("PRE-FLIGHT PASSED WITH WARNINGS")
            logger.warning("Review warnings before going live")
        else:
            logger.info("PRE-FLIGHT PASSED - Ready for live trading")

        return self.failed == 0

    async def _check_api_keys(self):
        logger.info("\n1. API Keys")
        if settings.binance.api_key and len(settings.binance.api_key) > 10:
            self._ok(f"API key present ({settings.binance.api_key[:8]}...)")
        else:
            self._fail("BINANCE_API_KEY not set or too short")

        if settings.binance.secret_key and len(settings.binance.secret_key) > 10:
            self._ok(f"Secret key present ({settings.binance.secret_key[:4]}...)")
        else:
            self._fail("BINANCE_SECRET_KEY not set or too short")

    async def _check_account_balance(self):
        logger.info("\n2. Account Balance")
        try:
            balance = await self.client.get_balance()
            if not balance:
                self._fail("Could not fetch USDT balance")
                return

            total = float(balance.get("balance", 0))
            available = float(balance.get("availableBalance", 0))

            logger.info(f"     Total: ${total:.2f} | Available: ${available:.2f}")

            if total >= 100:
                self._ok(f"Balance ${total:.2f} >= $100 minimum")
            elif total >= 50:
                self._warn(f"Balance ${total:.2f} is low (recommended: $100+)")
            else:
                self._fail(f"Balance ${total:.2f} too low for safe trading")

            if available >= total * 0.5:
                self._ok(f"Available balance is {available/total*100:.0f}% of total")
            else:
                self._warn(f"Only {available/total*100:.0f}% available - check open positions")

        except Exception as e:
            self._fail(f"Account access error: {e}")

    async def _check_exchange_access(self):
        logger.info("\n3. Exchange Access")
        try:
            info = await self.client.get_exchange_info()
            symbols = [s["symbol"] for s in info.get("symbols", []) if s.get("status") == "TRADING"]
            self._ok(f"Exchange info: {len(symbols)} active symbols")

            # Check if our pairs exist
            for pair in settings.arbitrage.pairs:
                sym_a, sym_b = pair.split("/")
                if sym_a in symbols and sym_b in symbols:
                    self._ok(f"Pair {pair} - both symbols active")
                else:
                    self._fail(f"Pair {pair} - symbol not found or inactive")

        except Exception as e:
            self._fail(f"Exchange access error: {e}")

        # Test signed endpoint
        try:
            positions = await self.client.get_positions()
            self._ok(f"Signed API works - {len(positions)} open positions on exchange")
        except Exception as e:
            self._fail(f"Signed API error (check API permissions): {e}")

    async def _check_backtest_results(self):
        logger.info("\n4. Backtest Validation")
        bt_path = Path(__file__).parent.parent / "data" / "backtest_results.json"
        if not bt_path.exists():
            self._warn("No backtest results found - run 'make backtest-enhanced' first")
            return

        with open(bt_path) as f:
            data = json.load(f)

        summary = data.get("summary", {})
        total_pnl = summary.get("total_pnl", 0)
        mc_valid = summary.get("mc_valid_count", 0)
        pairs_tested = summary.get("pairs_tested", 0)

        if total_pnl > 0:
            self._ok(f"Backtest PnL positive: ${total_pnl:.4f}")
        else:
            self._fail(f"Backtest PnL negative: ${total_pnl:.4f}")

        if mc_valid > 0:
            self._ok(f"Monte Carlo validated: {mc_valid}/{pairs_tested} pairs")
        else:
            self._fail("No pairs passed Monte Carlo validation")

    async def _check_monte_carlo(self):
        logger.info("\n5. Monte Carlo Simulation")
        try:
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/1")
            r = aioredis.from_url(redis_url, decode_responses=True)
            await r.ping()

            # Get returns from trade history
            trades_raw = await r.lrange("q:trades:history", 0, -1)
            await r.aclose()

            if len(trades_raw) < 5:
                self._warn(f"Insufficient trades for Monte Carlo ({len(trades_raw)}/5 minimum)")
                return

            trades = [json.loads(t) for t in trades_raw]

            # Calculate trade returns
            returns = []
            for t in trades:
                pnl = t.get("combined_pnl", 0)
                size = t.get("leg_a_qty", 0) * t.get("leg_a_entry", 0)
                if size > 0:
                    returns.append(pnl / size)

            if len(returns) < 5:
                self._warn(f"Insufficient calculable returns ({len(returns)}/5)")
                return

            mc = MonteCarloModel(n_simulations=settings.arbitrage.mc_simulations)
            balance_data = await self._get_balance_from_redis()
            result = mc.simulate(returns, initial_balance=balance_data)

            logger.info(f"     {mc.n_simulations} simulations, {len(returns)} trades")
            logger.info(f"     Sharpe Ratio: {result.sharpe_ratio}")
            logger.info(f"     Profit Probability: {result.probability_of_profit * 100:.1f}%")
            logger.info(f"     Ruin Probability: {result.probability_of_ruin * 100:.2f}%")
            logger.info(f"     Avg Max Drawdown: {result.mean_max_drawdown * 100:.2f}%")
            logger.info(f"     Worst Drawdown: {result.worst_max_drawdown * 100:.2f}%")
            logger.info(f"     5th Pct: {result.pct_5 * 100:.2f}% | 95th Pct: {result.pct_95 * 100:.2f}%")

            if result.passes_sharpe:
                self._ok(f"Sharpe {result.sharpe_ratio} >= {mc.min_sharpe}")
            else:
                self._fail(f"Sharpe {result.sharpe_ratio} < {mc.min_sharpe}")

            if result.passes_drawdown:
                self._ok(f"Avg drawdown {result.mean_max_drawdown*100:.2f}% <= {mc.max_drawdown*100:.0f}%")
            else:
                self._fail(f"Avg drawdown {result.mean_max_drawdown*100:.2f}% > {mc.max_drawdown*100:.0f}%")

            if result.passes_ruin:
                self._ok(f"Ruin probability {result.probability_of_ruin*100:.2f}% <= {mc.max_ruin_prob*100:.0f}%")
            else:
                self._fail(f"Ruin probability {result.probability_of_ruin*100:.2f}% > {mc.max_ruin_prob*100:.0f}%")

            if result.is_valid:
                logger.info("     ✓ Monte Carlo PASSED - strategy validated")
            else:
                logger.error("     ✗ Monte Carlo FAILED - do not go live!")

        except Exception as e:
            self._warn(f"Monte Carlo error: {e}")

    async def _get_balance_from_redis(self) -> float:
        try:
            redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/1")
            r = aioredis.from_url(redis_url, decode_responses=True)
            data = await r.get("q:account:balance")
            await r.aclose()
            if data:
                return json.loads(data).get("total_balance", 174.0)
        except Exception:
            pass
        return 174.0

    def _check_risk_config(self):
        logger.info("\n6. Risk Configuration")
        r = settings.risk

        if r.max_drawdown_percent <= 10:
            self._ok(f"Max drawdown: {r.max_drawdown_percent}%")
        else:
            self._fail(f"Max drawdown too high: {r.max_drawdown_percent}%")

        if r.max_daily_loss_percent <= 3:
            self._ok(f"Daily loss limit: {r.max_daily_loss_percent}%")
        else:
            self._warn(f"Daily loss limit high: {r.max_daily_loss_percent}%")

        if r.kill_switch_enabled:
            self._ok("Kill switch enabled")
        else:
            self._fail("Kill switch DISABLED - must be enabled for live")

        if r.default_leverage <= 5:
            self._ok(f"Default leverage: {r.default_leverage}x")
        else:
            self._warn(f"Leverage {r.default_leverage}x is aggressive")

    def _check_pairs_config(self):
        logger.info("\n7. Trading Pairs")
        pairs = settings.arbitrage.pairs
        if len(pairs) >= 1:
            self._ok(f"{len(pairs)} pairs configured: {pairs}")
        else:
            self._fail("No trading pairs configured")

        if len(pairs) <= 6:
            self._ok(f"Pair count within limits ({len(pairs)} <= 6)")
        else:
            self._warn(f"Many pairs ({len(pairs)}) - may exceed position limits")


async def main():
    checker = PreflightCheck()
    success = await checker.run()
    if not success:
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())
