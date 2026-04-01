"""
H4wkQuant - Cross-Exchange Arbitrage Strategy
Binance vs Bybit price difference arbitrage.
Buy on cheaper exchange, sell on more expensive exchange.
"""
import time
from typing import Optional, Dict
from loguru import logger

from strategies.base import BaseStrategy
from shared.schemas.models import ArbSignal, ArbSignalAction, StrategyType
from shared.config.settings import settings


class CrossExchangeStrategy(BaseStrategy):
    """
    Cross-exchange arbitrage between Binance and Bybit.

    Entry: price difference > threshold (after fees on both exchanges)
    Exit: price difference converges or timeout
    """

    def __init__(self):
        super().__init__("cross_exchange")
        self.threshold_pct = 0.15  # 0.15% price difference minimum
        self.min_edge_usd = 0.10  # Minimum absolute edge in USD
        self.symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"]
        self.pairs = [f"{s}:CE" for s in self.symbols]
        self.cooldown: Dict[str, float] = {}
        self.cooldown_seconds = 120

        # Fee estimates
        self.binance_taker_fee = settings.risk.taker_fee  # 0.04%
        self.bybit_taker_fee = 0.00055  # 0.055% taker

    def get_pairs(self):
        return self.pairs

    async def evaluate(self, market_data: Dict) -> Optional[ArbSignal]:
        """
        Check for cross-exchange arb opportunity.

        market_data expected:
            symbol: str
            binance_price: float
            bybit_price: float
            equity: float
        """
        symbol = market_data["symbol"]
        binance_price = market_data["binance_price"]
        bybit_price = market_data["bybit_price"]
        equity = market_data.get("equity", 174.0)

        if not binance_price or not bybit_price or binance_price <= 0 or bybit_price <= 0:
            return None

        # Cooldown check
        now = time.time()
        if now < self.cooldown.get(symbol, 0):
            return None

        # Calculate spread
        spread_pct = (binance_price - bybit_price) / bybit_price * 100

        # Total fees for round-trip
        total_fee_pct = (self.binance_taker_fee + self.bybit_taker_fee) * 2 * 100  # entry + exit

        # Net edge
        net_spread_pct = abs(spread_pct) - total_fee_pct

        if net_spread_pct < self.threshold_pct:
            return None

        # Determine direction
        notional = min(equity * settings.risk.default_leverage / 3, settings.risk.max_leg_size_usd)
        net_edge = notional * net_spread_pct / 100

        if net_edge < self.min_edge_usd:
            return None

        if spread_pct > 0:
            # Binance expensive, Bybit cheap: buy Bybit, sell Binance
            buy_exchange = "bybit"
            sell_exchange = "binance"
        else:
            # Bybit expensive, Binance cheap: buy Binance, sell Bybit
            buy_exchange = "binance"
            sell_exchange = "bybit"

        self.cooldown[symbol] = now + self.cooldown_seconds

        pair_id = f"{symbol}:CE"  # CE = cross-exchange
        logger.info(
            f"Cross-exchange signal: {symbol} spread={spread_pct:.4f}% "
            f"net={net_spread_pct:.4f}% edge=${net_edge:.4f}"
        )

        return ArbSignal(
            pair_id=pair_id,
            strategy=StrategyType.CROSS_EXCHANGE,
            action=ArbSignalAction.OPEN,
            zscore=spread_pct,  # Using zscore field for spread %
            edge_net=net_edge,
            confidence=min(net_spread_pct / 0.5, 1.0),
            leg_a={"symbol": symbol, "side": "LONG", "exchange": buy_exchange, "weight": 1.0},
            leg_b={"symbol": symbol, "side": "SHORT", "exchange": sell_exchange, "weight": 1.0},
            position_size_usd=round(notional, 2),
            metadata={
                "spread_pct": round(spread_pct, 4),
                "net_spread_pct": round(net_spread_pct, 4),
                "binance_price": binance_price,
                "bybit_price": bybit_price,
            },
        )

    async def should_close(self, position_data: Dict, market_data: Dict) -> Optional[ArbSignal]:
        """Check if cross-exchange position should close."""
        symbol = position_data.get("leg_a_symbol", "")
        pair_id = position_data.get("pair_id", "")

        binance_price = market_data.get("binance_price", 0)
        bybit_price = market_data.get("bybit_price", 0)

        if not binance_price or not bybit_price:
            return None

        spread_pct = abs(binance_price - bybit_price) / min(binance_price, bybit_price) * 100

        # Close if spread converged
        if spread_pct < 0.02:
            return ArbSignal(
                pair_id=pair_id,
                strategy=StrategyType.CROSS_EXCHANGE,
                action=ArbSignalAction.CLOSE,
                zscore=spread_pct,
                edge_net=0,
                confidence=1.0,
                leg_a={"symbol": symbol, "side": "CLOSE"},
                leg_b={"symbol": symbol, "side": "CLOSE"},
                metadata={"exit_reason": "spread_converged"},
            )

        # Close if held too long (30 min)
        hold_time = time.time() - position_data.get("entry_time", time.time())
        if hold_time > 1800:
            return ArbSignal(
                pair_id=pair_id,
                strategy=StrategyType.CROSS_EXCHANGE,
                action=ArbSignalAction.CLOSE,
                zscore=spread_pct,
                edge_net=0,
                confidence=1.0,
                leg_a={"symbol": symbol, "side": "CLOSE"},
                leg_b={"symbol": symbol, "side": "CLOSE"},
                metadata={"exit_reason": "timeout"},
            )

        return None
