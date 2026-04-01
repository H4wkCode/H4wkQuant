"""
H4wkQuant - Database Repository
Repository pattern for arbitrage trade operations
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from sqlalchemy import select, func, desc, and_
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from shared.database.models import ArbTrade, SpreadHistory, AccountSnapshot
from shared.database.connection import get_session_factory


class QuantRepository:

    def __init__(self):
        self._factory = None

    def _get_factory(self):
        if self._factory is None:
            self._factory = get_session_factory()
        return self._factory

    # =========================================================================
    # Arb Trades
    # =========================================================================

    async def save_arb_entry(self, **kwargs) -> int:
        async with self._get_factory()() as session:
            trade = ArbTrade(**kwargs, entry_time=datetime.utcnow())
            session.add(trade)
            await session.commit()
            await session.refresh(trade)
            return trade.id

    async def update_arb_exit(
        self, trade_id: int, leg_a_exit: float, leg_b_exit: float,
        combined_pnl: float, total_commission: float, exit_zscore: float,
        exit_reason: str = "zscore_revert",
    ):
        async with self._get_factory()() as session:
            stmt = select(ArbTrade).where(ArbTrade.id == trade_id)
            result = await session.execute(stmt)
            trade = result.scalar_one_or_none()
            if trade:
                trade.leg_a_exit_price = leg_a_exit
                trade.leg_b_exit_price = leg_b_exit
                trade.combined_pnl = combined_pnl
                trade.total_commission = total_commission
                trade.exit_zscore = exit_zscore
                trade.exit_reason = exit_reason
                trade.exit_time = datetime.utcnow()
                await session.commit()

    async def get_recent_trades(self, limit: int = 50, strategy: str = None) -> List[Dict]:
        async with self._get_factory()() as session:
            stmt = select(ArbTrade).order_by(desc(ArbTrade.entry_time)).limit(limit)
            if strategy:
                stmt = stmt.where(ArbTrade.strategy == strategy)
            result = await session.execute(stmt)
            trades = result.scalars().all()
            return [
                {
                    "id": t.id, "pair_id": t.pair_id, "strategy": t.strategy,
                    "leg_a": f"{t.leg_a_side} {t.leg_a_symbol}",
                    "leg_b": f"{t.leg_b_side} {t.leg_b_symbol}",
                    "combined_pnl": t.combined_pnl,
                    "entry_zscore": t.entry_zscore, "exit_zscore": t.exit_zscore,
                    "exit_reason": t.exit_reason,
                    "entry_time": t.entry_time.isoformat() if t.entry_time else None,
                    "exit_time": t.exit_time.isoformat() if t.exit_time else None,
                }
                for t in trades
            ]

    async def get_trade_stats(self, days: int = 30, strategy: str = None) -> Dict:
        async with self._get_factory()() as session:
            since = datetime.utcnow() - timedelta(days=days)
            conditions = [ArbTrade.entry_time >= since]
            if strategy:
                conditions.append(ArbTrade.strategy == strategy)

            stmt = select(ArbTrade).where(and_(*conditions))
            result = await session.execute(stmt)
            trades = result.scalars().all()

            if not trades:
                return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0, "total_pnl": 0}

            closed = [t for t in trades if t.combined_pnl is not None]
            wins = [t for t in closed if t.combined_pnl > 0]
            losses = [t for t in closed if t.combined_pnl <= 0]
            total_pnl = sum(t.combined_pnl for t in closed)

            return {
                "total": len(trades),
                "closed": len(closed),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0,
                "total_pnl": round(total_pnl, 4),
                "avg_pnl": round(total_pnl / len(closed), 4) if closed else 0,
                "best_trade": round(max((t.combined_pnl for t in closed), default=0), 4),
                "worst_trade": round(min((t.combined_pnl for t in closed), default=0), 4),
                "total_commission": round(sum(t.total_commission or 0 for t in closed), 4),
            }

    # =========================================================================
    # Spread History
    # =========================================================================

    async def save_spread(self, pair_id: str, ratio: float, spread: float,
                          zscore: float, mean: float, std: float, half_life: float = 0):
        async with self._get_factory()() as session:
            record = SpreadHistory(
                time=datetime.utcnow(), pair_id=pair_id,
                ratio=ratio, spread=spread, zscore=zscore,
                mean=mean, std=std, half_life=half_life,
            )
            session.add(record)
            await session.commit()

    async def get_spread_history(self, pair_id: str, hours: int = 8) -> List[Dict]:
        async with self._get_factory()() as session:
            since = datetime.utcnow() - timedelta(hours=hours)
            stmt = (
                select(SpreadHistory)
                .where(SpreadHistory.pair_id == pair_id, SpreadHistory.time >= since)
                .order_by(SpreadHistory.time)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            return [
                {"time": r.time.isoformat(), "ratio": r.ratio, "spread": r.spread,
                 "zscore": r.zscore, "half_life": r.half_life}
                for r in rows
            ]

    # =========================================================================
    # Account Snapshots
    # =========================================================================

    async def save_account_snapshot(self, **kwargs):
        async with self._get_factory()() as session:
            snapshot = AccountSnapshot(time=datetime.utcnow(), **kwargs)
            session.add(snapshot)
            await session.commit()

    async def get_balance_history(self, days: int = 30) -> List[Dict]:
        async with self._get_factory()() as session:
            since = datetime.utcnow() - timedelta(days=days)
            stmt = (
                select(AccountSnapshot)
                .where(AccountSnapshot.time >= since)
                .order_by(AccountSnapshot.time)
            )
            result = await session.execute(stmt)
            snapshots = result.scalars().all()
            return [
                {"time": s.time.isoformat(), "total_balance": s.total_balance,
                 "available_balance": s.available_balance, "unrealized_pnl": s.unrealized_pnl}
                for s in snapshots
            ]


_repository: Optional[QuantRepository] = None


def get_repository() -> QuantRepository:
    global _repository
    if _repository is None:
        _repository = QuantRepository()
    return _repository
