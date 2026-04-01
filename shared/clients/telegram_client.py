"""
H4wkQuant - Telegram Notifications
Lightweight alert client for arbitrage events
"""
import aiohttp
from typing import Optional
from loguru import logger

from shared.config.settings import settings


class TelegramNotifier:
    def __init__(self, bot_token: str = None, chat_id: str = None):
        self.bot_token = bot_token or settings.monitoring.telegram_bot_token
        self.chat_id = chat_id or settings.monitoring.telegram_chat_id
        self.enabled = settings.monitoring.telegram_enabled and self.bot_token and self.chat_id
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def send(self, message: str, parse_mode: str = "HTML"):
        if not self.enabled:
            return
        try:
            session = await self._get_session()
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            await session.post(url, json={
                "chat_id": self.chat_id,
                "text": message,
                "parse_mode": parse_mode,
            })
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    async def notify_arb_entry(self, pair_id: str, strategy: str, zscore: float,
                                edge: float, size_usd: float):
        await self.send(
            f"<b>ARB ENTRY</b>\n"
            f"Pair: <code>{pair_id}</code>\n"
            f"Strategy: {strategy}\n"
            f"Z-Score: {zscore:.2f}\n"
            f"Edge: {edge:.4f}\n"
            f"Size: ${size_usd:.2f}"
        )

    async def notify_arb_exit(self, pair_id: str, pnl: float, exit_reason: str):
        emoji = "+" if pnl >= 0 else ""
        await self.send(
            f"<b>ARB EXIT</b>\n"
            f"Pair: <code>{pair_id}</code>\n"
            f"PnL: {emoji}${pnl:.4f}\n"
            f"Reason: {exit_reason}"
        )

    async def notify_kill_switch(self, reason: str):
        await self.send(f"<b>KILL SWITCH ACTIVATED</b>\n{reason}")

    async def notify_system(self, message: str):
        await self.send(f"<b>H4wkQuant</b>\n{message}")

    def update_config(self, bot_token: str = None, chat_id: str = None, enabled: bool = None):
        """Update Telegram config at runtime (no restart needed)."""
        if bot_token is not None:
            self.bot_token = bot_token
        if chat_id is not None:
            self.chat_id = chat_id
        if enabled is not None:
            self.enabled = enabled and self.bot_token and self.chat_id
        else:
            self.enabled = bool(self.bot_token and self.chat_id)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


_notifier: Optional[TelegramNotifier] = None


def get_telegram_notifier() -> TelegramNotifier:
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier
