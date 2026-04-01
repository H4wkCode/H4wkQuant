"""
H4wkQuant - Bybit REST Client
HMAC-SHA256 signed REST API for Bybit linear futures.
"""
import hashlib
import hmac
import time
import json
from typing import Dict, Optional
from urllib.parse import urlencode

import aiohttp
from loguru import logger


class BybitRestClient:
    """Bybit Futures REST API client."""

    BASE_URL = "https://api.bybit.com"
    TESTNET_URL = "https://api-testnet.bybit.com"

    def __init__(self, api_key: str = "", secret_key: str = "", testnet: bool = False):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = self.TESTNET_URL if testnet else self.BASE_URL
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _sign(self, timestamp: str, params: str) -> str:
        """Generate HMAC-SHA256 signature."""
        sign_str = f"{timestamp}{self.api_key}5000{params}"
        return hmac.new(
            self.secret_key.encode(), sign_str.encode(), hashlib.sha256
        ).hexdigest()

    async def _request(self, method: str, endpoint: str, params: dict = None, signed: bool = False) -> dict:
        session = await self._get_session()
        url = f"{self.base_url}{endpoint}"
        headers = {"Content-Type": "application/json"}

        if signed and self.api_key:
            timestamp = str(int(time.time() * 1000))
            if method == "GET":
                param_str = urlencode(params) if params else ""
            else:
                param_str = json.dumps(params) if params else ""
            signature = self._sign(timestamp, param_str)
            headers.update({
                "X-BAPI-API-KEY": self.api_key,
                "X-BAPI-SIGN": signature,
                "X-BAPI-TIMESTAMP": timestamp,
                "X-BAPI-RECV-WINDOW": "5000",
            })

        try:
            if method == "GET":
                async with session.get(url, params=params, headers=headers) as resp:
                    data = await resp.json()
            else:
                async with session.post(url, json=params, headers=headers) as resp:
                    data = await resp.json()

            if data.get("retCode") != 0:
                logger.error(f"Bybit API error: {data.get('retMsg')} ({endpoint})")
            return data.get("result", {})
        except Exception as e:
            logger.error(f"Bybit request failed: {e}")
            return {}

    async def get_tickers(self, symbol: str = None) -> dict:
        """Get ticker info (price, funding rate)."""
        params = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/v5/market/tickers", params)

    async def get_orderbook(self, symbol: str, limit: int = 25) -> dict:
        return await self._request("GET", "/v5/market/orderbook", {
            "category": "linear", "symbol": symbol, "limit": limit
        })

    async def get_kline(self, symbol: str, interval: str = "1", limit: int = 200) -> dict:
        return await self._request("GET", "/v5/market/kline", {
            "category": "linear", "symbol": symbol, "interval": interval, "limit": limit
        })

    async def place_order(self, symbol: str, side: str, qty: float,
                           order_type: str = "Market", price: float = None,
                           reduce_only: bool = False) -> dict:
        params = {
            "category": "linear",
            "symbol": symbol,
            "side": side,  # "Buy" or "Sell"
            "orderType": order_type,
            "qty": str(qty),
            "reduceOnly": reduce_only,
        }
        if price and order_type == "Limit":
            params["price"] = str(price)
            params["timeInForce"] = "GTC"
        else:
            params["timeInForce"] = "IOC"

        return await self._request("POST", "/v5/order/create", params, signed=True)

    async def cancel_all_orders(self, symbol: str) -> dict:
        return await self._request("POST", "/v5/order/cancel-all", {
            "category": "linear", "symbol": symbol,
        }, signed=True)

    async def set_leverage(self, symbol: str, leverage: int) -> dict:
        return await self._request("POST", "/v5/position/set-leverage", {
            "category": "linear", "symbol": symbol,
            "buyLeverage": str(leverage), "sellLeverage": str(leverage),
        }, signed=True)

    async def get_wallet_balance(self) -> dict:
        return await self._request("GET", "/v5/account/wallet-balance", {
            "accountType": "UNIFIED",
        }, signed=True)

    async def get_positions(self, symbol: str = None) -> dict:
        params = {"category": "linear"}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/v5/position/list", params, signed=True)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


_bybit_client: Optional[BybitRestClient] = None


def get_bybit_client() -> BybitRestClient:
    global _bybit_client
    if _bybit_client is None:
        from shared.config.settings import settings
        _bybit_client = BybitRestClient(
            api_key=settings.bybit.api_key if hasattr(settings, "bybit") else "",
            secret_key=settings.bybit.secret_key if hasattr(settings, "bybit") else "",
            testnet=settings.bybit.testnet if hasattr(settings, "bybit") else False,
        )
    return _bybit_client
