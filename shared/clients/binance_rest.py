"""
H4wkQuant - Binance REST Client
Futures API for orderbook, funding rate, OI, account, orders
"""
import time
import hmac
import hashlib
from urllib.parse import urlencode
from typing import Dict, List, Optional
import aiohttp
from loguru import logger

from shared.config.settings import settings
from shared.utils.rate_limiter import get_rate_limiter
from shared.utils.retry import async_retry


class BinanceRestClient:
    def __init__(self):
        self.base_url = settings.binance.rest_url
        self.api_key = settings.binance.api_key
        self.secret_key = settings.binance.secret_key
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_limiter = get_rate_limiter()

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"X-MBX-APIKEY": self.api_key}
            )
        return self._session

    def _sign(self, params: Dict) -> Dict:
        params["timestamp"] = int(time.time() * 1000)
        query = urlencode(params)
        signature = hmac.new(
            self.secret_key.encode(), query.encode(), hashlib.sha256
        ).hexdigest()
        params["signature"] = signature
        return params

    async def _request(self, method: str, endpoint: str, params: Dict = None, signed: bool = False) -> Dict:
        session = await self._get_session()
        params = params or {}
        if signed:
            params = self._sign(params)
        url = f"{self.base_url}{endpoint}"
        _, data, _ = await self._rate_limiter.execute(session, method, url, params=params)
        return data

    # =========================================================================
    # Market Data
    # =========================================================================

    @async_retry(max_retries=3)
    async def get_orderbook(self, symbol: str, limit: int = 20) -> Dict:
        return await self._request("GET", "/fapi/v1/depth", {"symbol": symbol, "limit": limit})

    @async_retry(max_retries=3)
    async def get_funding_rate(self, symbol: str) -> Dict:
        data = await self._request("GET", "/fapi/v1/fundingRate", {"symbol": symbol, "limit": 1})
        return data[0] if data else {}

    @async_retry(max_retries=3)
    async def get_premium_index(self, symbol: str) -> Dict:
        """Get current funding rate + next funding time"""
        data = await self._request("GET", "/fapi/v1/premiumIndex", {"symbol": symbol})
        return data

    @async_retry(max_retries=3)
    async def get_open_interest(self, symbol: str) -> Dict:
        return await self._request("GET", "/fapi/v1/openInterest", {"symbol": symbol})

    @async_retry(max_retries=3)
    async def get_mark_price(self, symbol: str = None) -> Dict:
        params = {}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/fapi/v1/premiumIndex", params)

    @async_retry(max_retries=3)
    async def get_ticker_24h(self, symbol: str = None) -> Dict:
        params = {}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/fapi/v1/ticker/24hr", params)

    @async_retry(max_retries=3)
    async def get_klines(self, symbol: str, interval: str = "1m", limit: int = 500) -> List:
        return await self._request("GET", "/fapi/v1/klines", {
            "symbol": symbol, "interval": interval, "limit": limit
        })

    @async_retry(max_retries=3)
    async def get_exchange_info(self) -> Dict:
        return await self._request("GET", "/fapi/v1/exchangeInfo")

    # =========================================================================
    # Account & Orders (Signed)
    # =========================================================================

    @async_retry(max_retries=3)
    async def get_account(self) -> Dict:
        return await self._request("GET", "/fapi/v2/account", signed=True)

    @async_retry(max_retries=3)
    async def get_positions(self) -> List[Dict]:
        account = await self.get_account()
        return [p for p in account.get("positions", []) if float(p.get("positionAmt", 0)) != 0]

    @async_retry(max_retries=3)
    async def get_balance(self) -> Dict:
        balances = await self._request("GET", "/fapi/v2/balance", signed=True)
        for b in balances:
            if b["asset"] == "USDT":
                return b
        return {}

    async def place_order(
        self, symbol: str, side: str, order_type: str,
        quantity: float, price: float = None,
        reduce_only: bool = False, **kwargs
    ) -> Dict:
        params = {
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "quantity": f"{quantity:.6f}".rstrip("0").rstrip("."),
        }
        if price and order_type == "LIMIT":
            params["price"] = f"{price:.8f}".rstrip("0").rstrip(".")
            params["timeInForce"] = kwargs.get("timeInForce", "GTC")
        if reduce_only:
            params["reduceOnly"] = "true"
        params.update(kwargs)
        return await self._request("POST", "/fapi/v1/order", params, signed=True)

    async def cancel_order(self, symbol: str, order_id: int) -> Dict:
        return await self._request("DELETE", "/fapi/v1/order", {
            "symbol": symbol, "orderId": order_id
        }, signed=True)

    async def cancel_all_orders(self, symbol: str) -> Dict:
        return await self._request("DELETE", "/fapi/v1/allOpenOrders", {
            "symbol": symbol
        }, signed=True)

    async def set_leverage(self, symbol: str, leverage: int) -> Dict:
        return await self._request("POST", "/fapi/v1/leverage", {
            "symbol": symbol, "leverage": leverage
        }, signed=True)

    async def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> Dict:
        try:
            return await self._request("POST", "/fapi/v1/marginType", {
                "symbol": symbol, "marginType": margin_type
            }, signed=True)
        except Exception as e:
            if "No need to change margin type" in str(e):
                return {"msg": "Already set"}
            raise

    # =========================================================================
    # Cleanup
    # =========================================================================

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


_client: Optional[BinanceRestClient] = None


def get_binance_client(force_new: bool = False) -> BinanceRestClient:
    global _client
    if _client is None or force_new:
        _client = BinanceRestClient()
    return _client
