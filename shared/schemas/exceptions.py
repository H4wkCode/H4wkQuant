"""
H4wkQuant - Exception Hierarchy
"""


class H4wkQuantError(Exception):
    """Base exception"""
    pass


class InsufficientBalanceError(H4wkQuantError):
    pass


class OrderRejectedError(H4wkQuantError):
    pass


class KillSwitchActiveError(H4wkQuantError):
    pass


class RateLimitError(H4wkQuantError):
    def __init__(self, message: str = "Rate limit exceeded", retry_after: int = 60):
        super().__init__(message)
        self.retry_after = retry_after


class APIKeyMissingError(H4wkQuantError):
    pass


class IPBannedError(H4wkQuantError):
    pass


class CircuitBreakerOpenError(H4wkQuantError):
    pass


class RedisConnectionError(H4wkQuantError):
    pass


class DatabaseError(H4wkQuantError):
    pass


class NoCointegrationError(H4wkQuantError):
    """Pair is not cointegrated - cannot trade"""
    pass


class InsufficientDataError(H4wkQuantError):
    """Not enough historical data for calculation"""
    pass


class NegativeEdgeError(H4wkQuantError):
    """Expected value is negative after costs"""
    pass
