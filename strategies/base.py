"""
H4wkQuant - Base Strategy
Abstract base class for all arbitrage strategies
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, List
from shared.schemas.models import ArbSignal


class BaseStrategy(ABC):
    """
    All strategies must implement:
    - evaluate(): Check if trade conditions are met
    - should_close(): Check if existing position should be closed
    """

    def __init__(self, name: str):
        self.name = name
        self.is_active = True

    @abstractmethod
    async def evaluate(self, market_data: Dict) -> Optional[ArbSignal]:
        """
        Evaluate market data and return signal if conditions met.
        Returns None if no trade opportunity.
        """
        pass

    @abstractmethod
    async def should_close(self, position_data: Dict, market_data: Dict) -> Optional[ArbSignal]:
        """
        Check if existing position should be closed.
        Returns close signal or None.
        """
        pass

    @abstractmethod
    def get_pairs(self) -> List[str]:
        """Return list of pair_ids this strategy monitors"""
        pass
