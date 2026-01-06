"""Base strategy interface for all trading strategies."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging

from models.market_data import MarketData
from models.signal import SignalAction


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All strategies must inherit from this and implement generate_signal().
    Strategies are deterministic - same input always produces same output.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the strategy.
        
        Args:
            config: Strategy-specific configuration (thresholds, periods, etc.)
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def generate_signal(self, market_data: MarketData) -> SignalAction:
        """
        Generate a deterministic trading signal based on market data.
        
        Args:
            market_data: MarketData object with bars and dataframe
        
        Returns:
            SignalAction: BUY, SELL, or HOLD
        
        Raises:
            ValueError: If data is insufficient for the strategy
        """
        pass
    
    def _validate_data(self, market_data: MarketData, min_bars: int = 50) -> None:
        """
        Validate that market data has sufficient bars for analysis.
        
        Args:
            market_data: MarketData to validate
            min_bars: Minimum number of bars required
        
        Raises:
            ValueError: If data is insufficient
        """
        if not market_data.bars or len(market_data.bars) < min_bars:
            raise ValueError(
                f"Insufficient data: need at least {min_bars} bars, "
                f"got {len(market_data.bars) if market_data.bars else 0}"
            )
    
    def _get_dataframe(self, market_data: MarketData) -> 'pd.DataFrame':
        """
        Get or create DataFrame from MarketData.
        
        Args:
            market_data: MarketData object
        
        Returns:
            DataFrame with OHLCV data indexed by timestamp
        """
        import pandas as pd
        
        df = market_data.to_dataframe()
        if df.empty:
            raise ValueError("Market data DataFrame is empty")
        return df
    
    def _log_signal(self, symbol: str, action: SignalAction, reason: Optional[str] = None) -> None:
        """
        Log signal generation.
        
        Args:
            symbol: Stock symbol
            action: Generated signal action
            reason: Optional reason for the signal
        """
        msg = f"Strategy {self.__class__.__name__} generated {action.value} for {symbol}"
        if reason:
            msg += f": {reason}"
        self.logger.debug(msg)
    
    def apply_stop_loss(self, entry_price: float, current_price: float, stop_loss_pct: float = 0.08) -> bool:
        """
        Apply stop-loss to limit drawdown.
        
        Args:
            entry_price: Entry price of the position
            current_price: Current market price
            stop_loss_pct: Stop-loss percentage (default: 8% = 0.08)
        
        Returns:
            True if stop-loss is hit, False otherwise
        """
        return current_price <= entry_price * (1 - stop_loss_pct)
    
    def apply_take_profit(self, entry_price: float, current_price: float, take_profit_pct: float = 0.15) -> bool:
        """
        Apply take-profit to lock in gains.
        
        Args:
            entry_price: Entry price of the position
            current_price: Current market price
            take_profit_pct: Take-profit percentage (default: 15% = 0.15)
        
        Returns:
            True if take-profit is hit, False otherwise
        """
        return current_price >= entry_price * (1 + take_profit_pct)
    
    def reset_position_state(self):
        """Reset position tracking state. Useful for backtesting."""
        # Note: Position state tracking should be handled by the backtesting framework
        # This method is provided for strategies that want to track state internally
        pass

