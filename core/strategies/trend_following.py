"""Trend Following strategy using moving average crossover."""
import pandas as pd
from core.strategies.base_strategy import BaseStrategy
from models.market_data import MarketData
from models.signal import SignalAction


class TrendFollowing(BaseStrategy):
    """
    Trend Following strategy using moving average crossover.
    
    Signals:
    - BUY: Price > MA200 AND MA50 > MA200 (uptrend)
    - SELL: Price < MA200 (downtrend)
    - HOLD: Otherwise
    """
    
    def generate_signal(self, market_data: MarketData) -> SignalAction:
        """
        Generate signal based on moving average crossover.
        
        Args:
            market_data: MarketData with price bars
        
        Returns:
            SignalAction
        """
        self._validate_data(market_data, min_bars=200)
        df = self._get_dataframe(market_data)
        
        # Calculate moving averages
        df['MA50'] = df['close'].rolling(window=50).mean()
        df['MA200'] = df['close'].rolling(window=200).mean()
        
        # Get latest values
        current_close = df['close'].iloc[-1]
        ma50 = df['MA50'].iloc[-1]
        ma200 = df['MA200'].iloc[-1]
        
        # Trend following logic
        if pd.notna(ma200):
            if current_close > ma200 and ma50 > ma200:
                self._log_signal(
                    market_data.symbol,
                    SignalAction.BUY,
                    f"Uptrend: Price ${current_close:.2f} > MA200 ${ma200:.2f}, MA50 ${ma50:.2f} > MA200"
                )
                return SignalAction.BUY
            elif current_close < ma200:
                self._log_signal(
                    market_data.symbol,
                    SignalAction.SELL,
                    f"Downtrend: Price ${current_close:.2f} < MA200 ${ma200:.2f}"
                )
                return SignalAction.SELL
        
        return SignalAction.HOLD

