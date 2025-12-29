"""Volatility Breakout strategy using ATR."""
import pandas as pd
from core.strategies.base_strategy import BaseStrategy
from core.strategies.indicators import calculate_atr
from models.market_data import MarketData
from models.signal import SignalAction


class VolatilityBreakout(BaseStrategy):
    """
    Volatility Breakout strategy using Average True Range (ATR).
    
    Signals:
    - BUY: Price breaks above previous high + ATR multiplier
    - SELL: Price breaks below previous low - ATR multiplier
    - HOLD: Otherwise
    """
    
    def generate_signal(self, market_data: MarketData) -> SignalAction:
        """
        Generate signal based on ATR volatility breakouts.
        
        Args:
            market_data: MarketData with price bars
        
        Returns:
            SignalAction
        """
        self._validate_data(market_data, min_bars=30)
        df = self._get_dataframe(market_data)
        
        # Calculate ATR
        atr_period = self.config.get('atr_period', 14)
        atr_multiplier = self.config.get('atr_multiplier', 1.5)
        
        df['ATR'] = calculate_atr(df['high'], df['low'], df['close'], period=atr_period)
        
        # Lookback period for high/low
        lookback = self.config.get('lookback_period', 20)
        df['PreviousHigh'] = df['high'].rolling(window=lookback).max()
        df['PreviousLow'] = df['low'].rolling(window=lookback).min()
        
        current_close = df['close'].iloc[-1]
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        atr = df['ATR'].iloc[-1]
        prev_high = df['PreviousHigh'].iloc[-1]
        prev_low = df['PreviousLow'].iloc[-1]
        
        if pd.notna(atr) and pd.notna(prev_high) and pd.notna(prev_low):
            breakout_level = prev_high + (atr * atr_multiplier)
            breakdown_level = prev_low - (atr * atr_multiplier)
            
            if current_high > breakout_level:
                self._log_signal(
                    market_data.symbol,
                    SignalAction.BUY,
                    f"Volatility breakout: High ${current_high:.2f} > ${breakout_level:.2f}"
                )
                return SignalAction.BUY
            elif current_low < breakdown_level:
                self._log_signal(
                    market_data.symbol,
                    SignalAction.SELL,
                    f"Volatility breakdown: Low ${current_low:.2f} < ${breakdown_level:.2f}"
                )
                return SignalAction.SELL
        
        return SignalAction.HOLD

