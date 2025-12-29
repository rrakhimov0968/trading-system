"""Breakout strategy using Donchian Channels."""
import pandas as pd
from core.strategies.base_strategy import BaseStrategy
from core.strategies.indicators import calculate_donchian_channels
from models.market_data import MarketData
from models.signal import SignalAction


class Breakout(BaseStrategy):
    """
    Breakout strategy using Donchian Channels.
    
    Signals:
    - BUY: Price breaks above upper Donchian channel
    - SELL: Price breaks below lower Donchian channel
    - HOLD: Otherwise
    """
    
    def generate_signal(self, market_data: MarketData) -> SignalAction:
        """
        Generate signal based on Donchian channel breakouts.
        
        Args:
            market_data: MarketData with price bars
        
        Returns:
            SignalAction
        """
        self._validate_data(market_data, min_bars=60)
        df = self._get_dataframe(market_data)
        
        # Calculate Donchian channels
        period = self.config.get('donchian_period', 20)
        upper, lower = calculate_donchian_channels(df['high'], df['low'], period=period)
        
        current_close = df['close'].iloc[-1]
        current_high = df['high'].iloc[-1]
        current_low = df['low'].iloc[-1]
        upper_band = upper.iloc[-1]
        lower_band = lower.iloc[-1]
        
        if pd.notna(upper_band) and pd.notna(lower_band):
            # Breakout above upper channel
            if current_high > upper_band:
                self._log_signal(
                    market_data.symbol,
                    SignalAction.BUY,
                    f"Breakout: High ${current_high:.2f} > Upper ${upper_band:.2f}"
                )
                return SignalAction.BUY
            # Breakdown below lower channel
            elif current_low < lower_band:
                self._log_signal(
                    market_data.symbol,
                    SignalAction.SELL,
                    f"Breakdown: Low ${current_low:.2f} < Lower ${lower_band:.2f}"
                )
                return SignalAction.SELL
        
        return SignalAction.HOLD

