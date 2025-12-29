"""Bollinger Bands Reversion strategy."""
import pandas as pd
from core.strategies.base_strategy import BaseStrategy
from core.strategies.indicators import calculate_bollinger_bands
from models.market_data import MarketData
from models.signal import SignalAction


class BollingerBandsReversion(BaseStrategy):
    """
    Bollinger Bands Reversion strategy (mean reversion).
    
    Signals:
    - BUY: Price touches or goes below lower Bollinger Band (oversold)
    - SELL: Price touches or goes above upper Bollinger Band (overbought)
    - HOLD: Otherwise
    """
    
    def generate_signal(self, market_data: MarketData) -> SignalAction:
        """
        Generate signal based on Bollinger Bands reversion.
        
        Args:
            market_data: MarketData with price bars
        
        Returns:
            SignalAction
        """
        self._validate_data(market_data, min_bars=50)
        df = self._get_dataframe(market_data)
        
        # Calculate Bollinger Bands
        period = self.config.get('bb_period', 20)
        num_std = self.config.get('bb_std', 2.0)
        
        upper, middle, lower = calculate_bollinger_bands(df['close'], period=period, num_std=num_std)
        
        current_close = df['close'].iloc[-1]
        upper_band = upper.iloc[-1]
        lower_band = lower.iloc[-1]
        middle_band = middle.iloc[-1]
        
        if pd.notna(upper_band) and pd.notna(lower_band):
            if current_close <= lower_band:
                self._log_signal(
                    market_data.symbol,
                    SignalAction.BUY,
                    f"Oversold: Price ${current_close:.2f} <= Lower BB ${lower_band:.2f}"
                )
                return SignalAction.BUY
            elif current_close >= upper_band:
                self._log_signal(
                    market_data.symbol,
                    SignalAction.SELL,
                    f"Overbought: Price ${current_close:.2f} >= Upper BB ${upper_band:.2f}"
                )
                return SignalAction.SELL
        
        return SignalAction.HOLD

