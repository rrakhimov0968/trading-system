"""Mean Reversion strategy using RSI."""
import pandas as pd
from core.strategies.base_strategy import BaseStrategy
from core.strategies.indicators import calculate_rsi
from models.market_data import MarketData
from models.signal import SignalAction


class MeanReversion(BaseStrategy):
    """
    Mean Reversion strategy using RSI (Relative Strength Index).
    
    Signals:
    - BUY: RSI < 30 (oversold)
    - SELL: RSI > 70 (overbought)
    - HOLD: Otherwise
    """
    
    def generate_signal(self, market_data: MarketData) -> SignalAction:
        """
        Generate signal based on RSI.
        
        Args:
            market_data: MarketData with price bars
        
        Returns:
            SignalAction
        """
        self._validate_data(market_data, min_bars=30)
        df = self._get_dataframe(market_data)
        
        # Calculate RSI
        rsi_period = self.config.get('rsi_period', 14)
        oversold = self.config.get('oversold_level', 30)
        overbought = self.config.get('overbought_level', 70)
        
        df['RSI'] = calculate_rsi(df['close'], period=rsi_period)
        
        current_rsi = df['RSI'].iloc[-1]
        
        if pd.notna(current_rsi):
            if current_rsi < oversold:
                self._log_signal(
                    market_data.symbol,
                    SignalAction.BUY,
                    f"Oversold: RSI {current_rsi:.2f} < {oversold}"
                )
                return SignalAction.BUY
            elif current_rsi > overbought:
                self._log_signal(
                    market_data.symbol,
                    SignalAction.SELL,
                    f"Overbought: RSI {current_rsi:.2f} > {overbought}"
                )
                return SignalAction.SELL
        
        return SignalAction.HOLD

