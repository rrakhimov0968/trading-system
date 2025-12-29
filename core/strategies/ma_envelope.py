"""Moving Average Envelope strategy using bands around MA."""
import pandas as pd
from core.strategies.base_strategy import BaseStrategy
from models.market_data import MarketData
from models.signal import SignalAction


class MovingAverageEnvelope(BaseStrategy):
    """
    Moving Average Envelope strategy using percentage bands around moving average.
    
    Signals:
    - BUY: Price touches or goes below lower envelope (oversold)
    - SELL: Price touches or goes above upper envelope (overbought)
    - HOLD: Otherwise
    """
    
    def generate_signal(self, market_data: MarketData) -> SignalAction:
        """
        Generate signal based on moving average envelope.
        
        Args:
            market_data: MarketData with price bars
        
        Returns:
            SignalAction
        """
        self._validate_data(market_data, min_bars=50)
        df = self._get_dataframe(market_data)
        
        # Calculate moving average and envelope
        ma_period = self.config.get('ma_period', 20)
        envelope_percent = self.config.get('envelope_percent', 2.5)  # 2.5% bands
        
        df['MA'] = df['close'].rolling(window=ma_period).mean()
        df['UpperEnvelope'] = df['MA'] * (1 + envelope_percent / 100)
        df['LowerEnvelope'] = df['MA'] * (1 - envelope_percent / 100)
        
        current_close = df['close'].iloc[-1]
        upper = df['UpperEnvelope'].iloc[-1]
        lower = df['LowerEnvelope'].iloc[-1]
        ma = df['MA'].iloc[-1]
        
        if pd.notna(upper) and pd.notna(lower):
            if current_close <= lower:
                self._log_signal(
                    market_data.symbol,
                    SignalAction.BUY,
                    f"At lower envelope: Price ${current_close:.2f} <= ${lower:.2f} (MA=${ma:.2f})"
                )
                return SignalAction.BUY
            elif current_close >= upper:
                self._log_signal(
                    market_data.symbol,
                    SignalAction.SELL,
                    f"At upper envelope: Price ${current_close:.2f} >= ${upper:.2f} (MA=${ma:.2f})"
                )
                return SignalAction.SELL
        
        return SignalAction.HOLD

