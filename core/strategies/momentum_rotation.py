"""Momentum Rotation strategy based on 6-month momentum."""
import pandas as pd
from core.strategies.base_strategy import BaseStrategy
from core.strategies.indicators import calculate_momentum
from models.market_data import MarketData
from models.signal import SignalAction


class MomentumRotation(BaseStrategy):
    """
    Momentum Rotation strategy based on 6-month momentum.
    
    Signals:
    - BUY: Momentum > threshold (default 5%)
    - SELL: Momentum < 0 (negative momentum)
    - HOLD: Otherwise
    """
    
    def generate_signal(self, market_data: MarketData) -> SignalAction:
        """
        Generate signal based on momentum.
        
        Args:
            market_data: MarketData with price bars
        
        Returns:
            SignalAction
        """
        self._validate_data(market_data, min_bars=126)
        df = self._get_dataframe(market_data)
        
        # Calculate momentum (6 months = ~126 trading days)
        momentum_period = self.config.get('momentum_period', 126)
        momentum_threshold = self.config.get('momentum_threshold', 5.0)  # 5%
        
        df['Momentum'] = calculate_momentum(df['close'], period=momentum_period)
        
        current_momentum = df['Momentum'].iloc[-1]
        
        if pd.notna(current_momentum):
            if current_momentum > momentum_threshold:
                self._log_signal(
                    market_data.symbol,
                    SignalAction.BUY,
                    f"Strong momentum: {current_momentum:.2f}%"
                )
                return SignalAction.BUY
            elif current_momentum < 0:
                self._log_signal(
                    market_data.symbol,
                    SignalAction.SELL,
                    f"Negative momentum: {current_momentum:.2f}%"
                )
                return SignalAction.SELL
        
        return SignalAction.HOLD

