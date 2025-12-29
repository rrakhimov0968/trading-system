"""Sector Rotation strategy based on momentum ranking."""
import pandas as pd
from core.strategies.base_strategy import BaseStrategy
from core.strategies.indicators import calculate_momentum
from models.market_data import MarketData
from models.signal import SignalAction


class SectorRotation(BaseStrategy):
    """
    Sector Rotation strategy based on momentum ranking.
    
    Note: For single-symbol use, this acts as a momentum strategy.
    Full sector rotation requires multiple symbols for ranking.
    
    Signals:
    - BUY: High momentum (top quartile)
    - SELL: Negative momentum
    - HOLD: Otherwise
    """
    
    def generate_signal(self, market_data: MarketData) -> SignalAction:
        """
        Generate signal based on momentum ranking.
        
        Args:
            market_data: MarketData with price bars
        
        Returns:
            SignalAction
        """
        self._validate_data(market_data, min_bars=126)
        df = self._get_dataframe(market_data)
        
        # Calculate momentum
        momentum_period = self.config.get('momentum_period', 126)
        top_quartile_threshold = self.config.get('top_quartile_threshold', 10.0)  # Top 25% typically >10%
        
        df['Momentum'] = calculate_momentum(df['close'], period=momentum_period)
        
        current_momentum = df['Momentum'].iloc[-1]
        
        if pd.notna(current_momentum):
            # High momentum = buy (rotating into strong sector)
            if current_momentum > top_quartile_threshold:
                self._log_signal(
                    market_data.symbol,
                    SignalAction.BUY,
                    f"High momentum: {current_momentum:.2f}% (sector rotation)"
                )
                return SignalAction.BUY
            # Negative momentum = sell (rotating out)
            elif current_momentum < 0:
                self._log_signal(
                    market_data.symbol,
                    SignalAction.SELL,
                    f"Negative momentum: {current_momentum:.2f}%"
                )
                return SignalAction.SELL
        
        return SignalAction.HOLD

