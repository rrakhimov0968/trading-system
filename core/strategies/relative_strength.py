"""Relative Strength strategy comparing to benchmark."""
import pandas as pd
from core.strategies.base_strategy import BaseStrategy
from models.market_data import MarketData
from models.signal import SignalAction


class RelativeStrength(BaseStrategy):
    """
    Relative Strength strategy comparing performance to benchmark (e.g., SPY).
    
    Note: This strategy requires benchmark data. For now, uses price vs SMA as proxy.
    
    Signals:
    - BUY: Outperforming benchmark (price/SMA > threshold)
    - SELL: Underperforming (price/SMA < threshold)
    - HOLD: Otherwise
    """
    
    def generate_signal(self, market_data: MarketData) -> SignalAction:
        """
        Generate signal based on relative strength.
        
        Args:
            market_data: MarketData with price bars
        
        Returns:
            SignalAction
        """
        self._validate_data(market_data, min_bars=50)
        df = self._get_dataframe(market_data)
        
        # Calculate relative strength vs moving average (proxy for benchmark)
        period = self.config.get('strength_period', 50)
        threshold = self.config.get('strength_threshold', 1.05)  # 5% outperformance
        
        df['SMA'] = df['close'].rolling(window=period).mean()
        df['RelativeStrength'] = df['close'] / df['SMA']
        
        current_rs = df['RelativeStrength'].iloc[-1]
        current_close = df['close'].iloc[-1]
        sma = df['SMA'].iloc[-1]
        
        if pd.notna(current_rs):
            if current_rs > threshold:
                self._log_signal(
                    market_data.symbol,
                    SignalAction.BUY,
                    f"Outperforming: RS {current_rs:.2f} (Price ${current_close:.2f} vs SMA ${sma:.2f})"
                )
                return SignalAction.BUY
            elif current_rs < (1.0 / threshold):  # Inverse threshold for underperformance
                self._log_signal(
                    market_data.symbol,
                    SignalAction.SELL,
                    f"Underperforming: RS {current_rs:.2f}"
                )
                return SignalAction.SELL
        
        return SignalAction.HOLD

