"""Dual Momentum strategy combining absolute and relative momentum."""
import pandas as pd
from core.strategies.base_strategy import BaseStrategy
from core.strategies.indicators import calculate_momentum
from models.market_data import MarketData
from models.signal import SignalAction


class DualMomentum(BaseStrategy):
    """
    Dual Momentum strategy combining absolute and relative momentum.
    
    Signals:
    - BUY: Positive absolute momentum AND positive relative momentum (vs SMA)
    - SELL: Negative absolute momentum OR negative relative momentum
    - HOLD: Otherwise
    """
    
    def generate_signal(self, market_data: MarketData) -> SignalAction:
        """
        Generate signal based on dual momentum (absolute + relative).
        
        Args:
            market_data: MarketData with price bars
        
        Returns:
            SignalAction
        """
        self._validate_data(market_data, min_bars=126)
        df = self._get_dataframe(market_data)
        
        # Calculate absolute momentum
        momentum_period = self.config.get('momentum_period', 126)
        df['AbsoluteMomentum'] = calculate_momentum(df['close'], period=momentum_period)
        
        # Calculate relative momentum (vs SMA)
        sma_period = self.config.get('sma_period', 50)
        df['SMA'] = df['close'].rolling(window=sma_period).mean()
        df['RelativeMomentum'] = ((df['close'] - df['SMA']) / df['SMA']) * 100
        
        current_abs_mom = df['AbsoluteMomentum'].iloc[-1]
        current_rel_mom = df['RelativeMomentum'].iloc[-1]
        
        if pd.notna(current_abs_mom) and pd.notna(current_rel_mom):
            # Both positive = strong buy signal
            if current_abs_mom > 0 and current_rel_mom > 0:
                self._log_signal(
                    market_data.symbol,
                    SignalAction.BUY,
                    f"Dual momentum: Abs={current_abs_mom:.2f}%, Rel={current_rel_mom:.2f}%"
                )
                return SignalAction.BUY
            # Either negative = sell
            elif current_abs_mom < 0 or current_rel_mom < 0:
                self._log_signal(
                    market_data.symbol,
                    SignalAction.SELL,
                    f"Weak momentum: Abs={current_abs_mom:.2f}%, Rel={current_rel_mom:.2f}%"
                )
                return SignalAction.SELL
        
        return SignalAction.HOLD

