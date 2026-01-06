"""Test risk-managed strategy implementation."""
from models.market_data import MarketData
from models.signal import SignalAction
from core.strategies.base_strategy import BaseStrategy
import numpy as np


class RiskManagedMovingAverageEnvelope(BaseStrategy):
    """
    MovingAverageEnvelope with 8% stop-loss and 15% take-profit.
    
    This is a test implementation to demonstrate risk management integration.
    """
    
    def __init__(self, config=None):
        super().__init__(config)
        self.lookback_period = self.config.get('lookback_period', 20)
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.08)  # 8% stop-loss
        self.take_profit_pct = self.config.get('take_profit_pct', 0.15)  # 15% take-profit
        self.envelope_pct = self.config.get('envelope_pct', 0.05)  # 5% envelope
    
    def generate_signal(self, market_data: MarketData) -> SignalAction:
        """Generate signal with risk management."""
        # Check if we have enough data
        if not market_data.bars or len(market_data.bars) < self.lookback_period:
            return SignalAction.HOLD
        
        closes = [bar.close for bar in market_data.bars]
        current_close = closes[-1]
        
        # Calculate moving average
        ma = np.mean(closes[-self.lookback_period:])
        
        # Envelope bands
        upper_band = ma * (1 + self.envelope_pct)
        lower_band = ma * (1 - self.envelope_pct)
        
        # Check stop-loss/take-profit if in position
        if self.in_position and self.entry_price:
            # Check stop-loss
            if self.apply_stop_loss(self.entry_price, current_close, self.stop_loss_pct):
                self.in_position = False
                entry = self.entry_price
                self.entry_price = None
                return SignalAction.SELL
            
            # Check take-profit
            if self.apply_take_profit(self.entry_price, current_close, self.take_profit_pct):
                self.in_position = False
                entry = self.entry_price
                self.entry_price = None
                return SignalAction.SELL
        
        # Original trading logic
        if current_close <= lower_band and not self.in_position:
            # Buy signal: price is below lower envelope
            self.in_position = True
            self.entry_price = current_close
            return SignalAction.BUY
        elif current_close >= upper_band and self.in_position:
            # Sell signal: price is above upper envelope
            self.in_position = False
            self.entry_price = None
            return SignalAction.SELL
        
        return SignalAction.HOLD

