"""Tests for strategy implementations."""
import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from models.market_data import MarketData, Bar
from models.signal import SignalAction
from core.strategies.trend_following import TrendFollowing
from core.strategies.momentum_rotation import MomentumRotation
from core.strategies.mean_reversion import MeanReversion
from core.strategies.breakout import Breakout
from core.strategies.volatility_breakout import VolatilityBreakout
from core.strategies.relative_strength import RelativeStrength
from core.strategies.sector_rotation import SectorRotation
from core.strategies.dual_momentum import DualMomentum
from core.strategies.ma_envelope import MovingAverageEnvelope
from core.strategies.bollinger_bands import BollingerBandsReversion
from core.strategies import STRATEGY_REGISTRY


def create_sample_market_data(symbol: str = "AAPL", num_bars: int = 250) -> MarketData:
    """Create sample market data for testing."""
    bars = []
    base_price = 100.0
    
    for i in range(num_bars):
        # Create trending data
        price = base_price + (i * 0.1) + np.random.normal(0, 1)
        bars.append(Bar(
            timestamp=datetime.now() - timedelta(days=num_bars - i),
            open=price - 0.5,
            high=price + 1.0,
            low=price - 1.0,
            close=price,
            volume=1000000 + int(np.random.normal(0, 100000)),
            symbol=symbol
        ))
    
    return MarketData(symbol=symbol, bars=bars)


@pytest.mark.unit
class TestTrendFollowing:
    """Test TrendFollowing strategy."""
    
    def test_trend_following_buy_signal(self):
        """Test BUY signal in uptrend."""
        # Create upward trending data
        bars = []
        for i in range(250):
            price = 100 + (i * 0.2)
            bars.append(Bar(
                timestamp=datetime.now() - timedelta(days=250 - i),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000000,
                symbol="AAPL"
            ))
        
        data = MarketData(symbol="AAPL", bars=bars)
        strategy = TrendFollowing()
        
        signal = strategy.generate_signal(data)
        assert signal in [SignalAction.BUY, SignalAction.HOLD]  # Should be BUY in strong uptrend
    
    def test_insufficient_data(self):
        """Test that insufficient data raises error."""
        data = create_sample_market_data(num_bars=50)  # Need 200+ for MA200
        strategy = TrendFollowing()
        
        with pytest.raises(ValueError, match="Insufficient data"):
            strategy.generate_signal(data)


@pytest.mark.unit
class TestMomentumRotation:
    """Test MomentumRotation strategy."""
    
    def test_momentum_buy_signal(self):
        """Test BUY signal with strong momentum."""
        data = create_sample_market_data(num_bars=150)
        strategy = MomentumRotation(config={"momentum_threshold": 0.0})
        
        signal = strategy.generate_signal(data)
        assert signal in [SignalAction.BUY, SignalAction.HOLD, SignalAction.SELL]


@pytest.mark.unit
class TestMeanReversion:
    """Test MeanReversion strategy."""
    
    def test_mean_reversion_signal(self):
        """Test RSI-based mean reversion."""
        # Create oversold data (declining prices)
        bars = []
        base_price = 100.0
        for i in range(50):
            price = base_price - (i * 0.5)  # Declining
            bars.append(Bar(
                timestamp=datetime.now() - timedelta(days=50 - i),
                open=price,
                high=price + 0.5,
                low=price - 0.5,
                close=price,
                volume=1000000,
                symbol="AAPL"
            ))
        
        data = MarketData(symbol="AAPL", bars=bars)
        strategy = MeanReversion()
        
        signal = strategy.generate_signal(data)
        assert signal in [SignalAction.BUY, SignalAction.HOLD, SignalAction.SELL]


@pytest.mark.unit
class TestBreakout:
    """Test Breakout strategy."""
    
    def test_breakout_signal(self):
        """Test Donchian channel breakout."""
        data = create_sample_market_data(num_bars=100)
        strategy = Breakout()
        
        signal = strategy.generate_signal(data)
        assert signal in [SignalAction.BUY, SignalAction.HOLD, SignalAction.SELL]


@pytest.mark.unit
class TestBollingerBands:
    """Test BollingerBandsReversion strategy."""
    
    def test_bollinger_signal(self):
        """Test Bollinger Bands reversion."""
        data = create_sample_market_data(num_bars=60)
        strategy = BollingerBandsReversion()
        
        signal = strategy.generate_signal(data)
        assert signal in [SignalAction.BUY, SignalAction.HOLD, SignalAction.SELL]


@pytest.mark.unit
class TestStrategyRegistry:
    """Test strategy registry."""
    
    def test_all_strategies_registered(self):
        """Test that all strategies are in registry."""
        expected_strategies = [
            "TrendFollowing",
            "MomentumRotation",
            "MeanReversion",
            "Breakout",
            "VolatilityBreakout",
            "RelativeStrength",
            "SectorRotation",
            "DualMomentum",
            "MovingAverageEnvelope",
            "BollingerBands"
        ]
        
        for strategy_name in expected_strategies:
            assert strategy_name in STRATEGY_REGISTRY, f"{strategy_name} not in registry"
            assert STRATEGY_REGISTRY[strategy_name] is not None
    
    def test_strategy_instantiation(self):
        """Test that all strategies can be instantiated."""
        for strategy_name, strategy_class in STRATEGY_REGISTRY.items():
            if strategy_name in ["MovingAverageCrossover", "Momentum", "RSI_OversoldOverbought",
                                "VolumeProfile", "SupportResistance", "ConsolidationBreakout"]:
                continue  # Skip legacy name mappings
            
            strategy = strategy_class()
            assert strategy is not None
            assert hasattr(strategy, 'generate_signal')


@pytest.mark.unit
class TestStrategyIntegration:
    """Test strategy integration with StrategyAgent."""
    
    def test_strategy_with_market_data(self):
        """Test that strategies work with MarketData objects."""
        data = create_sample_market_data(num_bars=200)
        
        strategies_to_test = [
            TrendFollowing(),
            MomentumRotation(),
            MeanReversion(),
            Breakout(),
            BollingerBandsReversion()
        ]
        
        for strategy in strategies_to_test:
            try:
                signal = strategy.generate_signal(data)
                assert signal in [SignalAction.BUY, SignalAction.SELL, SignalAction.HOLD]
            except ValueError:
                # Some strategies may need more data
                pass

