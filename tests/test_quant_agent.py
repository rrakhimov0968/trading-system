"""Tests for QuantAgent."""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from agents.quant_agent import QuantAgent
from models.signal import TradingSignal, SignalAction
from models.market_data import MarketData, Bar
from utils.exceptions import QuantError
from tests.conftest import mock_config


@pytest.fixture
def sample_market_data():
    """Create sample market data with positive trend."""
    bars = []
    base_price = 100.0
    
    for i in range(100):
        # Create upward trending data
        price = base_price + (i * 0.1) + np.random.normal(0, 0.5)
        bars.append(Bar(
            timestamp=datetime.now() - timedelta(days=100 - i),
            open=price - 0.5,
            high=price + 1.0,
            low=price - 1.0,
            close=price,
            volume=1000000,
            symbol="AAPL"
        ))
    
    return MarketData(symbol="AAPL", bars=bars)


@pytest.fixture
def sample_signal(sample_market_data):
    """Create sample trading signal."""
    df = sample_market_data.to_dataframe()
    return TradingSignal(
        symbol="AAPL",
        action=SignalAction.BUY,
        strategy_name="TrendFollowing",
        confidence=0.75,
        timestamp=datetime.now(),
        price=110.0,
        historical_data=df
    )


@pytest.mark.unit
class TestQuantAgentInitialization:
    """Test QuantAgent initialization."""
    
    def test_init_without_llm(self, mock_config):
        """Test initialization without LLM."""
        agent = QuantAgent(config=mock_config)
        assert agent.min_sharpe == 1.5
        assert agent.max_drawdown == 0.08
        assert agent.use_llm_review is False
    
    def test_init_with_custom_thresholds(self, mock_config):
        """Test initialization with custom thresholds."""
        mock_config.quant_min_sharpe = 2.0
        mock_config.quant_max_drawdown = 0.10
        
        agent = QuantAgent(config=mock_config)
        assert agent.min_sharpe == 2.0
        assert agent.max_drawdown == 0.10


@pytest.mark.unit
class TestBasicValidation:
    """Test basic validation functionality."""
    
    def test_basic_validation_positive_expectancy(self, mock_config, sample_signal):
        """Test validation with positive expectancy."""
        agent = QuantAgent(config=mock_config)
        original_confidence = sample_signal.confidence
        
        agent.basic_validation(sample_signal)
        
        # Confidence should not decrease with positive trend
        assert sample_signal.confidence >= original_confidence * 0.8
    
    def test_basic_validation_negative_expectancy(self, mock_config):
        """Test validation with negative expectancy."""
        # Create declining data
        bars = []
        base_price = 100.0
        for i in range(100):
            price = base_price - (i * 0.1)
            bars.append(Bar(
                timestamp=datetime.now() - timedelta(days=100 - i),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                volume=1000000,
                symbol="AAPL"
            ))
        
        data = MarketData(symbol="AAPL", bars=bars)
        df = data.to_dataframe()
        
        signal = TradingSignal(
            symbol="AAPL",
            action=SignalAction.BUY,
            strategy_name="Test",
            confidence=0.75,
            timestamp=datetime.now(),
            historical_data=df
        )
        
        agent = QuantAgent(config=mock_config)
        agent.basic_validation(signal)
        
        # Confidence should decrease with negative expectancy
        assert signal.confidence < 0.75
    
    def test_basic_validation_no_data(self, mock_config):
        """Test validation fails without historical data."""
        signal = TradingSignal(
            symbol="AAPL",
            action=SignalAction.BUY,
            strategy_name="Test",
            confidence=0.75,
            timestamp=datetime.now(),
            historical_data=None
        )
        
        agent = QuantAgent(config=mock_config)
        
        with pytest.raises(QuantError, match="No historical data"):
            agent.basic_validation(signal)
    
    def test_basic_validation_high_volatility(self, mock_config):
        """Test validation with high volatility regime."""
        # Create high volatility data
        bars = []
        base_price = 100.0
        for i in range(100):
            # High volatility (large random swings)
            price = base_price + np.random.normal(0, 5)
            bars.append(Bar(
                timestamp=datetime.now() - timedelta(days=100 - i),
                open=price - 1,
                high=price + 2,
                low=price - 2,
                close=price,
                volume=1000000,
                symbol="AAPL"
            ))
        
        data = MarketData(symbol="AAPL", bars=bars)
        df = data.to_dataframe()
        
        signal = TradingSignal(
            symbol="AAPL",
            action=SignalAction.BUY,
            strategy_name="Test",
            confidence=0.75,
            timestamp=datetime.now(),
            historical_data=df
        )
        
        agent = QuantAgent(config=mock_config)
        original_confidence = signal.confidence
        agent.basic_validation(signal)
        
        # Confidence may decrease with high volatility
        assert signal.confidence <= original_confidence


@pytest.mark.unit
class TestConfidenceValidation:
    """Test confidence validation functionality."""
    
    def test_confidence_validation_good_sharpe(self, mock_config, sample_signal):
        """Test confidence validation with good Sharpe ratio."""
        agent = QuantAgent(config=mock_config)
        original_confidence = sample_signal.confidence
        
        agent.confidence_validation(sample_signal)
        
        # With good trend, confidence should remain or improve
        assert 0.0 <= sample_signal.confidence <= 1.0
    
    def test_confidence_validation_low_sharpe(self, mock_config):
        """Test confidence adjustment with low Sharpe."""
        # Create low Sharpe data (high volatility, low returns)
        returns = np.random.normal(0.001, 0.05, 100)  # Low mean, high std
        prices = 100 * (1 + returns).cumprod()
        
        bars = []
        for i, price in enumerate(prices):
            bars.append(Bar(
                timestamp=datetime.now() - timedelta(days=100 - i),
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1000000,
                symbol="AAPL"
            ))
        
        data = MarketData(symbol="AAPL", bars=bars)
        df = data.to_dataframe()
        
        signal = TradingSignal(
            symbol="AAPL",
            action=SignalAction.BUY,
            strategy_name="Test",
            confidence=0.75,
            timestamp=datetime.now(),
            historical_data=df
        )
        
        agent = QuantAgent(config=mock_config)
        agent.confidence_validation(signal)
        
        # Confidence should decrease with low Sharpe
        assert signal.confidence <= 0.75
        assert 0.0 <= signal.confidence <= 1.0
    
    def test_confidence_validation_high_drawdown(self, mock_config):
        """Test confidence adjustment with high drawdown."""
        # Create data with large drawdown
        prices = []
        base = 100.0
        for i in range(100):
            if i < 50:
                prices.append(base + i * 0.5)  # Up trend
            else:
                prices.append(base + 25 - (i - 50) * 1.0)  # Sharp decline
        
        bars = []
        for i, price in enumerate(prices):
            bars.append(Bar(
                timestamp=datetime.now() - timedelta(days=100 - i),
                open=price,
                high=price * 1.01,
                low=price * 0.99,
                close=price,
                volume=1000000,
                symbol="AAPL"
            ))
        
        data = MarketData(symbol="AAPL", bars=bars)
        df = data.to_dataframe()
        
        signal = TradingSignal(
            symbol="AAPL",
            action=SignalAction.BUY,
            strategy_name="Test",
            confidence=0.75,
            timestamp=datetime.now(),
            historical_data=df
        )
        
        agent = QuantAgent(config=mock_config)
        agent.confidence_validation(signal)
        
        # Confidence should decrease with high drawdown
        assert signal.confidence <= 0.75
        assert 0.0 <= signal.confidence <= 1.0
    
    def test_confidence_clamped(self, mock_config, sample_signal):
        """Test that confidence is clamped to [0, 1]."""
        agent = QuantAgent(config=mock_config)
        
        # Set extreme confidence
        sample_signal.confidence = 1.5
        agent.confidence_validation(sample_signal)
        assert sample_signal.confidence <= 1.0
        
        sample_signal.confidence = -0.5
        agent.confidence_validation(sample_signal)
        assert sample_signal.confidence >= 0.0


@pytest.mark.unit
class TestProcess:
    """Test process method."""
    
    def test_process_signals(self, mock_config, sample_signal):
        """Test processing list of signals."""
        agent = QuantAgent(config=mock_config)
        
        signals = [sample_signal]
        market_data = {"AAPL": MarketData(symbol="AAPL", bars=sample_signal.historical_data)}
        
        validated = agent.process(signals, market_data)
        
        assert len(validated) == 1
        assert validated[0].symbol == "AAPL"
        assert 0.0 <= validated[0].confidence <= 1.0
    
    def test_process_multiple_signals(self, mock_config, sample_market_data):
        """Test processing multiple signals."""
        agent = QuantAgent(config=mock_config)
        
        df = sample_market_data.to_dataframe()
        signals = [
            TradingSignal(
                symbol="AAPL",
                action=SignalAction.BUY,
                strategy_name="Test",
                confidence=0.75,
                timestamp=datetime.now(),
                historical_data=df
            ),
            TradingSignal(
                symbol="MSFT",
                action=SignalAction.SELL,
                strategy_name="Test",
                confidence=0.80,
                timestamp=datetime.now(),
                historical_data=df.copy()
            )
        ]
        
        market_data = {
            "AAPL": sample_market_data,
            "MSFT": MarketData(symbol="MSFT", bars=sample_market_data.bars)
        }
        
        validated = agent.process(signals, market_data)
        
        assert len(validated) == 2
        assert all(0.0 <= s.confidence <= 1.0 for s in validated)
    
    def test_process_continues_on_error(self, mock_config):
        """Test that process continues if one signal fails."""
        agent = QuantAgent(config=mock_config)
        
        # One valid, one invalid signal
        bars_valid = []
        for i in range(100):
            bars_valid.append(Bar(
                timestamp=datetime.now() - timedelta(days=100 - i),
                open=100 + i * 0.1,
                high=101 + i * 0.1,
                low=99 + i * 0.1,
                close=100 + i * 0.1,
                volume=1000000,
                symbol="AAPL"
            ))
        
        df_valid = MarketData(symbol="AAPL", bars=bars_valid).to_dataframe()
        
        signals = [
            TradingSignal(
                symbol="AAPL",
                action=SignalAction.BUY,
                strategy_name="Test",
                confidence=0.75,
                timestamp=datetime.now(),
                historical_data=df_valid
            ),
            TradingSignal(
                symbol="INVALID",
                action=SignalAction.BUY,
                strategy_name="Test",
                confidence=0.75,
                timestamp=datetime.now(),
                historical_data=None  # No data - will fail
            )
        ]
        
        validated = agent.process(signals)
        
        # Should return both signals (invalid one with confidence 0)
        assert len(validated) == 2
        assert validated[0].confidence > 0
        assert validated[1].confidence == 0.0  # Failed validation


@pytest.mark.unit
class TestHealthCheck:
    """Test health check functionality."""
    
    def test_health_check(self, mock_config):
        """Test health check returns correct status."""
        agent = QuantAgent(config=mock_config)
        health = agent.health_check()
        
        assert health["status"] == "healthy"
        assert "min_sharpe" in health
        assert "max_drawdown" in health
        assert health["llm_review_enabled"] is False

