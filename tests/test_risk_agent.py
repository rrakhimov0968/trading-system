"""Tests for RiskAgent."""
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from agents.risk_agent import RiskAgent
from models.signal import TradingSignal, SignalAction
from models.market_data import MarketData, Bar
from utils.exceptions import RiskCheckError
from tests.conftest import mock_config


@pytest.fixture
def sample_market_data():
    """Create sample market data."""
    bars = []
    base_price = 100.0
    
    for i in range(50):
        price = base_price + (i * 0.1) + np.random.normal(0, 0.5)
        bars.append(Bar(
            timestamp=datetime.now() - timedelta(days=50 - i),
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
        price=105.0,
        historical_data=df
    )


@pytest.mark.unit
class TestRiskAgentInitialization:
    """Test RiskAgent initialization."""
    
    def test_init_defaults(self, mock_config):
        """Test initialization with default values."""
        agent = RiskAgent(config=mock_config)
        assert agent.max_risk_per_trade == 0.02
        assert agent.max_daily_loss == 0.05
        assert agent.min_confidence == 0.3
        assert agent.use_llm_advisor is False
    
    def test_init_with_custom_thresholds(self, mock_config):
        """Test initialization with custom thresholds."""
        mock_config.risk_max_per_trade = 0.01
        mock_config.risk_max_daily_loss = 0.03
        
        agent = RiskAgent(config=mock_config)
        assert agent.max_risk_per_trade == 0.01
        assert agent.max_daily_loss == 0.03


@pytest.mark.unit
class TestEnforceRules:
    """Test rule enforcement."""
    
    def test_enforce_rules_low_confidence(self, mock_config, sample_signal):
        """Test that low confidence signals are rejected."""
        agent = RiskAgent(config=mock_config)
        sample_signal.confidence = 0.2  # Below minimum
        
        with pytest.raises(RiskCheckError, match="Confidence.*below minimum"):
            agent.enforce_rules(sample_signal)
    
    def test_enforce_rules_passes(self, mock_config, sample_signal):
        """Test that valid signals pass rule checks."""
        agent = RiskAgent(config=mock_config)
        sample_signal.confidence = 0.75  # Above minimum
        
        # Should not raise
        agent.enforce_rules(sample_signal)


@pytest.mark.unit
class TestPositionSizing:
    """Test position sizing calculation."""
    
    def test_calculate_position_sizing(self, mock_config, sample_signal):
        """Test position sizing calculation."""
        agent = RiskAgent(config=mock_config)
        agent._account_balance = 10000.0
        
        agent.calculate_position_sizing(sample_signal)
        
        assert sample_signal.qty is not None
        assert sample_signal.qty > 0
        assert sample_signal.qty <= agent.max_qty
        assert sample_signal.risk_amount is not None
        assert sample_signal.risk_amount > 0
        assert sample_signal.stop_loss is not None
    
    def test_position_sizing_no_data(self, mock_config):
        """Test that missing historical data raises error."""
        signal = TradingSignal(
            symbol="AAPL",
            action=SignalAction.BUY,
            strategy_name="Test",
            confidence=0.75,
            timestamp=datetime.now(),
            historical_data=None
        )
        
        agent = RiskAgent(config=mock_config)
        agent._account_balance = 10000.0
        
        with pytest.raises(RiskCheckError, match="No historical data"):
            agent.calculate_position_sizing(signal)
    
    def test_position_sizing_low_confidence(self, mock_config, sample_signal):
        """Test that low confidence signals are rejected."""
        agent = RiskAgent(config=mock_config)
        agent._account_balance = 10000.0
        sample_signal.confidence = 0.25  # Below minimum
        
        with pytest.raises(RiskCheckError, match="Confidence.*below minimum"):
            agent.calculate_position_sizing(sample_signal)
    
    def test_position_sizing_respects_max_qty(self, mock_config, sample_signal):
        """Test that position sizing respects max_qty limit."""
        agent = RiskAgent(config=mock_config)
        agent._account_balance = 1000000.0  # Large account
        agent.max_qty = 100
        
        agent.calculate_position_sizing(sample_signal)
        
        assert sample_signal.qty <= agent.max_qty
    
    def test_position_sizing_respects_max_risk(self, mock_config, sample_signal):
        """Test that position sizing respects max risk per trade."""
        agent = RiskAgent(config=mock_config)
        agent._account_balance = 10000.0
        
        agent.calculate_position_sizing(sample_signal)
        
        max_risk = agent.max_risk_per_trade * agent._account_balance
        assert sample_signal.risk_amount <= max_risk
    
    def test_position_sizing_daily_loss_limit(self, mock_config, sample_signal):
        """Test that daily loss limit is enforced."""
        agent = RiskAgent(config=mock_config)
        agent._account_balance = 10000.0
        # Set daily loss close to limit
        agent._current_daily_loss = 0.04 * agent._account_balance  # 4% already lost
        
        # This should fail because adding another 2% would exceed 5% daily limit
        with pytest.raises(RiskCheckError, match="daily loss limit"):
            agent.calculate_position_sizing(sample_signal)


@pytest.mark.unit
class TestProcess:
    """Test process method."""
    
    def test_process_approves_valid_signals(self, mock_config, sample_signal):
        """Test that valid signals are approved."""
        agent = RiskAgent(config=mock_config)
        agent._account_balance = 10000.0
        
        signals = [sample_signal]
        approved = agent.process(signals)
        
        assert len(approved) == 1
        assert approved[0].approved is True
        assert approved[0].qty is not None
        assert approved[0].risk_amount is not None
    
    def test_process_rejects_low_confidence(self, mock_config, sample_signal):
        """Test that low confidence signals are rejected."""
        agent = RiskAgent(config=mock_config)
        agent._account_balance = 10000.0
        sample_signal.confidence = 0.2  # Below minimum
        
        signals = [sample_signal]
        approved = agent.process(signals)
        
        # Should not be in approved list (filtered out)
        approved_only = [s for s in approved if s.approved]
        assert len(approved_only) == 0
    
    def test_process_handles_hold_signals(self, mock_config, sample_signal):
        """Test that HOLD signals are passed through."""
        agent = RiskAgent(config=mock_config)
        sample_signal.action = SignalAction.HOLD
        
        signals = [sample_signal]
        approved = agent.process(signals)
        
        assert len(approved) == 1
        assert approved[0].action == SignalAction.HOLD
        assert approved[0].approved is True  # HOLD signals auto-approved
    
    def test_process_multiple_signals(self, mock_config, sample_market_data):
        """Test processing multiple signals."""
        agent = RiskAgent(config=mock_config)
        agent._account_balance = 50000.0
        
        df = sample_market_data.to_dataframe()
        signals = [
            TradingSignal(
                symbol="AAPL",
                action=SignalAction.BUY,
                strategy_name="Test",
                confidence=0.8,
                timestamp=datetime.now(),
                price=105.0,
                historical_data=df
            ),
            TradingSignal(
                symbol="MSFT",
                action=SignalAction.BUY,
                strategy_name="Test",
                confidence=0.7,
                timestamp=datetime.now(),
                price=200.0,
                historical_data=df.copy()
            )
        ]
        
        approved = agent.process(signals)
        
        # Both should be approved (with sufficient account balance)
        approved_only = [s for s in approved if s.approved]
        assert len(approved_only) == 2
        assert all(s.qty is not None for s in approved_only)
    
    def test_process_fetches_account_balance(self, mock_config, sample_signal):
        """Test that process fetches account balance from execution agent."""
        agent = RiskAgent(config=mock_config)
        
        # Mock execution agent
        mock_execution = Mock()
        mock_account = Mock()
        mock_account.equity = 25000.0
        mock_account.cash = 25000.0
        mock_execution.get_account.return_value = mock_account
        
        signals = [sample_signal]
        approved = agent.process(signals, execution_agent=mock_execution)
        
        assert agent._account_balance == 25000.0
        mock_execution.get_account.assert_called_once()


@pytest.mark.unit
class TestResetDailyLoss:
    """Test daily loss reset functionality."""
    
    def test_reset_daily_loss(self, mock_config):
        """Test that daily loss can be reset."""
        agent = RiskAgent(config=mock_config)
        agent._current_daily_loss = 100.0
        
        agent.reset_daily_loss()
        
        assert agent._current_daily_loss == 0.0


@pytest.mark.unit
class TestHealthCheck:
    """Test health check functionality."""
    
    def test_health_check(self, mock_config):
        """Test health check returns correct information."""
        agent = RiskAgent(config=mock_config)
        agent._account_balance = 10000.0
        
        health = agent.health_check()
        
        assert health["status"] == "healthy"
        assert "max_risk_per_trade" in health
        assert "account_balance" in health
        assert health["llm_advisor_enabled"] is False

