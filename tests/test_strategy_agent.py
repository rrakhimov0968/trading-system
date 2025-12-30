"""Tests for StrategyAgent."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import pandas as pd
import json
from pydantic import ValidationError

from agents.strategy_agent import StrategyAgent, AVAILABLE_STRATEGIES
from models.market_data import MarketData, Bar
from models.signal import TradingSignal, SignalAction
from models.llm_schemas import LLMStrategySelection
from utils.exceptions import AgentError, StrategyError
from tests.conftest import mock_config


@pytest.mark.unit
class TestStrategyAgentInitialization:
    """Test StrategyAgent initialization."""
    
    def test_init_with_groq_config(self, mock_config):
        """Test initialization with Groq configuration."""
        from config.settings import LLMConfig
        
        mock_config.groq = LLMConfig(
            provider="groq",
            api_key="test_key",
            model="mixtral-8x7b-32768"
        )
        
        with patch('agents.strategy_agent.Groq') as mock_groq_class:
            mock_groq_client = Mock()
            mock_groq_class.return_value = mock_groq_client
            
            agent = StrategyAgent(config=mock_config)
            assert agent.model == "mixtral-8x7b-32768"
            assert agent.groq_client == mock_groq_client
    
    def test_init_fails_without_groq_config(self, mock_config):
        """Test initialization fails when Groq config is missing."""
        mock_config.groq = None
        
        with pytest.raises(AgentError, match="Groq configuration not found"):
            StrategyAgent(config=mock_config)


@pytest.mark.unit
class TestMarketContextCalculation:
    """Test market context calculation."""
    
    @patch('agents.strategy_agent.Groq')
    def test_calculate_market_context(self, mock_groq_class, mock_config):
        """Test calculation of market context metrics."""
        from config.settings import LLMConfig
        
        mock_config.groq = LLMConfig(provider="groq", api_key="test")
        mock_groq_class.return_value = Mock()
        
        agent = StrategyAgent(config=mock_config)
        
        # Create mock market data
        dates = pd.date_range('2024-01-01', periods=25, freq='D')
        bars = []
        for i, date in enumerate(dates):
            bars.append(Bar(
                timestamp=date.to_pydatetime(),
                open=100.0 + i,
                high=102.0 + i,
                low=99.0 + i,
                close=101.0 + i,
                volume=1000000,
                symbol="AAPL"
            ))
        
        market_data = MarketData(symbol="AAPL", bars=bars)
        context = agent._calculate_market_context("AAPL", market_data)
        
        assert "current_price" in context
        assert "volatility" in context
        assert "ma_20" in context
        assert "volume_ratio" in context
        assert "regime" in context
        assert context["symbol"] == "AAPL"
        assert context["bars_count"] == 25
    
    @patch('agents.strategy_agent.Groq')
    def test_calculate_context_with_insufficient_data(self, mock_groq_class, mock_config):
        """Test context calculation with insufficient data."""
        from config.settings import LLMConfig
        
        mock_config.groq = LLMConfig(provider="groq", api_key="test")
        mock_groq_class.return_value = Mock()
        
        agent = StrategyAgent(config=mock_config)
        
        # Empty market data
        market_data = MarketData(symbol="AAPL", bars=[])
        context = agent._calculate_market_context("AAPL", market_data)
        
        assert context == {}


@pytest.mark.unit
class TestStrategySelection:
    """Test strategy selection with LLM."""
    
    @patch('agents.strategy_agent.Groq')
    def test_select_strategy_with_llm_success(self, mock_groq_class, mock_config):
        """Test successful strategy selection via LLM."""
        from config.settings import LLMConfig
        
        mock_config.groq = LLMConfig(provider="groq", api_key="test")
        
        # Mock Groq response
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = json.dumps({
            "strategy_name": "MovingAverageCrossover",
            "action": "BUY",
            "confidence": 0.75,
            "reasoning": "Strong upward trend detected"
        })
        
        mock_groq_client = Mock()
        mock_groq_client.chat.completions.create.return_value = mock_completion
        mock_groq_class.return_value = mock_groq_client
        
        agent = StrategyAgent(config=mock_config)
        context = {"symbol": "AAPL", "current_price": 100.0}
        
        result = agent._select_strategy_with_llm("AAPL", context)
        
        assert result["strategy_name"] == "MovingAverageCrossover"
        assert result["action"] == "BUY"
        assert result["confidence"] == 0.75
        assert "reasoning" in result
    
    @patch('agents.strategy_agent.Groq')
    def test_select_strategy_invalid_strategy_name(self, mock_groq_class, mock_config):
        """Test handling of invalid strategy name from LLM (now validated by Pydantic)."""
        from config.settings import LLMConfig
        
        mock_config.groq = LLMConfig(provider="groq", api_key="test")
        
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        # LLM returns invalid strategy name
        mock_completion.choices[0].message.content = json.dumps({
            "strategy_name": "InvalidStrategy",
            "action": "BUY",
            "confidence": 0.75,
            "reasoning": "Test"
        })
        
        mock_groq_client = Mock()
        mock_groq_client.chat.completions.create.return_value = mock_completion
        mock_groq_class.return_value = mock_groq_client
        
        agent = StrategyAgent(config=mock_config)
        context = {"symbol": "AAPL"}
        
        # Pydantic validation should catch this and return safe default
        result = agent._select_strategy_with_llm("AAPL", context)
        
        # Should default to MovingAverageCrossover due to validation failure
        assert result["strategy_name"] == "MovingAverageCrossover"
        assert result["action"] == "HOLD"
    
    @patch('agents.strategy_agent.Groq')
    def test_select_strategy_llm_failure(self, mock_groq_class, mock_config):
        """Test fallback when LLM call fails."""
        from config.settings import LLMConfig
        
        mock_config.groq = LLMConfig(provider="groq", api_key="test")
        
        mock_groq_client = Mock()
        mock_groq_client.chat.completions.create.side_effect = Exception("API Error")
        mock_groq_class.return_value = mock_groq_client
        
        agent = StrategyAgent(config=mock_config)
        context = {"symbol": "AAPL"}
        
        result = agent._select_strategy_with_llm("AAPL", context)
        
        # Should return default fallback
        assert result["strategy_name"] == "MovingAverageCrossover"
        assert result["action"] == "HOLD"
        assert result["confidence"] == 0.3


@pytest.mark.unit
class TestSignalGeneration:
    """Test signal generation."""
    
    @patch('agents.strategy_agent.Groq')
    def test_process_with_valid_data(self, mock_groq_class, mock_config):
        """Test processing market data and generating signals."""
        from config.settings import LLMConfig
        
        mock_config.groq = LLMConfig(provider="groq", api_key="test")
        
        # Create mock market data (need at least 200 bars for some strategies)
        from datetime import timedelta
        bars = []
        start_date = datetime(2024, 1, 1)
        for i in range(250):
            bars.append(Bar(
                timestamp=start_date + timedelta(days=i),
                open=100.0 + i * 0.1,
                high=102.0 + i * 0.1,
                low=99.0 + i * 0.1,
                close=101.0 + i * 0.1,
                volume=1000000,
                symbol="AAPL"
            ))
        
        market_data = {
            "AAPL": MarketData(symbol="AAPL", bars=bars)
        }
        
        # Mock LLM response
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = json.dumps({
            "strategy_name": "Momentum",
            "action": "BUY",
            "confidence": 0.8,
            "reasoning": "Strong momentum detected"
        })
        
        mock_groq_client = Mock()
        mock_groq_client.chat.completions.create.return_value = mock_completion
        mock_groq_class.return_value = mock_groq_client
        
        agent = StrategyAgent(config=mock_config)
        signals = agent.process(market_data)
        
        assert len(signals) == 1
        assert signals[0].symbol == "AAPL"
        assert signals[0].action == SignalAction.BUY
        assert signals[0].strategy_name == "Momentum"
        assert signals[0].confidence == 0.8
    
    @patch('agents.strategy_agent.Groq')
    def test_process_with_insufficient_data(self, mock_groq_class, mock_config):
        """Test processing with insufficient data."""
        from config.settings import LLMConfig
        
        mock_config.groq = LLMConfig(provider="groq", api_key="test")
        mock_groq_class.return_value = Mock()
        
        market_data = {
            "AAPL": MarketData(symbol="AAPL", bars=[])  # No data
        }
        
        agent = StrategyAgent(config=mock_config)
        signals = agent.process(market_data)
        
        # Should skip symbols with insufficient data
        assert len(signals) == 0
    
    @patch('agents.strategy_agent.Groq')
    def test_process_multiple_symbols(self, mock_groq_class, mock_config):
        """Test processing multiple symbols."""
        from config.settings import LLMConfig
        
        mock_config.groq = LLMConfig(provider="groq", api_key="test")
        
        # Create market data for multiple symbols (need at least 200 bars)
        from datetime import timedelta
        start_date = datetime(2024, 1, 1)
        bars_aapl = [Bar(
            timestamp=start_date + timedelta(days=i),
            open=100.0 + i * 0.1,
            high=102.0 + i * 0.1,
            low=99.0 + i * 0.1,
            close=101.0 + i * 0.1,
            volume=1000000,
            symbol="AAPL"
        ) for i in range(250)]
        
        bars_msft = [Bar(
            timestamp=start_date + timedelta(days=i),
            open=200.0 + i * 0.1,
            high=202.0 + i * 0.1,
            low=199.0 + i * 0.1,
            close=201.0 + i * 0.1,
            volume=2000000,
            symbol="MSFT"
        ) for i in range(250)]
        
        market_data = {
            "AAPL": MarketData(symbol="AAPL", bars=bars_aapl),
            "MSFT": MarketData(symbol="MSFT", bars=bars_msft)
        }
        
        # Mock LLM response
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = json.dumps({
            "strategy_name": "TrendFollowing",
            "action": "BUY",
            "confidence": 0.7,
            "reasoning": "Trend detected"
        })
        
        mock_groq_client = Mock()
        mock_groq_client.chat.completions.create.return_value = mock_completion
        mock_groq_class.return_value = mock_groq_client
        
        agent = StrategyAgent(config=mock_config)
        signals = agent.process(market_data)
        
        assert len(signals) == 2
        assert {s.symbol for s in signals} == {"AAPL", "MSFT"}


@pytest.mark.unit
class TestHealthCheck:
    """Test health check functionality."""
    
    @patch('agents.strategy_agent.Groq')
    def test_health_check_healthy(self, mock_groq_class, mock_config):
        """Test health check when agent is healthy."""
        from config.settings import LLMConfig
        
        mock_config.groq = LLMConfig(provider="groq", api_key="test")
        
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message.content = "OK"
        
        mock_groq_client = Mock()
        mock_groq_client.chat.completions.create.return_value = mock_completion
        mock_groq_class.return_value = mock_groq_client
        
        agent = StrategyAgent(config=mock_config)
        health = agent.health_check()
        
        assert health["status"] == "healthy"
        assert health["llm_provider"] == "groq"
        assert health["llm_accessible"] is True
    
    @patch('agents.strategy_agent.Groq')
    def test_health_check_unhealthy(self, mock_groq_class, mock_config):
        """Test health check when agent is unhealthy."""
        from config.settings import LLMConfig
        
        mock_config.groq = LLMConfig(provider="groq", api_key="test")
        
        mock_groq_client = Mock()
        mock_groq_client.chat.completions.create.side_effect = Exception("API Error")
        mock_groq_class.return_value = mock_groq_client
        
        agent = StrategyAgent(config=mock_config)
        health = agent.health_check()
        
        assert health["status"] == "unhealthy"
        assert health["llm_accessible"] is False

