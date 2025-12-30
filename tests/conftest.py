"""Pytest configuration and shared fixtures."""
import os
import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Generator, Dict
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from config.settings import AppConfig, TradingMode, LogLevel, AlpacaConfig
from agents.execution_agent import ExecutionAgent
from models.market_data import MarketData, Bar
from models.signal import TradingSignal, SignalAction


@pytest.fixture
def mock_config() -> AppConfig:
    """Create a mock configuration for testing."""
    return AppConfig(
        trading_mode=TradingMode.PAPER,
        log_level=LogLevel.DEBUG,
        alpaca=AlpacaConfig(
            api_key="test_api_key",
            secret_key="test_secret_key",
            paper=True
        )
    )


@pytest.fixture
def mock_alpaca_client():
    """Create a mock Alpaca TradingClient."""
    client = Mock()
    
    # Mock account
    mock_account = Mock()
    mock_account.cash = "100000.00"
    mock_account.buying_power = "100000.00"
    mock_account.equity = "100000.00"
    client.get_account.return_value = mock_account
    
    # Mock positions
    client.get_all_positions.return_value = []
    
    # Mock order submission
    mock_order = Mock()
    mock_order.id = "test_order_123"
    mock_order.symbol = "AAPL"
    mock_order.qty = 10
    mock_order.side = Mock(value="buy")
    mock_order.status = Mock(value="new")
    client.submit_order.return_value = mock_order
    
    return client


@pytest.fixture
def execution_agent(mock_config, mock_alpaca_client) -> ExecutionAgent:
    """Create an ExecutionAgent with mocked dependencies."""
    with patch('agents.execution_agent.TradingClient', return_value=mock_alpaca_client):
        agent = ExecutionAgent(config=mock_config)
        agent.client = mock_alpaca_client  # Ensure we use the mock
        yield agent


@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Set up test environment variables."""
    monkeypatch.setenv("ALPACA_API_KEY", "test_key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "test_secret")
    monkeypatch.setenv("TRADING_MODE", "paper")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")


@pytest.fixture
def sample_order_request():
    """Sample order request for testing."""
    return {
        "symbol": "AAPL",
        "quantity": 10,
        "side": "buy",
        "order_type": "market"
    }


@pytest.fixture
def correlation_id() -> str:
    """Generate a test correlation ID."""
    import uuid
    return str(uuid.uuid4())


# ============================================================================
# Pipeline E2E Test Fixtures
# ============================================================================

@pytest.fixture
def mock_market_data() -> Dict[str, MarketData]:
    """Create mock market data for multiple symbols."""
    symbols = ["SPY", "QQQ", "AAPL"]
    market_data = {}
    
    for symbol in symbols:
        bars = []
        base_price = 100.0 if symbol == "SPY" else 150.0
        
        # Create 100 days of historical data with upward trend
        for i in range(100):
            price = base_price + (i * 0.1) + np.random.normal(0, 0.5)
            bars.append(Bar(
                timestamp=datetime.now() - timedelta(days=100 - i),
                open=price - 0.5,
                high=price + 1.0,
                low=price - 1.0,
                close=price,
                volume=1000000 + int(np.random.normal(0, 100000)),
                symbol=symbol
            ))
        
        market_data[symbol] = MarketData(symbol=symbol, bars=bars)
    
    return market_data


@pytest.fixture
def mock_llm_strategy_response():
    """Mock LLM response for strategy selection."""
    return {
        "strategy_name": "TrendFollowing",
        "action": "BUY",
        "confidence": 0.75,
        "reasoning": "Strong upward trend detected"
    }


@pytest.fixture
def mock_groq_client(mocker):
    """Mock Groq client for strategy selection."""
    mock_client = Mock()
    mock_completion = Mock()
    mock_completion.choices = [Mock()]
    mock_completion.choices[0].message = Mock()
    mock_completion.choices[0].message.content = '{"strategy_name": "TrendFollowing", "action": "BUY", "confidence": 0.75, "reasoning": "Strong trend"}'
    mock_client.chat.completions.create.return_value = mock_completion
    return mock_client


@pytest.fixture
def mock_anthropic_client(mocker):
    """Mock Anthropic client for audit reports."""
    mock_client = Mock()
    mock_message = Mock()
    mock_message.content = [Mock(text="Test audit report summary")]
    mock_client.messages.create.return_value = mock_message
    return mock_client


@pytest.fixture
def test_config_with_llms(mock_config):
    """Create test config with all LLM providers configured."""
    from config.settings import LLMConfig
    
    mock_config.groq = LLMConfig(
        provider="groq",
        api_key="test_groq_key",
        model="mixtral-8x7b-32768"
    )
    mock_config.anthropic = LLMConfig(
        provider="anthropic",
        api_key="test_anthropic_key",
        model="claude-3-opus-20240229"
    )
    mock_config.openai = LLMConfig(
        provider="openai",
        api_key="test_openai_key",
        model="gpt-4"
    )
    
    # Add symbols
    mock_config.symbols = ["SPY", "QQQ", "AAPL"]
    
    return mock_config


@pytest.fixture
def sample_trading_signal():
    """Create a sample trading signal for testing."""
    return TradingSignal(
        symbol="SPY",
        action=SignalAction.BUY,
        strategy_name="TrendFollowing",
        confidence=0.75,
        timestamp=datetime.now(),
        price=450.0,
        stop_loss=445.0,
        take_profit=460.0,
        reasoning="Strong upward trend detected"
    )

