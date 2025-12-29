"""Pytest configuration and shared fixtures."""
import os
import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Generator

from config.settings import AppConfig, TradingMode, LogLevel, AlpacaConfig
from agents.execution_agent import ExecutionAgent


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

