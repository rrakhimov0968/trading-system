"""Tests for ExecutionAgent."""
import pytest
from unittest.mock import Mock, patch
from alpaca.common.exceptions import APIError as AlpacaAPIError

from agents.execution_agent import ExecutionAgent
from models.enums import OrderSide
from utils.exceptions import ExecutionError, ValidationError, APIError
from tests.conftest import mock_config, mock_alpaca_client


@pytest.mark.unit
class TestExecutionAgentInitialization:
    """Test ExecutionAgent initialization."""
    
    def test_init_with_config(self, mock_config, mock_alpaca_client):
        """Test initialization with provided config."""
        with patch('agents.execution_agent.TradingClient', return_value=mock_alpaca_client):
            agent = ExecutionAgent(config=mock_config)
            assert agent.config == mock_config
            assert agent.client == mock_alpaca_client
    
    def test_init_loads_config_from_env(self, mock_alpaca_client):
        """Test initialization loads config from environment."""
        with patch('agents.execution_agent.TradingClient', return_value=mock_alpaca_client):
            with patch('agents.execution_agent.get_config', return_value=mock_config):
                agent = ExecutionAgent()
                assert agent.config is not None
    
    def test_init_fails_without_api_keys(self):
        """Test initialization fails when API keys are missing."""
        with patch('agents.execution_agent.get_config') as mock_get_config:
            mock_config_no_keys = Mock()
            mock_config_no_keys.alpaca = Mock()
            mock_config_no_keys.alpaca.api_key = None
            mock_config_no_keys.alpaca.secret_key = None
            mock_get_config.return_value = mock_config_no_keys
            
            with pytest.raises(ExecutionError, match="Failed to initialize Alpaca client"):
                ExecutionAgent(config=mock_config_no_keys)


@pytest.mark.unit
class TestExecutionAgentValidation:
    """Test input validation in ExecutionAgent."""
    
    def test_process_validates_symbol(self, execution_agent, sample_order_request):
        """Test that process validates symbol format."""
        sample_order_request["symbol"] = ""
        with pytest.raises(ValidationError, match="Symbol cannot be empty"):
            execution_agent.process(sample_order_request)
    
    def test_process_validates_quantity(self, execution_agent, sample_order_request):
        """Test that process validates quantity."""
        sample_order_request["quantity"] = -5
        with pytest.raises(ValidationError, match="Quantity must be positive"):
            execution_agent.process(sample_order_request)
        
        sample_order_request["quantity"] = 0
        with pytest.raises(ValidationError, match="Quantity must be positive"):
            execution_agent.process(sample_order_request)
    
    def test_process_validates_order_side(self, execution_agent, sample_order_request):
        """Test that process validates order side."""
        sample_order_request["side"] = "invalid"
        with pytest.raises(ValidationError):
            execution_agent.process(sample_order_request)


@pytest.mark.unit
class TestExecutionAgentOrders:
    """Test order execution in ExecutionAgent."""
    
    def test_process_successful_order(self, execution_agent, sample_order_request, mock_alpaca_client):
        """Test successful order processing."""
        result = execution_agent.process(sample_order_request)
        
        assert result["order_id"] == "test_order_123"
        assert result["symbol"] == "AAPL"
        assert result["quantity"] == 10
        assert result["side"] == "buy"
        assert "correlation_id" in result
        
        # Verify order was submitted
        mock_alpaca_client.submit_order.assert_called_once()
    
    def test_place_market_order_buy(self, execution_agent, mock_alpaca_client):
        """Test placing a buy market order."""
        order = execution_agent.place_market_order("AAPL", 10, OrderSide.BUY)
        
        assert order.id == "test_order_123"
        mock_alpaca_client.submit_order.assert_called_once()
    
    def test_place_market_order_sell(self, execution_agent, mock_alpaca_client):
        """Test placing a sell market order."""
        order = execution_agent.place_market_order("AAPL", 10, OrderSide.SELL)
        
        assert order.id == "test_order_123"
        mock_alpaca_client.submit_order.assert_called_once()
    
    def test_place_market_order_api_error(self, execution_agent, mock_alpaca_client):
        """Test handling of API errors when placing orders."""
        mock_alpaca_client.submit_order.side_effect = AlpacaAPIError(
            "API Error", status_code=400
        )
        
        with pytest.raises(ExecutionError, match="Failed to place order"):
            execution_agent.place_market_order("AAPL", 10, OrderSide.BUY)
    
    def test_unsupported_order_type(self, execution_agent, sample_order_request):
        """Test that unsupported order types raise an error."""
        sample_order_request["order_type"] = "limit"
        with pytest.raises(ExecutionError, match="Unsupported order type"):
            execution_agent.process(sample_order_request)


@pytest.mark.unit
class TestExecutionAgentAccount:
    """Test account operations in ExecutionAgent."""
    
    def test_get_account_success(self, execution_agent, mock_alpaca_client):
        """Test successful account retrieval."""
        account = execution_agent.get_account()
        
        assert account.cash == "100000.00"
        assert account.buying_power == "100000.00"
        mock_alpaca_client.get_account.assert_called_once()
    
    def test_get_account_api_error(self, execution_agent, mock_alpaca_client):
        """Test handling of API errors when getting account."""
        mock_alpaca_client.get_account.side_effect = AlpacaAPIError(
            "API Error", status_code=401
        )
        
        with pytest.raises(APIError, match="Alpaca API error"):
            execution_agent.get_account()
    
    def test_get_positions_success(self, execution_agent, mock_alpaca_client):
        """Test successful positions retrieval."""
        positions = execution_agent.get_positions()
        
        assert positions == []
        mock_alpaca_client.get_all_positions.assert_called_once()
    
    def test_get_positions_api_error(self, execution_agent, mock_alpaca_client):
        """Test handling of API errors when getting positions."""
        mock_alpaca_client.get_all_positions.side_effect = AlpacaAPIError(
            "API Error", status_code=500
        )
        
        with pytest.raises(APIError, match="Alpaca API error"):
            execution_agent.get_positions()


@pytest.mark.unit
class TestExecutionAgentHealthCheck:
    """Test health check functionality."""
    
    def test_health_check_healthy(self, execution_agent, mock_alpaca_client):
        """Test health check when agent is healthy."""
        health = execution_agent.health_check()
        
        assert health["status"] == "healthy"
        assert health["agent"] == "ExecutionAgent"
        assert health["account_accessible"] is True
        assert health["trading_mode"] == "paper"
    
    def test_health_check_unhealthy(self, execution_agent, mock_alpaca_client):
        """Test health check when agent is unhealthy."""
        mock_alpaca_client.get_account.side_effect = AlpacaAPIError("API Error")
        
        health = execution_agent.health_check()
        
        assert health["status"] == "unhealthy"
        assert health["account_accessible"] is False
        assert "error" in health


@pytest.mark.unit
class TestExecutionAgentCorrelationID:
    """Test correlation ID functionality."""
    
    def test_process_generates_correlation_id(self, execution_agent, sample_order_request):
        """Test that process generates a correlation ID."""
        assert execution_agent.correlation_id is None
        
        execution_agent.process(sample_order_request)
        
        assert execution_agent.correlation_id is not None
    
    def test_correlation_id_in_result(self, execution_agent, sample_order_request):
        """Test that correlation ID is included in result."""
        result = execution_agent.process(sample_order_request)
        
        assert "correlation_id" in result
        assert result["correlation_id"] == execution_agent.correlation_id

