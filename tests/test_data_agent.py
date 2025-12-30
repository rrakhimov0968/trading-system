"""Tests for DataAgent."""
import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
import pandas as pd

from agents.data_agent import DataAgent
from models.market_data import MarketData, Bar
from config.settings import DataProvider, DataProviderConfig
from utils.exceptions import AgentError, ValidationError
from tests.conftest import mock_config


@pytest.mark.unit
class TestDataAgentInitialization:
    """Test DataAgent initialization."""
    
    def test_init_with_yahoo_provider(self, mock_config):
        """Test initialization with Yahoo Finance provider."""
        # Create config with Yahoo provider
        yahoo_config = DataProviderConfig.yahoo_from_env()
        mock_config.data_provider = yahoo_config
        
        agent = DataAgent(config=mock_config)
        assert agent.provider == DataProvider.YAHOO
        assert agent.data_config == yahoo_config
    
    def test_init_fails_without_provider(self):
        """Test initialization fails when no provider is configured."""
        from config.settings import AppConfig, TradingMode, LogLevel, AlpacaConfig
        
        config_no_provider = AppConfig(
            trading_mode=TradingMode.PAPER,
            log_level=LogLevel.INFO,
            alpaca=AlpacaConfig(api_key="test", secret_key="test", paper=True),
            data_provider=None
        )
        
        with pytest.raises(AgentError, match="No data provider configured"):
            DataAgent(config=config_no_provider)
    
    def test_init_with_alpaca_provider(self, mock_config):
        """Test initialization with Alpaca provider."""
        alpaca_config = DataProviderConfig(
            provider=DataProvider.ALPACA,
            api_key="test_key",
            base_url=None
        )
        mock_config.data_provider = alpaca_config
        
        with patch('agents.data_agent.StockHistoricalDataClient') as mock_client:
            agent = DataAgent(config=mock_config)
            assert agent.provider == DataProvider.ALPACA
            mock_client.assert_called_once()


@pytest.mark.unit
class TestDataAgentProcess:
    """Test DataAgent process method."""
    
    @patch('yfinance.Ticker')
    def test_process_yahoo_single_symbol(self, mock_ticker, mock_config):
        """Test processing a single symbol with Yahoo Finance."""
        # Setup mock
        yahoo_config = DataProviderConfig.yahoo_from_env()
        mock_config.data_provider = yahoo_config
        
        # Create mock dataframe
        mock_df = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [102.0, 103.0],
            'Low': [99.0, 100.0],
            'Close': [101.0, 102.0],
            'Volume': [1000000, 1100000]
        }, index=pd.date_range('2024-01-01', periods=2, freq='D'))
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker.return_value = mock_ticker_instance
        
        agent = DataAgent(config=mock_config)
        results = agent.process(["AAPL"], timeframe="1Day", limit=100)
        
        assert "AAPL" in results
        assert isinstance(results["AAPL"], MarketData)
        assert len(results["AAPL"].bars) == 2
    
    @patch('yfinance.Ticker')
    def test_process_yahoo_multiple_symbols(self, mock_ticker, mock_config):
        """Test processing multiple symbols with Yahoo Finance."""
        yahoo_config = DataProviderConfig.yahoo_from_env()
        mock_config.data_provider = yahoo_config
        
        # Setup mock dataframe
        mock_df = pd.DataFrame({
            'Open': [100.0],
            'High': [102.0],
            'Low': [99.0],
            'Close': [101.0],
            'Volume': [1000000]
        }, index=pd.date_range('2024-01-01', periods=1, freq='D'))
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker.return_value = mock_ticker_instance
        
        agent = DataAgent(config=mock_config)
        results = agent.process(["AAPL", "MSFT"], timeframe="1Day", limit=100)
        
        assert len(results) == 2
        assert "AAPL" in results
        assert "MSFT" in results
    
    @patch('yfinance.Ticker')
    def test_process_validates_symbols(self, mock_ticker, mock_config):
        """Test that process validates symbol format."""
        yahoo_config = DataProviderConfig.yahoo_from_env()
        mock_config.data_provider = yahoo_config
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance
        
        agent = DataAgent(config=mock_config)
        
        # Invalid symbol should be skipped, not raise exception
        results = agent.process(["INVALID!!"], timeframe="1Day", limit=100)
        # Should handle gracefully - either empty or skip invalid
        assert isinstance(results, dict)


@pytest.mark.unit
class TestDataAgentYahooFetch:
    """Test Yahoo Finance data fetching."""
    
    @patch('yfinance.Ticker')
    def test_fetch_yahoo_data_success(self, mock_ticker, mock_config):
        """Test successful Yahoo Finance data fetch."""
        yahoo_config = DataProviderConfig.yahoo_from_env()
        mock_config.data_provider = yahoo_config
        
        # Create realistic mock data
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        mock_df = pd.DataFrame({
            'Open': [100.0, 101.0, 102.0, 101.5, 102.5],
            'High': [102.0, 103.0, 104.0, 103.5, 104.5],
            'Low': [99.0, 100.0, 101.0, 100.5, 101.5],
            'Close': [101.0, 102.0, 103.0, 102.5, 103.5],
            'Volume': [1000000, 1100000, 1200000, 1150000, 1250000]
        }, index=dates)
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker.return_value = mock_ticker_instance
        
        agent = DataAgent(config=mock_config)
        market_data = agent._fetch_yahoo_data(
            "AAPL",
            "1Day",
            datetime(2024, 1, 1),
            datetime(2024, 1, 6),
            100
        )
        
        assert isinstance(market_data, MarketData)
        assert market_data.symbol == "AAPL"
        assert len(market_data.bars) == 5
        assert market_data.dataframe is not None
        assert len(market_data.dataframe) == 5
    
    @patch('yfinance.Ticker')
    def test_fetch_yahoo_data_empty(self, mock_ticker, mock_config):
        """Test handling of empty data from Yahoo Finance."""
        yahoo_config = DataProviderConfig.yahoo_from_env()
        mock_config.data_provider = yahoo_config
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance
        
        agent = DataAgent(config=mock_config)
        market_data = agent._fetch_yahoo_data(
            "AAPL",
            "1Day",
            datetime(2024, 1, 1),
            datetime(2024, 1, 6),
            100
        )
        
        assert isinstance(market_data, MarketData)
        assert market_data.bars == []
    
    @patch('yfinance.Ticker')
    def test_fetch_yahoo_data_with_limit(self, mock_ticker, mock_config):
        """Test that limit parameter is respected."""
        yahoo_config = DataProviderConfig.yahoo_from_env()
        mock_config.data_provider = yahoo_config
        
        # Create 10 days of data
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        mock_df = pd.DataFrame({
            'Open': [100.0] * 10,
            'High': [102.0] * 10,
            'Low': [99.0] * 10,
            'Close': [101.0] * 10,
            'Volume': [1000000] * 10
        }, index=dates)
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker.return_value = mock_ticker_instance
        
        agent = DataAgent(config=mock_config)
        market_data = agent._fetch_yahoo_data(
            "AAPL",
            "1Day",
            datetime(2024, 1, 1),
            datetime(2024, 1, 11),
            5  # Limit to 5 bars
        )
        
        assert len(market_data.bars) == 5  # Should be limited to 5


@pytest.mark.unit
class TestDataAgentCache:
    """Test caching functionality."""
    
    @patch('yfinance.Ticker')
    def test_cache_hit(self, mock_ticker, mock_config):
        """Test that cached data is returned when available."""
        yahoo_config = DataProviderConfig(
            provider=DataProvider.YAHOO,
            cache_ttl_seconds=300  # 5 minutes
        )
        mock_config.data_provider = yahoo_config
        
        mock_df = pd.DataFrame({
            'Open': [100.0],
            'High': [102.0],
            'Low': [99.0],
            'Close': [101.0],
            'Volume': [1000000]
        }, index=pd.date_range('2024-01-01', periods=1, freq='D'))
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker.return_value = mock_ticker_instance
        
        agent = DataAgent(config=mock_config)
        
        # First call - should fetch
        results1 = agent.process(["AAPL"], timeframe="1Day", limit=100)
        
        # Second call - should use cache
        results2 = agent.process(["AAPL"], timeframe="1Day", limit=100)
        
        # Ticker.history should only be called once
        assert mock_ticker_instance.history.call_count == 1
        assert "AAPL" in results1
        assert "AAPL" in results2


@pytest.mark.unit
class TestDataAgentHealthCheck:
    """Test health check functionality."""
    
    @patch('yfinance.Ticker')
    def test_health_check_healthy(self, mock_ticker, mock_config):
        """Test health check when agent is healthy."""
        yahoo_config = DataProviderConfig.yahoo_from_env()
        mock_config.data_provider = yahoo_config
        
        mock_df = pd.DataFrame({
            'Open': [100.0],
            'High': [102.0],
            'Low': [99.0],
            'Close': [101.0],
            'Volume': [1000000]
        }, index=pd.date_range('2024-01-01', periods=1, freq='D'))
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker.return_value = mock_ticker_instance
        
        agent = DataAgent(config=mock_config)
        health = agent.health_check()
        
        assert health["status"] == "healthy"
        assert health["provider"] == "yahoo"
        assert health["data_accessible"] is True
    
    @patch('yfinance.Ticker')
    def test_health_check_unhealthy(self, mock_ticker, mock_config):
        """Test health check when agent is unhealthy."""
        yahoo_config = DataProviderConfig.yahoo_from_env()
        mock_config.data_provider = yahoo_config
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.side_effect = Exception("API Error")
        mock_ticker.return_value = mock_ticker_instance
        
        agent = DataAgent(config=mock_config)
        health = agent.health_check()
        
        assert health["status"] == "unhealthy"
        assert health["data_accessible"] is False
        assert "error" in health


@pytest.mark.unit
@pytest.mark.asyncio
class TestDataAgentAsync:
    """Test async functionality of DataAgent."""
    
    @patch('yfinance.Ticker')
    async def test_fetch_data_async(self, mock_ticker, mock_config):
        """Test async fetch for a single symbol."""
        yahoo_config = DataProviderConfig.yahoo_from_env()
        mock_config.data_provider = yahoo_config
        
        mock_df = pd.DataFrame({
            'Open': [100.0, 101.0],
            'High': [102.0, 103.0],
            'Low': [99.0, 100.0],
            'Close': [101.0, 102.0],
            'Volume': [1000000, 1100000]
        }, index=pd.date_range('2024-01-01', periods=2, freq='D'))
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker.return_value = mock_ticker_instance
        
        agent = DataAgent(config=mock_config)
        market_data = await agent.fetch_data_async("AAPL", "1Day", limit=100)
        
        assert isinstance(market_data, MarketData)
        assert market_data.symbol == "AAPL"
        assert len(market_data.bars) == 2
    
    @patch('yfinance.Ticker')
    async def test_process_async_multiple_symbols(self, mock_ticker, mock_config):
        """Test async processing of multiple symbols in parallel."""
        yahoo_config = DataProviderConfig.yahoo_from_env()
        mock_config.data_provider = yahoo_config
        
        mock_df = pd.DataFrame({
            'Open': [100.0],
            'High': [102.0],
            'Low': [99.0],
            'Close': [101.0],
            'Volume': [1000000]
        }, index=pd.date_range('2024-01-01', periods=1, freq='D'))
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker.return_value = mock_ticker_instance
        
        agent = DataAgent(config=mock_config)
        results = await agent.process_async(["AAPL", "MSFT", "GOOGL"], timeframe="1Day", limit=100)
        
        assert len(results) == 3
        assert "AAPL" in results
        assert "MSFT" in results
        assert "GOOGL" in results
        assert all(isinstance(data, MarketData) for data in results.values())
    
    @patch('yfinance.Ticker')
    async def test_process_queue(self, mock_ticker, mock_config):
        """Test queue-based async processing."""
        yahoo_config = DataProviderConfig.yahoo_from_env()
        mock_config.data_provider = yahoo_config
        
        mock_df = pd.DataFrame({
            'Open': [100.0],
            'High': [102.0],
            'Low': [99.0],
            'Close': [101.0],
            'Volume': [1000000]
        }, index=pd.date_range('2024-01-01', periods=1, freq='D'))
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker.return_value = mock_ticker_instance
        
        agent = DataAgent(config=mock_config)
        queue = await agent.process_queue(["AAPL", "MSFT"], timeframe="1Day", limit=100)
        
        # Collect results from queue
        results = {}
        while not queue.empty():
            symbol, data = await queue.get()
            if data is not None:
                results[symbol] = data
        
        assert len(results) == 2
        assert "AAPL" in results
        assert "MSFT" in results
    
    @patch('yfinance.Ticker')
    async def test_async_cache_usage(self, mock_ticker, mock_config):
        """Test that async methods use cache."""
        yahoo_config = DataProviderConfig(
            provider=DataProvider.YAHOO,
            cache_ttl_seconds=300
        )
        mock_config.data_provider = yahoo_config
        
        mock_df = pd.DataFrame({
            'Open': [100.0],
            'High': [102.0],
            'Low': [99.0],
            'Close': [101.0],
            'Volume': [1000000]
        }, index=pd.date_range('2024-01-01', periods=1, freq='D'))
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_df
        mock_ticker.return_value = mock_ticker_instance
        
        agent = DataAgent(config=mock_config)
        
        # First call - should fetch
        await agent.fetch_data_async("AAPL", "1Day", limit=100)
        
        # Second call - should use cache
        await agent.fetch_data_async("AAPL", "1Day", limit=100)
        
        # Ticker.history should only be called once
        assert mock_ticker_instance.history.call_count == 1
    
    async def test_cleanup_async_resources(self, mock_config):
        """Test cleanup of async resources."""
        yahoo_config = DataProviderConfig.yahoo_from_env()
        mock_config.data_provider = yahoo_config
        
        agent = DataAgent(config=mock_config)
        
        # Initialize async session
        session = await agent._get_async_session()
        assert session is not None
        assert not session.closed
        
        # Cleanup
        await agent.cleanup_async_resources()
        
        # Session should be closed
        assert agent._async_session is None or agent._async_session.closed

