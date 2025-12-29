"""Data agent for fetching market data from various providers."""
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import pandas as pd
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest, StockTradesRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.common.exceptions import APIError as AlpacaAPIError

from agents.base import BaseAgent
from models.market_data import MarketData, Bar, Quote, Trade
from models.validation import validate_symbol
from utils.exceptions import APIError, AgentError
from utils.retry import retry_with_backoff, RetryConfig
from config.settings import DataProvider, DataProviderConfig


class DataAgent(BaseAgent):
    """
    Agent responsible for fetching market data from various providers.
    
    This is a pure code agent with no LLM involvement.
    Supports: Alpaca Data API, Polygon.io, Yahoo Finance
    """
    
    def __init__(self, config=None):
        """
        Initialize the data agent.
        
        Args:
            config: Application configuration. If None, loads from environment.
        """
        super().__init__(config)
        
        self.data_config = self.config.data_provider
        if not self.data_config:
            raise AgentError(
                "No data provider configured. Set POLYGON_API_KEY, ALPACA_API_KEY, or use Yahoo Finance.",
                correlation_id=self._correlation_id
            )
        
        self.provider = self.data_config.provider
        self._cache: Dict[str, tuple] = {}  # Simple in-memory cache: {key: (data, timestamp)}
        
        # Initialize provider-specific clients
        if self.provider == DataProvider.ALPACA:
            try:
                self.alpaca_client = StockHistoricalDataClient(
                    api_key=self.data_config.api_key,
                    secret_key=self.config.alpaca.secret_key,
                    url_override=self.data_config.base_url
                )
                self.log_info(f"DataAgent initialized with Alpaca provider")
            except Exception as e:
                error = self.handle_error(e, context={"provider": "alpaca"})
                raise AgentError(
                    "Failed to initialize Alpaca data client",
                    correlation_id=self._correlation_id
                ) from error
        elif self.provider == DataProvider.POLYGON:
            # Polygon client initialization would go here
            # For now, we'll use Alpaca's Polygon integration if available
            self.log_info(f"DataAgent initialized with Polygon provider")
        elif self.provider == DataProvider.YAHOO:
            # Yahoo Finance requires no initialization
            self.log_info(f"DataAgent initialized with Yahoo Finance provider")
    
    def process(
        self, 
        symbols: List[str], 
        timeframe: str = "1Day",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> Dict[str, MarketData]:
        """
        Fetch market data for multiple symbols.
        
        This is the main entry point for the agent.
        
        Args:
            symbols: List of stock symbols
            timeframe: Bar timeframe (1Min, 5Min, 15Min, 1Hour, 1Day, etc.)
            start_date: Start date for historical data
            end_date: End date for historical data
            limit: Maximum number of bars to return
        
        Returns:
            Dictionary mapping symbol to MarketData
        
        Raises:
            AgentError: If data fetching fails
        """
        self.generate_correlation_id()
        self.log_info(
            f"Fetching market data for {len(symbols)} symbols",
            symbols=symbols,
            timeframe=timeframe,
            provider=self.provider.value
        )
        
        results = {}
        for symbol in symbols:
            try:
                # Validate symbol
                validated_symbol = validate_symbol(symbol)
                
                # Check cache first
                cache_key = f"{validated_symbol}_{timeframe}_{limit}"
                cached_data = self._get_from_cache(cache_key)
                if cached_data:
                    self.log_debug(f"Using cached data for {validated_symbol}")
                    results[validated_symbol] = cached_data
                    continue
                
                # Fetch data based on provider
                market_data = self._fetch_data(
                    validated_symbol,
                    timeframe,
                    start_date,
                    end_date,
                    limit
                )
                
                # Cache the result
                self._cache_data(cache_key, market_data)
                
                results[validated_symbol] = market_data
                
            except Exception as e:
                self.log_exception(f"Failed to fetch data for {symbol}", e)
                # Continue with other symbols instead of failing completely
                continue
        
        self.log_info(
            f"Successfully fetched data for {len(results)}/{len(symbols)} symbols",
            successful_symbols=list(results.keys())
        )
        
        return results
    
    def _fetch_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        limit: int
    ) -> MarketData:
        """Fetch data using the configured provider."""
        if self.provider == DataProvider.ALPACA:
            return self._fetch_alpaca_data(symbol, timeframe, start_date, end_date, limit)
        elif self.provider == DataProvider.POLYGON:
            return self._fetch_polygon_data(symbol, timeframe, start_date, end_date, limit)
        elif self.provider == DataProvider.YAHOO:
            return self._fetch_yahoo_data(symbol, timeframe, start_date, end_date, limit)
        else:
            raise AgentError(
                f"Unsupported data provider: {self.provider}",
                correlation_id=self._correlation_id
            )
    
    @retry_with_backoff(config=RetryConfig(max_attempts=3, initial_delay=1.0))
    def _fetch_alpaca_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        limit: int
    ) -> MarketData:
        """Fetch data from Alpaca Data API."""
        try:
            # Map timeframe string to Alpaca TimeFrame
            tf_mapping = {
                "1Min": TimeFrame.Minute,
                "5Min": TimeFrame(5, TimeFrame.Minute),
                "15Min": TimeFrame(15, TimeFrame.Minute),
                "1Hour": TimeFrame.Hour,
                "1Day": TimeFrame.Day,
            }
            
            alpaca_tf = tf_mapping.get(timeframe, TimeFrame.Day)
            
            # Set default dates if not provided
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # Request bars
            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=alpaca_tf,
                start=start_date,
                end=end_date,
                limit=limit
            )
            
            bars_response = self.alpaca_client.get_stock_bars(request_params)
            
            # Convert to our Bar model
            bars = []
            if symbol in bars_response:
                for bar in bars_response[symbol]:
                    bars.append(Bar(
                        timestamp=bar.timestamp,
                        open=float(bar.open),
                        high=float(bar.high),
                        low=float(bar.low),
                        close=float(bar.close),
                        volume=int(bar.volume),
                        symbol=symbol,
                        timeframe=timeframe
                    ))
            
            return MarketData(symbol=symbol, bars=bars)
            
        except AlpacaAPIError as e:
            self.log_exception(f"Alpaca API error fetching data for {symbol}", e)
            raise APIError(
                f"Alpaca API error: {str(e)}",
                status_code=getattr(e, 'status_code', None),
                correlation_id=self._correlation_id
            ) from e
        except Exception as e:
            self.log_exception(f"Unexpected error fetching Alpaca data for {symbol}", e)
            raise AgentError(
                f"Failed to fetch Alpaca data: {str(e)}",
                correlation_id=self._correlation_id
            ) from e
    
    def _fetch_polygon_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        limit: int
    ) -> MarketData:
        """
        Fetch data from Polygon.io.
        
        Note: Full Polygon implementation would require the polygon-api-client library.
        This is a placeholder that can be extended.
        """
        # For now, fall back to Yahoo Finance if Polygon is selected but not fully implemented
        self.log_warning(
            "Polygon provider selected but not fully implemented. Falling back to Yahoo Finance.",
            symbol=symbol
        )
        return self._fetch_yahoo_data(symbol, timeframe, start_date, end_date, limit)
    
    def _fetch_yahoo_data(
        self,
        symbol: str,
        timeframe: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        limit: int
    ) -> MarketData:
        """Fetch data from Yahoo Finance using yfinance."""
        try:
            # Map timeframe to yfinance interval
            interval_mapping = {
                "1Min": "1m",
                "5Min": "5m",
                "15Min": "15m",
                "1Hour": "1h",
                "1Day": "1d",
                "1Week": "1wk",
                "1Month": "1mo"
            }
            
            yf_interval = interval_mapping.get(timeframe, "1d")
            
            # Set default dates
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                # Default to 30 days for daily, less for intraday
                if timeframe in ["1Day", "1Week", "1Month"]:
                    start_date = end_date - timedelta(days=30)
                else:
                    start_date = end_date - timedelta(days=5)
            
            # Fetch data
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=yf_interval
            )
            
            if df.empty:
                self.log_warning(f"No data returned from Yahoo Finance for {symbol}")
                return MarketData(symbol=symbol, bars=[])
            
            # Limit rows if needed
            if len(df) > limit:
                df = df.tail(limit)
            
            # Convert to Bar objects
            bars = []
            for idx, row in df.iterrows():
                bars.append(Bar(
                    timestamp=idx.to_pydatetime() if hasattr(idx, 'to_pydatetime') else datetime.fromtimestamp(idx.timestamp()),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=int(row['Volume']),
                    symbol=symbol,
                    timeframe=timeframe
                ))
            
            # Create MarketData with both bars and DataFrame
            market_data = MarketData(symbol=symbol, bars=bars, dataframe=df)
            
            return market_data
            
        except Exception as e:
            self.log_exception(f"Error fetching Yahoo Finance data for {symbol}", e)
            raise AgentError(
                f"Failed to fetch Yahoo Finance data: {str(e)}",
                correlation_id=self._correlation_id
            ) from e
    
    def _get_from_cache(self, cache_key: str) -> Optional[MarketData]:
        """Get data from cache if not expired."""
        if cache_key not in self._cache:
            return None
        
        data, cached_time = self._cache[cache_key]
        age = (datetime.now() - cached_time).total_seconds()
        
        if age > self.data_config.cache_ttl_seconds:
            # Cache expired
            del self._cache[cache_key]
            return None
        
        return data
    
    def _cache_data(self, cache_key: str, data: MarketData) -> None:
        """Store data in cache."""
        self._cache[cache_key] = (data, datetime.now())
        
        # Simple cache size management (keep last 100 entries)
        if len(self._cache) > 100:
            # Remove oldest entry
            oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
    
    def get_latest_quote(self, symbol: str) -> Optional[Quote]:
        """
        Get latest quote for a symbol.
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Quote object or None if unavailable
        """
        validated_symbol = validate_symbol(symbol)
        
        if self.provider == DataProvider.ALPACA:
            try:
                request = StockQuotesRequest(symbol_or_symbols=[validated_symbol], limit=1)
                quotes = self.alpaca_client.get_stock_quotes(request)
                if validated_symbol in quotes and quotes[validated_symbol]:
                    quote = quotes[validated_symbol][0]
                    return Quote(
                        symbol=validated_symbol,
                        bid=float(quote.bid_price),
                        ask=float(quote.ask_price),
                        bid_size=int(quote.bid_size),
                        ask_size=int(quote.ask_size),
                        timestamp=quote.timestamp
                    )
            except Exception as e:
                self.log_exception(f"Error fetching quote for {symbol}", e)
        
        # For Yahoo Finance, we'd need to use real-time data API
        # For now, return None
        return None
    
    def health_check(self) -> Dict[str, Any]:
        """Check agent health by attempting to fetch data."""
        health = super().health_check()
        
        try:
            # Try to fetch data for a test symbol (AAPL)
            test_data = self._fetch_data(
                "AAPL",
                "1Day",
                datetime.now() - timedelta(days=1),
                datetime.now(),
                1
            )
            health.update({
                "provider": self.provider.value,
                "data_accessible": True,
                "cache_size": len(self._cache)
            })
        except Exception as e:
            health.update({
                "status": "unhealthy",
                "provider": self.provider.value,
                "data_accessible": False,
                "error": str(e),
                "cache_size": len(self._cache)
            })
        
        return health

