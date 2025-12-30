"""Data agent for fetching market data from various providers."""
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import pandas as pd
import yfinance as yf
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest, StockTradesRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.common.exceptions import APIError as AlpacaAPIError
import asyncio
import aiohttp
from asyncio import Queue

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
        
        # Async session for HTTP requests (initialized lazily)
        self._async_session: Optional[aiohttp.ClientSession] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
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
            
            market_data = MarketData(symbol=symbol, bars=bars)
            
            # Validate data quality before returning
            if not self._validate_market_data(market_data, symbol):
                raise AgentError(
                    f"Data validation failed for {symbol}",
                    correlation_id=self._correlation_id
                )
            
            return market_data
            
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
            
            # Validate data quality before returning
            if not self._validate_market_data(market_data, symbol):
                raise AgentError(
                    f"Data validation failed for {symbol}",
                    correlation_id=self._correlation_id
                )
            
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
    
    def _validate_market_data(self, data: MarketData, symbol: str) -> bool:
        """
        Validate data quality before passing to strategies.
        
        Args:
            data: MarketData to validate
            symbol: Stock symbol for logging
        
        Returns:
            True if data is valid, False otherwise
        """
        from datetime import timedelta
        import pytz
        
        # Check for sufficient data
        if not data.bars or len(data.bars) < 20:
            self.log_warning(
                f"Insufficient data for {symbol}: "
                f"{len(data.bars) if data.bars else 0} bars (minimum 20 required)"
            )
            return False
        
        latest_bar = data.bars[-1]
        
        # Check for stale data (more than 10 minutes old for intraday, or 1 day for daily)
        # For daily data, allow up to 2 days old (weekend/holiday tolerance)
        now = datetime.now(pytz.UTC) if latest_bar.timestamp.tzinfo else datetime.utcnow()
        
        # Determine max age based on timeframe
        if hasattr(latest_bar, 'timeframe') and latest_bar.timeframe:
            if 'Day' in latest_bar.timeframe or 'D' in latest_bar.timeframe:
                max_age = timedelta(days=2)  # Daily data - 2 days tolerance
            else:
                max_age = timedelta(minutes=10)  # Intraday - 10 minutes
        else:
            # Default to 1 day if timeframe not specified
            max_age = timedelta(days=1)
        
        bar_time = latest_bar.timestamp
        if bar_time.tzinfo:
            bar_time_utc = bar_time.astimezone(pytz.UTC)
        else:
            bar_time_utc = pytz.UTC.localize(bar_time)
        
        age = now - bar_time_utc
        if age > max_age:
            self.log_warning(
                f"Stale data for {symbol}: latest bar is {age.total_seconds() / 3600:.1f} hours old "
                f"(max allowed: {max_age.total_seconds() / 3600:.1f} hours)"
            )
            return False
        
        # Check for anomalous values
        if latest_bar.volume <= 0:
            self.log_warning(f"Invalid volume for {symbol}: {latest_bar.volume}")
            return False
        
        if latest_bar.close <= 0:
            self.log_warning(f"Invalid close price for {symbol}: {latest_bar.close}")
            return False
        
        # Check for reasonable OHLC relationships
        if latest_bar.high < latest_bar.low:
            self.log_warning(
                f"Invalid OHLC for {symbol}: high ({latest_bar.high}) < low ({latest_bar.low})"
            )
            return False
        
        if latest_bar.close > latest_bar.high or latest_bar.close < latest_bar.low:
            self.log_warning(
                f"Invalid close price for {symbol}: close ({latest_bar.close}) "
                f"outside high-low range [{latest_bar.low}, {latest_bar.high}]"
            )
            return False
        
        if latest_bar.open > latest_bar.high or latest_bar.open < latest_bar.low:
            self.log_warning(
                f"Invalid open price for {symbol}: open ({latest_bar.open}) "
                f"outside high-low range [{latest_bar.low}, {latest_bar.high}]"
            )
            return False
        
        self.log_debug(f"Data validation passed for {symbol}: {len(data.bars)} bars, latest at {latest_bar.timestamp}")
        return True
    
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
    
    # ========================================================================
    # Async Methods
    # ========================================================================
    
    async def _get_async_session(self) -> aiohttp.ClientSession:
        """Get or create async HTTP session."""
        if self._async_session is None or self._async_session.closed:
            self._async_session = aiohttp.ClientSession()
        return self._async_session
    
    async def _get_event_loop(self) -> asyncio.AbstractEventLoop:
        """Get current event loop."""
        try:
            # Try to get the running loop first (Python 3.7+)
            return asyncio.get_running_loop()
        except RuntimeError:
            # No running loop, get the current event loop
            return asyncio.get_event_loop()
    
    async def cleanup_async_resources(self) -> None:
        """Clean up async resources."""
        if self._async_session and not self._async_session.closed:
            await self._async_session.close()
            self._async_session = None
    
    async def fetch_data_async(
        self,
        symbol: str,
        timeframe: str = "1Day",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> MarketData:
        """
        Fetch market data asynchronously for a single symbol.
        
        Args:
            symbol: Stock symbol
            timeframe: Bar timeframe
            start_date: Start date for historical data
            end_date: End date for historical data
            limit: Maximum number of bars to return
        
        Returns:
            MarketData object
        
        Raises:
            AgentError: If data fetching fails
        """
        try:
            validated_symbol = validate_symbol(symbol)
            
            # Check cache first
            cache_key = f"{validated_symbol}_{timeframe}_{limit}"
            cached_data = self._get_from_cache(cache_key)
            if cached_data:
                self.log_debug(f"Using cached data for {validated_symbol}")
                return cached_data
            
            # Fetch data using async wrapper around sync methods
            # Since yfinance and Alpaca clients are sync, we wrap them in executor
            loop = await self._get_event_loop()
            
            if self.provider == DataProvider.ALPACA:
                market_data = await loop.run_in_executor(
                    None,
                    self._fetch_alpaca_data,
                    validated_symbol,
                    timeframe,
                    start_date,
                    end_date,
                    limit
                )
            elif self.provider == DataProvider.POLYGON:
                market_data = await loop.run_in_executor(
                    None,
                    self._fetch_polygon_data,
                    validated_symbol,
                    timeframe,
                    start_date,
                    end_date,
                    limit
                )
            elif self.provider == DataProvider.YAHOO:
                market_data = await loop.run_in_executor(
                    None,
                    self._fetch_yahoo_data,
                    validated_symbol,
                    timeframe,
                    start_date,
                    end_date,
                    limit
                )
            else:
                raise AgentError(
                    f"Unsupported data provider: {self.provider}",
                    correlation_id=self._correlation_id
                )
            
            # Validate data quality before caching and returning
            if not self._validate_market_data(market_data, validated_symbol):
                raise AgentError(
                    f"Data validation failed for {validated_symbol}",
                    correlation_id=self._correlation_id
                )
            
            # Validate data quality before caching and returning
            if not self._validate_market_data(market_data, validated_symbol):
                raise AgentError(
                    f"Data validation failed for {validated_symbol}",
                    correlation_id=self._correlation_id
                )
            
            # Cache the result
            self._cache_data(cache_key, market_data)
            
            return market_data
            
        except Exception as e:
            self.log_exception(f"Async fetch failed for {symbol}", e)
            raise AgentError(
                f"Async fetch failed for {symbol}: {str(e)}",
                correlation_id=self._correlation_id
            ) from e
    
    async def process_async(
        self,
        symbols: List[str],
        timeframe: str = "1Day",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> Dict[str, MarketData]:
        """
        Fetch market data asynchronously for multiple symbols in parallel.
        
        This method uses asyncio.gather to fetch multiple symbols concurrently,
        which is much faster than sequential fetching.
        
        Args:
            symbols: List of stock symbols
            timeframe: Bar timeframe
            start_date: Start date for historical data
            end_date: End date for historical data
            limit: Maximum number of bars to return
        
        Returns:
            Dictionary mapping symbol to MarketData
        """
        self.generate_correlation_id()
        self.log_info(
            f"Fetching market data asynchronously for {len(symbols)} symbols",
            symbols=symbols,
            timeframe=timeframe,
            provider=self.provider.value
        )
        
        # Create async tasks for all symbols
        tasks = [
            self.fetch_data_async(symbol, timeframe, start_date, end_date, limit)
            for symbol in symbols
        ]
        
        # Fetch all symbols concurrently
        try:
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            results = {}
            for symbol, result in zip(symbols, results_list):
                if isinstance(result, Exception):
                    self.log_exception(f"Failed to fetch data for {symbol}", result)
                    continue
                results[symbol] = result
            
            self.log_info(
                f"Successfully fetched data for {len(results)}/{len(symbols)} symbols asynchronously",
                successful_symbols=list(results.keys())
            )
            
            return results
            
        except Exception as e:
            self.log_exception("Error in async data fetching", e)
            raise AgentError(
                f"Async data fetching failed: {str(e)}",
                correlation_id=self._correlation_id
            ) from e
    
    async def process_queue(
        self,
        symbols: List[str],
        timeframe: str = "1Day",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        queue: Optional[Queue] = None
    ) -> Queue:
        """
        Fetch data for multiple symbols and put results in a queue.
        
        This method is useful for producer-consumer patterns where you want
        to process data as it becomes available rather than waiting for all.
        
        Args:
            symbols: List of stock symbols
            timeframe: Bar timeframe
            start_date: Start date for historical data
            end_date: End date for historical data
            limit: Maximum number of bars to return
            queue: Optional asyncio Queue to use (creates new one if not provided)
        
        Returns:
            Queue containing (symbol, MarketData) tuples
        """
        if queue is None:
            queue = Queue()
        
        async def fetch_and_queue(symbol: str):
            """Fetch data for a symbol and add to queue."""
            try:
                data = await self.fetch_data_async(symbol, timeframe, start_date, end_date, limit)
                await queue.put((symbol, data))
            except Exception as e:
                self.log_exception(f"Failed to fetch and queue data for {symbol}", e)
                await queue.put((symbol, None))  # Put None to indicate failure
        
        # Create tasks for all symbols
        tasks = [fetch_and_queue(symbol) for symbol in symbols]
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return queue
    
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
    
    def __del__(self):
        """Cleanup async resources on deletion."""
        # Check if _async_session exists (may not be initialized)
        try:
            if hasattr(self, '_async_session') and self._async_session and not self._async_session.closed:
                # Try to close session synchronously (best effort)
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Can't close in running loop, schedule cleanup
                        loop.create_task(self._async_session.close())
                    else:
                        loop.run_until_complete(self._async_session.close())
                except (RuntimeError, AttributeError):
                    # No event loop or session already closed
                    pass
        except (AttributeError, RuntimeError):
            # Object partially deleted or no event loop available
            pass

