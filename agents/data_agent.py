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
from utils.rate_limiter import get_rate_limiter
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
        
        # Initialize rate limiter for API calls
        # Default: 200 requests/minute for Alpaca (their standard limit)
        max_requests = self.data_config.rate_limit_per_minute if self.data_config else 200
        # Use shared rate limiter name since Alpaca's limit is per account
        # All DataAgent instances must share the same limiter
        self.rate_limiter = get_rate_limiter(
            name="alpaca_data",  # Shared across all DataAgent instances
            max_requests=max_requests,
            window_seconds=60
        )
        self.log_info(f"Rate limiter configured: {max_requests} requests/minute")
        
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
        limit: int = 252  # Default to 1 year of trading days (252) to satisfy strategy requirements
    ) -> Dict[str, MarketData]:
        """
        Fetch market data for multiple symbols.
        
        This is the main entry point for the agent.
        Uses batch fetching for Alpaca provider (1 API call for all symbols).
        
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
        
        # For Alpaca, use batch fetching (1 API call for all symbols)
        if self.provider == DataProvider.ALPACA and len(symbols) > 1:
            return self._fetch_alpaca_batch(symbols, timeframe, start_date, end_date, limit)
        
        # For other providers or single symbol, use sequential fetching
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
        limit: int,
        strict_validation: bool = True
    ) -> MarketData:
        """Fetch data using the configured provider."""
        if self.provider == DataProvider.ALPACA:
            return self._fetch_alpaca_data(symbol, timeframe, start_date, end_date, limit, strict_validation=strict_validation)
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
        limit: int,
        strict_validation: bool = True
    ) -> MarketData:
        """Fetch data from Alpaca Data API."""
        # Acquire rate limiter before making API call
        if self.provider == DataProvider.ALPACA:
            if not self.rate_limiter.acquire(blocking=True, timeout=60.0):
                raise AgentError(
                    f"Rate limiter timeout for {symbol} - too many requests",
                    correlation_id=self._correlation_id
                )
        
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
            # Alpaca API requires timezone-aware dates
            import pytz
            if not end_date:
                end_date = datetime.now(pytz.UTC)
            elif end_date.tzinfo is None:
                # Make timezone-aware if not already
                end_date = pytz.UTC.localize(end_date)
            
            if not start_date:
                # Use 1 year lookback (252 trading days) to satisfy strategies that need 200+ bars
                # 365 calendar days ≈ 252 trading days (accounts for weekends/holidays)
                start_date = end_date - timedelta(days=365)
            elif start_date.tzinfo is None:
                # Make timezone-aware if not already
                start_date = pytz.UTC.localize(start_date)
            
            # Request bars
            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=alpaca_tf,
                start=start_date,
                end=end_date,
                limit=limit
            )
            
            self.log_info(
                f"Alpaca API request for {symbol}: "
                f"timeframe={alpaca_tf}, start={start_date}, end={end_date}, limit={limit}"
            )
            
            bars_response = self.alpaca_client.get_stock_bars(request_params)
            
            # Log what Alpaca actually returned (INFO level for visibility)
            self.log_info(
                f"Alpaca API response for {symbol}: "
                f"type={type(bars_response)}, "
                f"is_tuple={isinstance(bars_response, tuple)}, "
                f"is_dict={isinstance(bars_response, dict)}, "
                f"hasattr_data={hasattr(bars_response, 'data') if bars_response else False}"
            )
            
            # Check if response is None or empty
            if bars_response is None:
                self.log_warning(f"Alpaca returned None for {symbol}")
                raise AgentError(
                    f"Alpaca API returned None for {symbol}",
                    correlation_id=self._correlation_id
                )
            
            # Convert to our Bar model
            bars = []
            
            # Helper function to extract bar data from different formats
            def extract_bar_data(bar_item):
                """Extract bar data from Alpaca bar object, tuple, or dict."""
                # Try Bar object with attributes first
                if hasattr(bar_item, 'timestamp'):
                    return {
                        'timestamp': bar_item.timestamp,
                        'open': float(bar_item.open),
                        'high': float(bar_item.high),
                        'low': float(bar_item.low),
                        'close': float(bar_item.close),
                        'volume': int(bar_item.volume)
                    }
                # Try dict (most common format from Alpaca)
                elif isinstance(bar_item, dict):
                    return {
                        'timestamp': bar_item.get('timestamp') or bar_item.get('t'),
                        'open': float(bar_item.get('open') or bar_item.get('o', 0)),
                        'high': float(bar_item.get('high') or bar_item.get('h', 0)),
                        'low': float(bar_item.get('low') or bar_item.get('l', 0)),
                        'close': float(bar_item.get('close') or bar_item.get('c', 0)),
                        'volume': int(bar_item.get('volume') or bar_item.get('v', 0))
                    }
                # Try tuple (timestamp, open, high, low, close, volume, ...)
                elif isinstance(bar_item, tuple):
                    if len(bar_item) >= 6:
                        return {
                            'timestamp': bar_item[0],
                            'open': float(bar_item[1]),
                            'high': float(bar_item[2]),
                            'low': float(bar_item[3]),
                            'close': float(bar_item[4]),
                            'volume': int(bar_item[5])
                        }
                    else:
                        raise ValueError(f"Invalid tuple format for bar: {bar_item}")
                else:
                    raise ValueError(f"Unsupported bar format: {type(bar_item)}")
            
            # Handle Alpaca response structure: can be tuple ('data', {...}) or dict directly
            response_data = None
            
            if isinstance(bars_response, tuple):
                # Alpaca returns tuple: ('data', {'AAPL': [bars...]})
                if len(bars_response) >= 2:
                    self.log_info(f"Alpaca response is tuple with {len(bars_response)} elements for {symbol}")
                    response_data = bars_response[1]  # Get the dict from tuple
                    self.log_info(f"Extracted response_data type: {type(response_data)}, keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'N/A'}")
                else:
                    self.log_warning(f"Unexpected tuple format for {symbol}: {bars_response}")
                    response_data = None
            elif isinstance(bars_response, dict):
                # Direct dict response
                self.log_info(f"Alpaca response is direct dict for {symbol}")
                response_data = bars_response
            else:
                # Try to convert or access as dict - Alpaca may return response object
                self.log_info(f"Alpaca response is object type {type(bars_response)} for {symbol}, checking attributes...")
                if hasattr(bars_response, 'data'):
                    # Alpaca response object with .data attribute
                    data_attr = bars_response.data
                    self.log_info(f"Found .data attribute: type={type(data_attr)}, is_dict={isinstance(data_attr, dict)}")
                    # .data might be the dict we need, or it might be another tuple
                    if isinstance(data_attr, dict):
                        response_data = data_attr
                    elif isinstance(data_attr, tuple) and len(data_attr) >= 2:
                        # Nested tuple: response.data = ('data', {...})
                        response_data = data_attr[1]
                        self.log_info(f"Extracted from nested tuple in .data attribute")
                    else:
                        self.log_warning(f"Unexpected .data format: {type(data_attr)}")
                        response_data = None
                elif hasattr(bars_response, '__dict__'):
                    response_data = bars_response.__dict__
                    self.log_info(f"Using __dict__ for {symbol}")
                else:
                    self.log_warning(f"Unexpected response format for {symbol}: {type(bars_response)}, no .data or __dict__")
                    # Try to see if it's iterable or has other attributes
                    if hasattr(bars_response, '__iter__'):
                        try:
                            # Maybe it's an iterable of bars?
                            bars_list = list(bars_response)
                            self.log_info(f"Response is iterable, extracted {len(bars_list)} items")
                            # Create a dict with symbol as key
                            response_data = {symbol: bars_list}
                        except Exception as e:
                            self.log_warning(f"Failed to iterate response: {e}")
                            response_data = None
                    else:
                        response_data = None
            
            # Extract bars from response data
            if response_data and isinstance(response_data, dict):
                if symbol in response_data:
                    bar_list = response_data[symbol]
                    self.log_info(f"Found {len(bar_list)} bars for {symbol} in Alpaca response")
                    for bar_item in bar_list:
                        try:
                            bar_data = extract_bar_data(bar_item)
                            bars.append(Bar(
                                timestamp=bar_data['timestamp'],
                                open=bar_data['open'],
                                high=bar_data['high'],
                                low=bar_data['low'],
                                close=bar_data['close'],
                                volume=bar_data['volume'],
                                symbol=symbol,
                                timeframe=timeframe
                            ))
                        except Exception as e:
                            self.log_warning(
                                f"Failed to parse bar for {symbol}: {e}. "
                                f"Bar type: {type(bar_item)}"
                            )
                            continue
                else:
                    self.log_warning(
                        f"Symbol {symbol} not found in Alpaca response dict. "
                        f"Available keys: {list(response_data.keys())}"
                    )
            else:
                if response_data is None:
                    self.log_warning(
                        f"Could not extract response_data for {symbol}. "
                        f"Original response type: {type(bars_response)}"
                    )
                else:
                    self.log_warning(
                        f"Response data is not a dict for {symbol}: {type(response_data)}"
                    )
            
            self.log_info(f"Converted {len(bars)} bars for {symbol} from Alpaca response")
            
            market_data = MarketData(symbol=symbol, bars=bars)
            
            # Validate data quality before returning (unless strict_validation=False)
            # During health checks, we just need to verify API connectivity, not perfect data
            if strict_validation and not self._validate_market_data(market_data, symbol):
                raise AgentError(
                    f"Data validation failed for {symbol}",
                    correlation_id=self._correlation_id
                )
            elif not strict_validation:
                # For non-strict validation, just log a warning if validation fails
                if not self._validate_market_data(market_data, symbol):
                    self.log_warning(
                        f"Data validation warning for {symbol} (non-strict mode - continuing anyway)"
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
    
    @retry_with_backoff(config=RetryConfig(max_attempts=3, initial_delay=1.0))
    def _fetch_alpaca_batch(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        limit: int
    ) -> Dict[str, MarketData]:
        """
        Fetch data for multiple symbols in a single batch API call.
        
        This is much more efficient than sequential fetching:
        - 29 symbols = 1 API call (vs 29 calls)
        - Faster execution (parallel processing on server side)
        - Better rate limit utilization
        
        Args:
            symbols: List of stock symbols to fetch
            timeframe: Bar timeframe
            start_date: Start date
            end_date: End date
            limit: Maximum bars per symbol
            
        Returns:
            Dictionary mapping symbol to MarketData
        """
        # Acquire rate limiter (only need 1 token for batch request)
        if not self.rate_limiter.acquire(blocking=True, timeout=60.0):
            raise AgentError(
                f"Rate limiter timeout for batch request - too many requests",
                correlation_id=self._correlation_id
            )
        
        try:
            # Validate all symbols
            validated_symbols = [validate_symbol(s) for s in symbols]
            
            # Check cache for all symbols first
            results = {}
            symbols_to_fetch = []
            
            for symbol in validated_symbols:
                cache_key = f"{symbol}_{timeframe}_{limit}"
                cached_data = self._get_from_cache(cache_key)
                if cached_data:
                    self.log_debug(f"Using cached data for {symbol}")
                    results[symbol] = cached_data
                else:
                    symbols_to_fetch.append(symbol)
            
            # If all symbols are cached, return early
            if not symbols_to_fetch:
                self.log_info(f"All {len(symbols)} symbols retrieved from cache")
                return results
            
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
            import pytz
            if not end_date:
                end_date = datetime.now(pytz.UTC)
            elif end_date.tzinfo is None:
                end_date = pytz.UTC.localize(end_date)
            
            if not start_date:
                start_date = end_date - timedelta(days=365)
            elif start_date.tzinfo is None:
                start_date = pytz.UTC.localize(start_date)
            
            # Batch request: pass all symbols at once
            self.log_info(
                f"Batch fetching {len(symbols_to_fetch)} symbols from Alpaca "
                f"(1 API call instead of {len(symbols_to_fetch)})"
            )
            
            request_params = StockBarsRequest(
                symbol_or_symbols=symbols_to_fetch,  # Multiple symbols in one request
                timeframe=alpaca_tf,
                start=start_date,
                end=end_date,
                limit=limit
            )
            
            bars_response = self.alpaca_client.get_stock_bars(request_params)
            
            if bars_response is None:
                self.log_warning(f"Alpaca batch request returned None")
                # Fallback to individual fetches for remaining symbols
                for symbol in symbols_to_fetch:
                    try:
                        results[symbol] = self._fetch_alpaca_data(
                            symbol, timeframe, start_date, end_date, limit, strict_validation=False
                        )
                    except Exception as e:
                        self.log_warning(f"Failed to fetch {symbol} in fallback: {e}")
                return results
            
            # Parse batch response (same structure as single symbol, but dict keyed by symbol)
            # Helper function to extract bar data (reuse from _fetch_alpaca_data)
            def extract_bar_data(bar_item):
                if hasattr(bar_item, 'timestamp'):
                    return {
                        'timestamp': bar_item.timestamp,
                        'open': float(bar_item.open),
                        'high': float(bar_item.high),
                        'low': float(bar_item.low),
                        'close': float(bar_item.close),
                        'volume': int(bar_item.volume)
                    }
                elif isinstance(bar_item, dict):
                    return {
                        'timestamp': bar_item.get('timestamp') or bar_item.get('t'),
                        'open': float(bar_item.get('open') or bar_item.get('o', 0)),
                        'high': float(bar_item.get('high') or bar_item.get('h', 0)),
                        'low': float(bar_item.get('low') or bar_item.get('l', 0)),
                        'close': float(bar_item.get('close') or bar_item.get('c', 0)),
                        'volume': int(bar_item.get('volume') or bar_item.get('v', 0))
                    }
                elif isinstance(bar_item, tuple) and len(bar_item) >= 6:
                    return {
                        'timestamp': bar_item[0],
                        'open': float(bar_item[1]),
                        'high': float(bar_item[2]),
                        'low': float(bar_item[3]),
                        'close': float(bar_item[4]),
                        'volume': int(bar_item[5])
                    }
                else:
                    raise ValueError(f"Unsupported bar format: {type(bar_item)}")
            
            # Extract response data (handle different formats)
            response_data = None
            
            if isinstance(bars_response, tuple) and len(bars_response) >= 2:
                response_data = bars_response[1]
            elif isinstance(bars_response, dict):
                response_data = bars_response
            elif hasattr(bars_response, 'data'):
                data_attr = bars_response.data
                if isinstance(data_attr, dict):
                    response_data = data_attr
                elif isinstance(data_attr, tuple) and len(data_attr) >= 2:
                    response_data = data_attr[1]
            
            # Parse bars for each symbol
            if response_data and isinstance(response_data, dict):
                for symbol in symbols_to_fetch:
                    try:
                        if symbol not in response_data:
                            self.log_warning(f"Symbol {symbol} not in batch response, keys: {list(response_data.keys())}")
                            continue
                        
                        bar_list = response_data[symbol]
                        if not bar_list:
                            self.log_warning(f"No bars returned for {symbol} in batch response")
                            continue
                        
                        bars = []
                        for bar_item in bar_list:
                            try:
                                bar_data = extract_bar_data(bar_item)
                                bars.append(Bar(
                                    timestamp=bar_data['timestamp'],
                                    open=bar_data['open'],
                                    high=bar_data['high'],
                                    low=bar_data['low'],
                                    close=bar_data['close'],
                                    volume=bar_data['volume'],
                                    symbol=symbol,
                                    timeframe=timeframe
                                ))
                            except Exception as e:
                                self.log_warning(f"Failed to parse bar for {symbol}: {e}")
                                continue
                        
                        if bars:
                            market_data = MarketData(symbol=symbol, bars=bars)
                            # Cache the result
                            cache_key = f"{symbol}_{timeframe}_{limit}"
                            self._cache_data(cache_key, market_data)
                            results[symbol] = market_data
                            self.log_info(f"Fetched {len(bars)} bars for {symbol} from batch request")
                        else:
                            self.log_warning(f"No valid bars parsed for {symbol}")
                    except Exception as e:
                        self.log_exception(f"Error processing {symbol} from batch response", e)
                        continue
            else:
                self.log_warning(
                    f"Unexpected batch response format: {type(bars_response)}, "
                    f"falling back to individual fetches"
                )
                # Fallback to individual fetches
                for symbol in symbols_to_fetch:
                    try:
                        results[symbol] = self._fetch_alpaca_data(
                            symbol, timeframe, start_date, end_date, limit, strict_validation=False
                        )
                    except Exception as e:
                        self.log_warning(f"Failed to fetch {symbol} in fallback: {e}")
            
            self.log_info(
                f"Batch fetch completed: {len(results)}/{len(symbols)} symbols retrieved "
                f"({len(symbols_to_fetch)} from API, {len(symbols) - len(symbols_to_fetch)} from cache)"
            )
            
            return results
            
        except AlpacaAPIError as e:
            self.log_exception(f"Alpaca API error in batch fetch", e)
            # Fallback to individual fetches
            self.log_info("Falling back to individual symbol fetches")
            results = {}
            for symbol in symbols:
                try:
                    results[symbol] = self._fetch_alpaca_data(
                        symbol, timeframe, start_date, end_date, limit, strict_validation=False
                    )
                except Exception as e:
                    self.log_warning(f"Failed to fetch {symbol}: {e}")
            return results
        except Exception as e:
            self.log_exception(f"Unexpected error in batch fetch", e)
            raise AgentError(
                f"Failed to batch fetch Alpaca data: {str(e)}",
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
                # Use 1 year lookback for daily bars to satisfy strategies that need 200+ bars
                if timeframe in ["1Day", "1Week", "1Month"]:
                    # 365 calendar days ≈ 252 trading days (accounts for weekends/holidays)
                    start_date = end_date - timedelta(days=365)
                else:
                    start_date = end_date - timedelta(days=5)
            
            # Fetch data with error handling for Yahoo Finance API issues
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(
                    start=start_date,
                    end=end_date,
                    interval=yf_interval,
                    timeout=10  # 10 second timeout
                )
            except Exception as yf_error:
                error_msg = str(yf_error)
                # Handle common yfinance errors
                if "Expecting value" in error_msg or "No timezone found" in error_msg:
                    self.log_warning(
                        f"Yahoo Finance API error for {symbol}: {error_msg}. "
                        "This may be temporary (rate limiting or service issue)."
                    )
                else:
                    self.log_warning(f"Yahoo Finance error for {symbol}: {error_msg}")
                # Return empty MarketData instead of raising error
                return MarketData(symbol=symbol, bars=[])
            
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
        limit: int = 252  # Default to 1 year of trading days (252) to satisfy strategy requirements
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
            
            # Acquire rate limiter before making API call (for Alpaca)
            if self.provider == DataProvider.ALPACA:
                if not await self.rate_limiter.acquire_async(blocking=True, timeout=60.0):
                    raise AgentError(
                        f"Rate limiter timeout for {validated_symbol} - too many requests",
                        correlation_id=self._correlation_id
                    )
            
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
        
        # Set default date range if not provided
        # Use 1 year lookback (252 trading days) to satisfy strategies that need 200+ bars
        if not start_date:
            # 365 calendar days ≈ 252 trading days (accounts for weekends/holidays)
            start_date = datetime.now() - timedelta(days=365)
        if not end_date:
            end_date = datetime.now()
        
        # Create async tasks for all symbols
        tasks = [
            self.fetch_data_async(symbol, timeframe, start_date, end_date, limit)
            for symbol in symbols
        ]
        
        # Fetch all symbols concurrently
        try:
            results_list = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results - separate validation errors from real API errors
            results = {}
            failed_symbols = []
            
            for symbol, result in zip(symbols, results_list):
                if isinstance(result, Exception):
                    error_msg = str(result)
                    # Check if it's a validation error (no data available - market closed)
                    if "validation" in error_msg.lower() or "insufficient" in error_msg.lower():
                        self.log_warning(
                            f"No data available for {symbol} from {self.provider.value} "
                            f"(market may be closed): {error_msg}"
                        )
                        failed_symbols.append(symbol)
                    else:
                        self.log_exception(f"Failed to fetch data for {symbol}", result)
                        failed_symbols.append(symbol)
                    continue
                results[symbol] = result
            
            # Try Yahoo Finance fallback for symbols that failed due to validation/no data
            if failed_symbols and self.provider != DataProvider.YAHOO:
                self.log_info(f"Attempting Yahoo Finance fallback for {len(failed_symbols)} symbols...")
                original_provider = self.provider
                original_client = None
                
                # Temporarily switch to Yahoo provider
                try:
                    import yfinance as yf
                    self.provider = DataProvider.YAHOO
                    
                    yahoo_tasks = [
                        self.fetch_data_async(
                            symbol,
                            timeframe,
                            start_date,
                            end_date,
                            limit
                        )
                        for symbol in failed_symbols
                    ]
                    
                    yahoo_results = await asyncio.gather(*yahoo_tasks, return_exceptions=True)
                    
                    for symbol, result in zip(failed_symbols, yahoo_results):
                        if isinstance(result, Exception):
                            self.log_warning(f"Yahoo Finance fallback also failed for {symbol}: {result}")
                        else:
                            results[symbol] = result
                            self.log_info(f"Successfully fetched {symbol} from Yahoo Finance fallback")
                    
                except ImportError:
                    self.log_warning("yfinance not available for fallback")
                except Exception as e:
                    self.log_warning(f"Yahoo Finance fallback failed: {e}")
                finally:
                    # Restore original provider
                    self.provider = original_provider
            
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
            # Try to fetch data for a test symbol with wider date range
            # Use 30 days lookback to ensure we get data even if market was closed recently
            # Use strict_validation=False for health checks - we just need to verify API connectivity
            test_data = self._fetch_data(
                "AAPL",
                "1Day",
                datetime.now() - timedelta(days=30),
                datetime.now(),
                100,  # Request more bars to ensure we get some data
                strict_validation=False  # Don't fail health check due to stale/insufficient data
            )
            
            # Check if we got any data (more tolerant - just need API to work)
            if test_data and test_data.bars and len(test_data.bars) > 0:
                health.update({
                    "provider": self.provider.value,
                    "data_accessible": True,
                    "bars_received": len(test_data.bars),
                    "cache_size": len(self._cache)
                })
            else:
                # API is accessible but no data (market closed/holiday) - still healthy
                health.update({
                    "provider": self.provider.value,
                    "data_accessible": True,
                    "bars_received": 0,
                    "warning": "No data available (market may be closed)",
                    "cache_size": len(self._cache)
                })
        except Exception as e:
            # Only mark unhealthy if it's an API/auth error, not data validation
            error_msg = str(e)
            if "validation" in error_msg.lower() or "insufficient" in error_msg.lower():
                # Data validation failure - API works but no data available
                # This is OK during market closed times
                health.update({
                    "provider": self.provider.value,
                    "data_accessible": True,
                    "warning": f"API accessible but no data: {error_msg}",
                    "cache_size": len(self._cache)
                })
            else:
                # Real API/connection error - mark unhealthy
                health.update({
                    "status": "unhealthy",
                    "provider": self.provider.value,
                    "data_accessible": False,
                    "error": error_msg,
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

