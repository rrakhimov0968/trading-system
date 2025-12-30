# DataAgent Async Implementation

## Overview

Enhanced the `DataAgent` to support asynchronous data fetching while maintaining backward compatibility with existing synchronous code.

## Changes Made

### 1. **Added Async Dependencies**
- `asyncio` - Already part of Python standard library
- `aiohttp` - Already in requirements.txt (v3.9.1)
- `pytest-asyncio` - Already in requirements.txt (v0.23.3)

### 2. **New Async Methods**

#### `fetch_data_async()`
- Async version of single symbol data fetching
- Wraps sync provider methods (`_fetch_alpaca_data`, `_fetch_yahoo_data`, etc.) in `asyncio.run_in_executor()`
- Maintains cache checking and validation
- Returns `MarketData` object

#### `process_async()`
- Async version of `process()` method
- Fetches multiple symbols **concurrently** using `asyncio.gather()`
- Significantly faster than sequential fetching
- Returns `Dict[str, MarketData]`

#### `process_queue()`
- Producer-consumer pattern implementation
- Fetches data and puts results into an `asyncio.Queue`
- Useful for processing data as it becomes available
- Returns a queue containing `(symbol, MarketData)` tuples

### 3. **Async Resource Management**

#### `_get_async_session()`
- Lazy initialization of `aiohttp.ClientSession`
- Reuses session if already created and not closed

#### `_get_event_loop()`
- Safely retrieves the current event loop
- Handles both running and non-running loop scenarios

#### `cleanup_async_resources()`
- Properly closes async HTTP sessions
- Should be called when done with async operations

#### `__del__()`
- Cleanup method that attempts to close async resources on object deletion
- Best-effort cleanup (handles errors gracefully)

### 4. **Backward Compatibility**

- **All existing sync methods remain unchanged**
- `process()` method still works synchronously
- Existing code using `DataAgent` continues to work without modifications

## Usage Examples

### Basic Async Fetch

```python
import asyncio
from agents.data_agent import DataAgent
from config.settings import get_config

async def main():
    config = get_config()
    agent = DataAgent(config)
    
    # Fetch single symbol
    market_data = await agent.fetch_data_async("AAPL", "1Day", limit=100)
    print(f"Fetched {len(market_data.bars)} bars")
    
    # Cleanup
    await agent.cleanup_async_resources()

asyncio.run(main())
```

### Concurrent Multi-Symbol Fetching

```python
async def main():
    config = get_config()
    agent = DataAgent(config)
    
    # Fetch multiple symbols in parallel (much faster!)
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    results = await agent.process_async(symbols, timeframe="1Day", limit=100)
    
    for symbol, market_data in results.items():
        print(f"{symbol}: {len(market_data.bars)} bars")
    
    await agent.cleanup_async_resources()

asyncio.run(main())
```

### Queue-Based Processing

```python
async def process_queue_example():
    config = get_config()
    agent = DataAgent(config)
    
    symbols = ["AAPL", "MSFT", "GOOGL"]
    queue = await agent.process_queue(symbols, timeframe="1Day", limit=100)
    
    # Process data as it becomes available
    while not queue.empty():
        symbol, market_data = await queue.get()
        if market_data:
            print(f"Processed {symbol}: {len(market_data.bars)} bars")
    
    await agent.cleanup_async_resources()
```

### Integration with Sync Code

You can still use the sync `process()` method:

```python
from agents.data_agent import DataAgent
from config.settings import get_config

config = get_config()
agent = DataAgent(config)

# Sync method (unchanged)
results = agent.process(["AAPL", "MSFT"], timeframe="1Day", limit=100)
```

## Performance Benefits

### Sequential (Sync)
- Fetching 5 symbols: ~5-10 seconds (1-2 seconds per symbol)

### Concurrent (Async)
- Fetching 5 symbols: ~1-2 seconds (all fetched in parallel)

**Speedup: ~3-5x faster** for multi-symbol fetches

## Testing

Added comprehensive async tests in `tests/test_data_agent.py`:

- `test_fetch_data_async()` - Single symbol async fetch
- `test_process_async_multiple_symbols()` - Concurrent multi-symbol fetching
- `test_process_queue()` - Queue-based processing
- `test_async_cache_usage()` - Cache functionality with async methods
- `test_cleanup_async_resources()` - Resource cleanup

Run async tests:
```bash
pytest tests/test_data_agent.py::TestDataAgentAsync -v
```

## Implementation Details

### Wrapping Sync Methods

Since `yfinance` and Alpaca's `StockHistoricalDataClient` are synchronous libraries, we wrap their calls in `asyncio.run_in_executor()`:

```python
loop = await self._get_event_loop()
market_data = await loop.run_in_executor(
    None,  # Use default executor
    self._fetch_yahoo_data,  # Sync method
    validated_symbol,  # Arguments
    timeframe,
    start_date,
    end_date,
    limit
)
```

This allows multiple fetches to run concurrently even though the underlying libraries are sync.

### Cache Sharing

The async methods share the same cache as sync methods, so:
- Fetching a symbol synchronously will cache it for async methods
- Fetching a symbol asynchronously will cache it for sync methods
- Cache TTL and size limits apply to both

### Error Handling

Async methods have the same error handling as sync methods:
- Symbol validation
- Provider-specific error handling
- Retry logic (via existing retry decorators on underlying methods)
- Graceful failures (skip failed symbols, continue with others)

## Future Enhancements

1. **Native Async Providers**: If async versions of data providers become available (e.g., `aioyfinance`), we can use them directly for better performance.

2. **Rate Limiting**: Add async-aware rate limiting using `asyncio.Semaphore` to respect provider rate limits.

3. **Streaming**: Implement real-time data streaming using async generators.

4. **Connection Pooling**: Optimize HTTP session management with connection pooling.

## Notes

- The async implementation is **optional** - existing code continues to work
- Async methods are best for fetching multiple symbols simultaneously
- For single symbol fetches, the performance difference is minimal
- Always call `cleanup_async_resources()` when done with async operations (or let `__del__` handle it)

