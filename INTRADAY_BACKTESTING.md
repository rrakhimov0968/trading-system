# Intraday Backtesting with Hourly Bars

## Overview

The backtesting system now supports **hourly and intraday timeframes** for more accurate backtesting. Instead of only using end-of-day prices, you can now use hourly (or even 15-minute, 5-minute, or 1-minute) bars to:

1. **Execute trades during market hours** - Not just at market close
2. **Better exit timing** - Sell at optimal prices during the day
3. **More realistic simulations** - Match actual trading behavior
4. **Intraday stop-loss/take-profit** - Risk management triggers during trading hours

## Timeframe Options

### Supported Timeframes

- **`1Day`** (default): Daily bars, end-of-day prices
- **`1Hour`**: Hourly bars - **Recommended for better exit timing**
- **`15Min`**: 15-minute bars - High frequency analysis
- **`5Min`**: 5-minute bars - Very granular
- **`1Min`**: 1-minute bars - Maximum granularity

### Data Availability

- **Daily bars**: ~252 bars per year (trading days only)
- **Hourly bars**: ~1,638 bars per year (6.5 hours/day × 252 trading days)
- **15-minute bars**: ~6,552 bars per year
- **5-minute bars**: ~19,656 bars per year

## Usage

### Command Line Script

```bash
# Use hourly bars for better exit timing
python scripts/backtest_all_strategies_sp500.py --timeframe 1Hour

# Use daily bars (default)
python scripts/backtest_all_strategies_sp500.py --timeframe 1Day

# Use 15-minute bars for high-frequency analysis
python scripts/backtest_all_strategies_sp500.py --timeframe 15Min

# Example: Backtest with hourly bars on 10 symbols
python scripts/backtest_all_strategies_sp500.py --timeframe 1Hour --max-symbols 10
```

### In Code

```python
from tests.backtest.backtest_engine import BacktestEngine
from tests.backtest.strategy_backtester import StrategyBacktester

engine = BacktestEngine(...)
backtester = StrategyBacktester("TrendFollowing", engine)

# Backtest with hourly bars
results = backtester.backtest(
    symbols=["AAPL", "GOOGL"],
    start_date="2023-01-01",
    end_date="2024-01-01",
    timeframe="1Hour"  # Use hourly bars
)
```

## Benefits of Hourly Backtesting

### 1. **Better Exit Timing**

**Daily bars (old way):**
- Strategy generates SELL signal
- Order executes at end-of-day price (market close)
- Might miss better prices during the day

**Hourly bars (new way):**
- Strategy generates SELL signal
- Order executes at that hour's price
- Can capture better intraday prices

### 2. **Intraday Risk Management**

- **Stop-loss triggers during market hours** - Not just at close
- **Take-profit executes when hit** - Even if it's 10 AM
- **Trailing stops update hourly** - More responsive

### 3. **More Accurate Simulation**

- Trades execute at realistic times
- Price movements during the day are captured
- Strategy performance reflects actual trading conditions

## How It Works

### Data Fetching

1. **DataAgent** fetches hourly bars from your data provider (Alpaca/Yahoo)
2. **BacktestEngine** processes each hourly bar sequentially
3. **Strategies** analyze hourly price patterns
4. **Signals** can be generated at any hour, not just end-of-day

### Signal Generation

```python
# Strategy processes hourly bars
for i in range(min_bars, len(bars)):
    current_bar = bars[i]  # Hourly bar
    current_market_data = MarketData(symbol=symbol, bars=bars[:i+1])
    
    # Generate signal based on hourly data
    signal = strategy.generate_signal(current_market_data)
    
    # Signal executes at that hour's price
    if signal == SignalAction.SELL:
        # Exit at current hourly price, not end-of-day
        ...
```

### Example: Hourly Exit Timing

**Scenario:** Strategy generates SELL signal at 2 PM

- **Daily bars**: Exit at 4 PM close price ($100.00)
- **Hourly bars**: Exit at 2 PM hourly price ($101.50) ✅ **Better price!**

## Performance Considerations

### API Limits

- **Hourly data**: ~24x more API calls than daily
- **15-minute data**: ~96x more API calls
- Make sure your data provider supports the volume needed

### Processing Time

- **Hourly backtests**: ~24x more bars to process
- **Strategies run 24x more iterations**
- More accurate but slower

### Storage

- Hourly CSV results are larger
- More data points per symbol
- Consider disk space for large backtests

## Recommendations

### For Most Users: Use `1Hour`

```bash
python scripts/backtest_all_strategies_sp500.py --timeframe 1Hour
```

**Why:**
- ✅ Good balance of accuracy and performance
- ✅ Captures intraday price movements
- ✅ Realistic exit timing
- ✅ Not too slow or data-heavy

### For Quick Tests: Use `1Day`

```bash
python scripts/backtest_all_strategies_sp500.py --timeframe 1Day
```

**Why:**
- ✅ Fast execution
- ✅ Less API usage
- ✅ Good for initial strategy validation

### For High-Frequency: Use `15Min` or `5Min`

**Why:**
- ✅ Maximum granularity
- ✅ Best exit timing
- ⚠️ Much slower and more data-intensive

## Comparison: Daily vs Hourly

### Daily Bars (1Day)
- **Bars per year**: ~252
- **Exit timing**: End of day only
- **Processing speed**: Fast
- **API calls**: Low
- **Best for**: Quick validation, long-term strategies

### Hourly Bars (1Hour)
- **Bars per year**: ~1,638
- **Exit timing**: Any hour during market hours
- **Processing speed**: Slower (but still reasonable)
- **API calls**: Higher
- **Best for**: Realistic backtesting, better exit prices

## Tips

1. **Start with hourly** for better accuracy
2. **Use daily for quick tests** to validate strategy logic
3. **Monitor API limits** when running large hourly backtests
4. **Check data availability** - some symbols might not have hourly data
5. **Results are more realistic** with hourly bars

## Example Output

```
📈 DATA LOADED (1Hour timeframe):
  Symbols: ['AAPL', 'GOOGL']
  Period: 2023-01-01 09:30:00 to 2024-01-01 16:00:00
  Calendar Days: ~365
  Hours: 1,638
```

The system now shows the timeframe being used and provides detailed bar counts.

