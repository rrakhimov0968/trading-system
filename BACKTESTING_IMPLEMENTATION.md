# Backtesting Framework Implementation

## Overview

A comprehensive backtesting framework integrated with your existing trading system architecture. This framework allows you to validate strategies before deploying them to paper/live trading.

## ✅ Implemented Features

### 1. **Core Backtest Engine** (`tests/backtest/backtest_engine.py`)
- VectorBT integration for fast vectorized backtesting
- Full OHLC data support (not just Close prices)
- Data gap handling (forward-fill, drop NaNs)
- Configurable risk-free rate for accurate Sharpe calculations
- Per-symbol and portfolio-level analysis
- Integration with existing strategy classes
- Database persistence of results

### 2. **Strategy Backtester** (`tests/backtest/strategy_backtester.py`)
- Wrapper for all 10 existing strategies
- Uses `STRATEGY_REGISTRY` - no code duplication
- Supports per-symbol and portfolio-level backtesting
- Strategy configuration support

### 3. **Walk-Forward Analysis** (`tests/backtest/walk_forward.py`)
- Tests strategy across multiple time periods
- Detects overfitting by checking consistency
- Configurable number of splits (default: 4 periods)
- Validates robustness across different market regimes

### 4. **Database Integration**
- New `BacktestResults` table in `utils/database.py`
- Automatic persistence of all backtest results
- Tracks strategy parameters, metrics, and validation status
- Supports walk-forward period tracking

### 5. **CLI Tool** (`tests/backtest_strategies.py`)
- Command-line interface for running backtests
- Support for single strategy or all strategies
- Walk-forward analysis option
- Configurable parameters (risk-free rate, commission, dates)
- Comprehensive result reporting

### 6. **Configuration Integration**
- Uses thresholds from `config/settings.py`:
  - `QUANT_MIN_SHARPE` (default: 1.5)
  - `QUANT_MAX_DRAWDOWN` (default: 0.08 = 8%)
- Configurable risk-free rate (default: 4% annual)
- Configurable commission (default: 0.1%)

## 📋 Usage

### Basic Usage

```bash
# Test a single strategy
python tests/backtest_strategies.py --strategy TrendFollowing --symbols AAPL,SPY

# Test all strategies
python tests/backtest_strategies.py --all --symbols AAPL,GOOGL,MSFT,SPY,QQQ

# Test with walk-forward analysis
python tests/backtest_strategies.py --strategy MeanReversion --walk-forward

# Custom parameters
python tests/backtest_strategies.py \
    --strategy TrendFollowing \
    --symbols AAPL,GOOGL \
    --start-date 2020-01-01 \
    --end-date 2024-12-31 \
    --rf-rate 0.05 \
    --commission 0.002 \
    --initial-cash 20000
```

### Command-Line Options

```
--strategy          Strategy name (required unless --all)
--all               Test all strategies
--symbols           Comma-separated symbols (default: AAPL,GOOGL,MSFT,AMZN,SPY,QQQ)
--start-date        Start date YYYY-MM-DD (default: 2021-01-01)
--end-date          End date YYYY-MM-DD (default: today)
--initial-cash      Initial capital (default: 10000)
--commission        Commission rate (default: 0.001 = 0.1%)
--rf-rate           Annual risk-free rate (default: 0.04 = 4%)
--walk-forward      Run walk-forward analysis
--monte-carlo       Run Monte Carlo simulation (not yet implemented)
```

## 📊 Results Interpretation

### Validation Thresholds

A strategy **PASSES** if:
- ✅ Sharpe Ratio ≥ 1.5 (or `QUANT_MIN_SHARPE`)
- ✅ Max Drawdown ≤ 8% (or `QUANT_MAX_DRAWDOWN`)
- ✅ Win Rate ≥ 45%

### Example Output

```
================================================================================
📊 RESULTS SUMMARY: TrendFollowing
================================================================================

Symbol     Return       Sharpe     Max DD       Win Rate     Trades     Status
--------------------------------------------------------------------------------
AAPL       25.30%       1.45       12.50%       52.00%       45         ✅ PASS
GOOGL      -5.20%       0.72       18.30%       42.00%       38         ❌ FAIL
SPY        15.80%       1.20       10.20%       48.00%       42         ✅ PASS

================================================================================
📊 VERDICT: 2/3 symbols passed validation
✅ Strategy shows promise on multiple symbols
================================================================================
```

### Walk-Forward Analysis Output

```
================================================================================
🔍 WALK-FORWARD ANALYSIS
================================================================================

Walk-Forward Summary:
  Periods Tested: 4
  Periods Passed: 3
  Avg Sharpe: 1.35
  Sharpe Range: 0.85
  Consistency: ✅ Consistent
```

## 🏗️ Architecture

### Integration with Existing Code

The backtesting framework **reuses** your existing code:

1. **Strategy Classes**: Uses `STRATEGY_REGISTRY` - all 10 strategies are tested without modification
2. **Configuration**: Reads from `config/settings.py` for thresholds
3. **Database**: Uses existing `DatabaseManager` for persistence
4. **Data Models**: Uses `MarketData`, `Bar` models for signal generation

### File Structure

```
tests/
├── backtest/
│   ├── __init__.py              # Package exports
│   ├── backtest_engine.py       # Core engine with VectorBT
│   ├── strategy_backtester.py   # Strategy wrapper
│   └── walk_forward.py          # Walk-forward analyzer
├── backtest_strategies.py       # CLI entry point
└── results/                     # Generated charts (created automatically)
```

## 📈 Key Features

### 1. **Full OHLC Support**
- Fetches Open, High, Low, Close, Volume data
- Strategies that need OHLC (e.g., ATR-based) work correctly
- Handles data gaps gracefully

### 2. **Risk-Free Rate Adjustment**
- Configurable annual risk-free rate (default: 4%)
- Accurately calculates Sharpe ratio in high-rate environments
- Passed to VectorBT's Sharpe calculation

### 3. **Data Quality**
- Forward-fills gaps
- Drops symbols with insufficient data
- Validates data before backtesting

### 4. **Per-Symbol vs Portfolio Analysis**
- Can test individual symbols separately
- Can test portfolio-level performance
- Both modes supported

### 5. **Overfitting Detection**
- Walk-forward analysis tests consistency
- Warns if performance varies significantly across periods
- Helps identify overfit strategies

## 🚀 Next Steps

### Recommended Workflow

1. **Run Backtests**: Test all strategies on historical data
   ```bash
   python tests/backtest_strategies.py --all
   ```

2. **Identify Winners**: Note which strategies pass on multiple symbols

3. **Walk-Forward Validation**: Test promising strategies with walk-forward
   ```bash
   python tests/backtest_strategies.py --strategy TrendFollowing --walk-forward
   ```

4. **Paper Trade**: Deploy passing strategies to paper trading

5. **Compare Results**: Compare paper trading results to backtest predictions

### Future Enhancements (Not Yet Implemented)

- **Monte Carlo Simulation**: Test robustness by shuffling trade order
- **Position Sizing**: Integrate `RiskAgent` position sizing logic
- **Slippage Modeling**: More realistic transaction costs
- **Market Impact**: Model impact of larger positions
- **Multi-Timeframe Testing**: Test strategies across different timeframes

## ⚠️ Important Notes

### Backtest vs Live Trading

Backtests are **optimistic**:
- No real slippage (you set 0.1%, reality might be 0.2-0.5%)
- Perfect data (no gaps, no failed orders)
- No emotional decisions

**Rule of thumb**: If backtest Sharpe is 1.5, expect live Sharpe of 1.0-1.2.

### Overfitting Risk

If you test 50 parameter combinations and pick the best:
- That's overfitting
- It won't work live

**Solution**: Use walk-forward analysis to detect overfitting.

### Strategy Selection

**DO NOT TRADE** strategies that:
- ❌ Fail on all symbols
- ❌ Pass only on 1-2 symbols (likely overfit)
- ❌ Show inconsistent walk-forward results

**DO TRADE** strategies that:
- ✅ Pass on 5+ symbols
- ✅ Show consistent walk-forward results
- ✅ Have Sharpe > 1.5, Max DD < 8%

## 📚 Dependencies

- `vectorbt==0.25.2`: Vectorized backtesting (already in `requirements.txt`)
- `pandas`, `numpy`: Data manipulation (already installed)
- `yfinance`: Historical data (already installed)

## 🔧 Troubleshooting

### "No data returned from yfinance"
- Check internet connection
- Verify symbol names are correct
- Try a different date range

### "Insufficient data" warnings
- Increase `--start-date` to get more historical data
- Remove symbols that don't have enough history

### "Strategy failed on ALL symbols"
- This is normal for some strategies
- Delete failing strategies or adjust parameters
- Focus on strategies that pass

## 📝 Example: Full Testing Workflow

```bash
# 1. Test all strategies
python tests/backtest_strategies.py --all --symbols AAPL,GOOGL,MSFT,SPY,QQQ

# 2. If TrendFollowing passes, validate with walk-forward
python tests/backtest_strategies.py --strategy TrendFollowing --walk-forward

# 3. If walk-forward is consistent, paper trade it
# (This happens in your main.py with the live system)

# 4. Compare paper trading results to backtest
# (Use monitoring.py dashboard to track performance)
```

## 🎯 Success Criteria

A strategy is ready for paper trading if:
1. ✅ Passes on 5+ symbols
2. ✅ Walk-forward analysis shows consistency
3. ✅ Sharpe ratio > 1.5 across multiple periods
4. ✅ Max drawdown < 8% across periods
5. ✅ Win rate > 45%

---

**Note**: This implementation integrates seamlessly with your existing system. All strategies in `STRATEGY_REGISTRY` are automatically testable without any code changes.

