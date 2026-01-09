# Trading System Commands Reference

## 🧪 Testing Commands (NEW - Critical Fixes)

### Quick Test All Critical Fixes
```bash
python3 tests/test_critical_fixes.py
```

### Run All Tests via Script
```bash
./run_tests.sh              # Run all basic tests
./run_tests.sh critical     # Test critical fixes only
./run_tests.sh hybrid       # Test hybrid scaling
./run_tests.sh config       # Test config validation
```

### Individual Critical Fix Tests
```bash
# Problem 1: Deterministic Pipeline
python3 -c "from tests.test_critical_fixes import TestProblem1_DeterministicPipeline; TestProblem1_DeterministicPipeline().test_pipeline_enforces_order()"

# Problem 2: Atomic Locking
python3 -c "from tests.test_critical_fixes import TestProblem2_AtomicLocking; TestProblem2_AtomicLocking().test_reserve_release_cycle()"

# Problem 3: Live Account Value
python3 -c "from tests.test_critical_fixes import TestProblem3_LiveAccountValue; TestProblem3_LiveAccountValue().test_sizer_uses_live_equity()"

# Problem 4: Tier Validation (requires invalid config)
python3 -c "
from config.settings import AppConfig
import os
os.environ['ENABLE_TIERED_ALLOCATION'] = 'true'
os.environ['TIER1_ALLOCATION'] = '0.50'
os.environ['TIER2_ALLOCATION'] = '0.30'
os.environ['TIER3_ALLOCATION'] = '0.10'  # Invalid
try:
    AppConfig.from_env()
    print('❌ Should have failed')
except ValueError as e:
    print('✅ Validation works:', str(e)[:50])
"

# Problem 5: Scanner Diversity
python3 tests/test_critical_fixes.py  # Includes scanner test

# Problem 6: Fractional Check
python3 -c "from tests.test_critical_fixes import TestProblem6_FractionalCheck; TestProblem6_FractionalCheck().test_fractional_validation_exists()"

# Problem 7: Structured Logging
python3 -c "from tests.test_critical_fixes import TestProblem7_StructuredLogging; TestProblem7_StructuredLogging().test_validation_decision_has_metadata()"
```

### Full Test Suite (if pytest installed)
```bash
pytest tests/ -v                    # All tests
pytest tests/test_critical_fixes.py -v  # Critical fixes only
pytest tests/test_hybrid_scaling.py -v  # Hybrid scaling
```

---

# Trading System Commands Reference

Complete reference guide for all commands to run the trading system, backtests, and database queries.

---

## Table of Contents

1. [Running the Trading System](#running-the-trading-system)
2. [Backtesting](#backtesting)
3. [Database Queries](#database-queries)
4. [Alpaca Synchronization](#alpaca-synchronization)
5. [Monitoring & Utilities](#monitoring--utilities)
6. [Emergency Controls](#emergency-controls)

---

## Running the Trading System

### Main Trading System (Continuous Loop)

Start the main trading system in continuous mode (default: async orchestrator):

```bash
python main.py
```

**Environment Variables:**

```bash
# Use async orchestrator (default: true)
USE_ASYNC_ORCHESTRATOR=true python main.py

# Use synchronous orchestrator
USE_ASYNC_ORCHESTRATOR=false python main.py

# Set log level
LOG_LEVEL=DEBUG python main.py
LOG_LEVEL=INFO python main.py
LOG_LEVEL=WARNING python main.py
```

**What it does:**
- Fetches market data for configured symbols
- Generates trading signals using registered strategies
- Validates signals through quant and risk agents
- Executes approved trades via Alpaca
- Manages positions (stop-loss, take-profit, trailing stops)
- Runs continuously, checking market hours and executing trades

**Configuration:**
- Symbols: Set in `.env` file (`SYMBOLS=SPY,QQQ,AAPL,GOOGL,NVDA`)
- Trading mode: `TRADING_MODE=PAPER` or `TRADING_MODE=LIVE`
- Loop interval: `LOOP_INTERVAL_SECONDS=60` (default: 60 seconds)

**To stop:**
- Press `Ctrl+C` for graceful shutdown
- Or create `EMERGENCY_STOP` file (see [Emergency Controls](#emergency-controls))

---

## Backtesting

### Single Strategy - Single Symbol

Backtest a single strategy on a single symbol:

```bash
# Basic usage
python -m pytest tests/backtest_strategies.py::test_backtest_single_strategy -v

# Or run directly with StrategyBacktester
python -c "
from tests.backtest.strategy_backtester import StrategyBacktester
from config.settings import get_config

config = get_config()
backtester = StrategyBacktester(config=config)
result = backtester.backtest(strategy_name='MeanReversion', symbol='AAPL', timeframe='1Day')
print(result)
"
```

### Single Strategy - All S&P 500 Symbols

Backtest one strategy across all S&P 500 stocks:

```bash
# Basic usage (daily bars)
python scripts/backtest_sp500.py --strategy MeanReversion --output results_mean_reversion.csv

# With custom date range
python scripts/backtest_sp500.py \
    --strategy TrendFollowing \
    --start-date 2022-01-01 \
    --end-date 2024-12-31 \
    --output results_trend.csv

# Use hourly bars (more granular)
python scripts/backtest_sp500.py \
    --strategy MeanReversion \
    --timeframe 1Hour \
    --output results_hourly.csv

# Limit number of symbols (for testing)
python scripts/backtest_sp500.py \
    --strategy MeanReversion \
    --max-symbols 10 \
    --output results_test.csv
```

**Available Strategies:**
- `MeanReversion`
- `TrendFollowing`
- `Momentum`
- `Breakout`
- `RSIMeanReversion`
- `MACDCrossover`
- `BollingerBands`

**Timeframe Options:**
- `1Day` (default) - Daily bars
- `1Hour` - Hourly bars
- `15Min` - 15-minute bars
- `5Min` - 5-minute bars
- `1Min` - 1-minute bars

### All Strategies - All S&P 500 Symbols (Overnight Run)

Backtest ALL strategies on ALL S&P 500 symbols (designed for overnight runs):

```bash
# Run all strategies on all S&P 500 stocks (daily bars)
python scripts/backtest_all_strategies_sp500.py

# Use hourly bars
python scripts/backtest_all_strategies_sp500.py --timeframe 1Hour

# Custom date range
python scripts/backtest_all_strategies_sp500.py \
    --start-date 2021-01-01 \
    --end-date 2024-12-31

# Limit symbols for testing
python scripts/backtest_all_strategies_sp500.py --max-symbols 50

# Resume from checkpoint (if interrupted)
python scripts/backtest_all_strategies_sp500.py --resume
```

**Output:**
- Creates separate CSV files per strategy in `backtest_results/` directory
- Format: `backtest_results/{strategy_name}_sp500_results_{timestamp}.csv`
- Includes progress tracking in `backtest_progress.json`

**What it does:**
1. Iterates through all registered strategies
2. For each strategy, backtests all S&P 500 symbols one by one
3. Saves results incrementally (after each stock completes)
4. Can resume from checkpoint if interrupted

---

## Database Queries

### Query Today's Trades

```bash
# Show all today's trades
python query_todays_trades.py

# Show only BUY orders
python query_todays_trades.py --buys

# Show only SELL orders
python query_todays_trades.py --sells

# Filter by symbol
python query_todays_trades.py --symbol AAPL

# Include failed trades
python query_todays_trades.py --include-failed

# Combined filters
python query_todays_trades.py --buys --symbol AAPL
```

**Output includes:**
- Timestamp
- Action (BUY/SELL)
- Symbol
- Strategy name
- Quantity
- Price / Fill price
- Amount
- Execution status

### Query Recent Trades

```bash
# Last 7 days (default)
python query_recent_trades.py

# Last 30 days
python query_recent_trades.py --days 30

# Filter by symbol
python query_recent_trades.py --symbol AAPL --days 7

# Limit results
python query_recent_trades.py --days 7 --limit 50
```

### Direct SQL Queries

Connect to the SQLite database directly:

```bash
# Open SQLite shell
sqlite3 trading_system.db

# Useful queries inside SQLite:
```

```sql
-- View today's trades
SELECT * FROM trade_history 
WHERE date(timestamp) = date('now')
ORDER BY timestamp DESC;

-- View all executed BUY orders today
SELECT symbol, qty, fill_price, strategy_name, timestamp 
FROM trade_history 
WHERE action = 'BUY' 
  AND executed = 1 
  AND date(timestamp) = date('now')
ORDER BY timestamp DESC;

-- View open positions (buy orders without matching sells)
SELECT symbol, SUM(qty) as total_qty, AVG(fill_price) as avg_price
FROM trade_history
WHERE action = 'BUY' AND executed = 1
GROUP BY symbol
HAVING total_qty > 0;

-- View trade summary by strategy
SELECT strategy_name, 
       COUNT(*) as total_trades,
       SUM(CASE WHEN executed = 1 THEN 1 ELSE 0 END) as executed_trades,
       SUM(CASE WHEN action = 'BUY' AND executed = 1 THEN qty * fill_price ELSE 0 END) as total_buy_amount
FROM trade_history
WHERE date(timestamp) = date('now')
GROUP BY strategy_name;

-- View equity curve (daily performance)
SELECT * FROM equity_curve 
ORDER BY date DESC 
LIMIT 30;

-- View risk metrics
SELECT * FROM risk_metrics 
ORDER BY timestamp DESC 
LIMIT 10;
```

**Exit SQLite:**
```sql
.quit
```

---

## Alpaca Synchronization

### Sync Alpaca Orders to Database

Sync executed orders from Alpaca broker to local database:

```bash
# Sync today's orders
python sync_alpaca_orders.py

# Sync orders for specific date
python sync_alpaca_orders.py --date 2024-01-15

# Sync last N days
python sync_alpaca_orders.py --days 7

# Verbose output
python sync_alpaca_orders.py --verbose
```

**What it does:**
- Fetches orders from Alpaca API
- Matches with database records
- Creates missing trade history entries
- Updates execution status

### Sync Alpaca Positions to Database

Sync current open positions from Alpaca to database (useful for reconciliation):

```bash
# Preview what would be synced (dry run)
python sync_alpaca_positions.py --dry-run

# Actually sync positions
python sync_alpaca_positions.py
```

**What it does:**
- Fetches all open positions from Alpaca
- Creates BUY trade records in database for untracked positions
- Updates quantities if positions changed
- Helps reconcile system state with broker reality

**Use cases:**
- System starts with existing positions
- Manual trades made outside the system
- Reconciliation after system restart

---

## Monitoring & Utilities

### Streamlit Dashboard

Launch the monitoring dashboard:

```bash
streamlit run monitoring.py
```

**Features:**
- Real-time position monitoring
- Trade history visualization
- Performance metrics
- Account status
- Risk metrics

**Access:**
- Opens automatically in browser (usually `http://localhost:8501`)
- Auto-refreshes every 60 seconds

### Test Safety Checks

Run tests for safety mechanisms:

```bash
# Run all safety check tests
python test_safety_checks.py

# Run with pytest (more verbose)
pytest test_safety_checks.py -v

# Run specific test class
pytest test_safety_checks.py::TestOrderTracker -v
```

**What it tests:**
- Order cooldown functionality
- Daily order limits
- Account health validation
- Position reconciliation
- Emergency stop mechanism

### Check Database Schema

```bash
# View database schema
sqlite3 trading_system.db ".schema"

# Or view specific table structure
sqlite3 trading_system.db ".schema trade_history"
```

---

## Market Regime Filtering

### Enable Regime Filter (Recommended)

The market regime agent evaluates SPY vs SMA200 to automatically scale position sizes based on market conditions. This provides system-level protection against bear markets without "choking" strategies.

```bash
# Enable soft mode (default - scales positions, never blocks)
export ENABLE_REGIME_FILTER="true"
# STRICT_REGIME defaults to false (soft scalar mode)

# Enable strict mode (hard gate - blocks all trading in bear markets)
export ENABLE_REGIME_FILTER="true"
export STRICT_REGIME="true"

# Run with regime filtering
python main.py
```

**Soft Mode (Default - Recommended):**
- Bull market (SPY > SMA200): Full position sizing (risk_scalar = 1.0)
- Bear market (SPY < SMA200): Scaled down positions (risk_scalar = 0.0-0.5)
- Strategies continue to work, positions automatically reduced
- Expert-friendly: doesn't "choke" the system

**Strict Mode:**
- Bull market: Full position sizing
- Bear market: **No trading at all** (complete protection)
- Use for capital preservation during stress events

**Expected Impact (Soft Mode):**
- Max drawdown: ↓ 20-35%
- Sharpe ratio: ↑ 0.4-0.8
- Same winners survive (strategies not choked)
- Fewer trades in choppy markets

See `REGIME_FILTER_IMPLEMENTATION.md` for detailed documentation.

---

## Emergency Controls

### Emergency Stop

Stop trading immediately by creating an emergency stop file:

```bash
# Stop trading
touch EMERGENCY_STOP

# Resume trading (remove file)
rm EMERGENCY_STOP
```

**What it does:**
- System checks for `EMERGENCY_STOP` file before each iteration
- If file exists, trading stops immediately
- No new orders are placed
- Existing positions remain (system doesn't close positions automatically)

**Check if stop is active:**
```bash
# Check if file exists
ls -la EMERGENCY_STOP

# Or in system logs, look for:
# "🚨 EMERGENCY STOP ACTIVE - NOT PLACING ANY ORDERS!"
```

### View Logs

```bash
# View latest log file
tail -f logs/trading_system_*.log

# Search for errors
grep -i error logs/trading_system_*.log

# Search for specific symbol
grep "AAPL" logs/trading_system_*.log

# View last 100 lines
tail -n 100 logs/trading_system_*.log
```

---

## Common Workflows

### Daily Trading Workflow

```bash
# 1. Check today's trades
python query_todays_trades.py

# 2. Sync with Alpaca (if needed)
python sync_alpaca_positions.py --dry-run
python sync_alpaca_positions.py

# 3. Start trading system
python main.py

# 4. Monitor dashboard (separate terminal)
streamlit run monitoring.py
```

### Weekly Backtest Workflow

```bash
# 1. Run backtest for specific strategy
python scripts/backtest_sp500.py --strategy MeanReversion --output weekly_results.csv

# 2. Analyze results
python -c "
import pandas as pd
df = pd.read_csv('weekly_results.csv')
print(df.describe())
print(f\"Win rate: {df['win_rate'].mean():.2%}\")
print(f\"Avg return: {df['total_return'].mean():.2%}\")
"
```

### Overnight Full Backtest

```bash
# 1. Start overnight backtest
nohup python scripts/backtest_all_strategies_sp500.py --timeframe 1Hour > backtest.log 2>&1 &

# 2. Check progress
tail -f backtest.log

# 3. Check results in morning
ls -lh backtest_results/
```

### System Maintenance Workflow

```bash
# 1. Emergency stop
touch EMERGENCY_STOP

# 2. Check current positions
python query_todays_trades.py --buys

# 3. Reconcile with Alpaca
python sync_alpaca_positions.py --dry-run

# 4. Update database if needed
python sync_alpaca_positions.py

# 5. Review logs for issues
grep -i "error\|warning" logs/trading_system_*.log | tail -20

# 6. Test safety checks
pytest test_safety_checks.py -v

# 7. Resume trading (if ready)
rm EMERGENCY_STOP
python main.py
```

---

## Environment Variables Reference

Set these in your `.env` file or export before running commands:

```bash
# Trading Configuration
TRADING_MODE=PAPER              # PAPER or LIVE
USE_ASYNC_ORCHESTRATOR=true     # true or false
LOOP_INTERVAL_SECONDS=60        # Seconds between iterations

# Symbols
SYMBOLS=SPY,QQQ,AAPL,GOOGL,NVDA

# Alpaca API (required for live/paper trading)
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_PAPER=true               # Use paper trading account

# Risk Management
RISK_MAX_PER_TRADE=0.02         # 2% max risk per trade
RISK_MAX_DAILY_LOSS=0.05        # 5% max daily loss
RISK_MAX_QTY=1000               # Max shares per order

# Quant Agent Thresholds
QUANT_MIN_SHARPE=0.8            # Minimum Sharpe ratio
QUANT_MAX_DRAWDOWN=0.15         # Maximum drawdown (15%)

# Market Regime Filtering (NEW)
ENABLE_REGIME_FILTER=false      # Enable market regime filtering
STRICT_REGIME=false             # true = hard gate (block in bear), false = soft scalar (default)
REGIME_BENCHMARK=SPY            # Benchmark symbol for regime check
REGIME_SMA_PERIOD=200           # SMA period for trend check

# Logging
LOG_LEVEL=INFO                  # DEBUG, INFO, WARNING, ERROR
```

---

## Troubleshooting

### System Not Starting

```bash
# Check configuration
python -c "from config.settings import get_config; print(get_config())"

# Check Alpaca connection
python -c "
from agents.execution_agent import ExecutionAgent
from config.settings import get_config
agent = ExecutionAgent(get_config())
print(agent.get_account())
"
```

### Database Issues

```bash
# Check database exists
ls -lh trading_system.db

# Backup database
cp trading_system.db trading_system.db.backup

# Check database integrity
sqlite3 trading_system.db "PRAGMA integrity_check;"
```

### No Trades Executing

```bash
# Check if market is open
python -c "
from utils.market_hours import is_market_open
print('Market open:', is_market_open())
"

# Check account health
python -c "
from agents.execution_agent import ExecutionAgent
from config.settings import get_config
agent = ExecutionAgent(get_config())
account = agent.get_account()
print(f'Equity: {account.equity}')
print(f'Buying power: {account.buying_power}')
"

# Check emergency stop
ls -la EMERGENCY_STOP
```

---

## Quick Reference Card

```bash
# START TRADING
python main.py

# CHECK TODAY'S TRADES
python query_todays_trades.py

# RUN BACKTEST
python scripts/backtest_sp500.py --strategy MeanReversion

# SYNC WITH ALPACA
python sync_alpaca_positions.py

# VIEW DASHBOARD
streamlit run monitoring.py

# EMERGENCY STOP
touch EMERGENCY_STOP

# VIEW LOGS
tail -f logs/trading_system_*.log
```

---

## Additional Resources

- `CRITICAL_FIXES.md` - Safety checks and risk management
- `INTRADAY_BACKTESTING.md` - Hourly backtesting guide
- `README.md` - System overview and setup
- `ARCHITECTURE_*.md` - System architecture documentation

---

**Last Updated:** 2025-01-06

**Recent Updates:**
- ✅ Market Regime Agent: System-level risk adjustment with soft scalar mode (default)
- ✅ Production-grade agent architecture for regime filtering
