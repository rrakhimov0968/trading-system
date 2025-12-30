# Database Persistence Implementation

## Overview

Implemented comprehensive database persistence using SQLAlchemy to track trade history, equity curves, risk metrics, iteration logs, and audit reports. This enables historical analysis, performance tracking, and audit trails.

## Database Schema

### 1. TradeHistory Table

Stores executed trades with full signal and execution details:

- `id` (String, PK) - Correlation ID
- `symbol` (String) - Stock symbol
- `action` (Enum) - BUY, SELL, HOLD
- `strategy_name` (String) - Strategy used
- `qty` (Integer) - Number of shares
- `price` (Float) - Signal price
- `fill_price` (Float) - Actual execution price
- `stop_loss`, `take_profit` (Float) - Risk management levels
- `risk_amount` (Float) - Risk per trade
- `confidence` (Float) - Signal confidence
- `timestamp` (DateTime) - Signal timestamp
- `execution_time` (DateTime) - Execution timestamp
- `order_id` (String) - Broker order ID
- `executed` (Boolean) - Execution status
- `error` (Text) - Error message if failed
- `sharpe`, `drawdown`, `expectancy` (Float) - Quantitative metrics
- `reasoning` (Text) - LLM reasoning
- `correlation_id` (String) - For tracking
- `created_at` (DateTime) - Record creation time

### 2. EquityCurve Table

Tracks portfolio equity over time:

- `id` (String, PK) - Unique ID
- `timestamp` (DateTime) - Snapshot time
- `equity` (Float) - Current equity
- `cash` (Float) - Cash balance
- `buying_power` (Float) - Available buying power
- `total_return` (Float) - Total return percentage
- `daily_return` (Float) - Daily return percentage
- `created_at` (DateTime) - Record creation time

### 3. RiskMetrics Table

Stores risk metrics snapshots:

- `id` (String, PK) - Unique ID
- `timestamp` (DateTime) - Snapshot time
- `total_positions` (Integer) - Number of open positions
- `total_exposure` (Float) - Total portfolio exposure
- `max_position_size` (Float) - Largest position size
- `portfolio_value` (Float) - Total portfolio value
- `daily_pnl` (Float) - Daily profit/loss
- `total_pnl` (Float) - Total profit/loss
- `max_drawdown` (Float) - Maximum drawdown
- `sharpe_ratio` (Float) - Sharpe ratio
- `risk_per_trade` (Float) - Risk per trade limit
- `daily_loss_limit` (Float) - Daily loss limit
- `current_daily_loss` (Float) - Current daily loss
- `created_at` (DateTime) - Record creation time

### 4. IterationLog Table

Logs each trading iteration:

- `id` (String, PK) - Unique ID
- `iteration_number` (Integer) - Iteration number
- `timestamp` (DateTime) - Iteration time
- `symbols_processed` (Text, JSON) - List of symbols
- `signals_generated` (Integer) - Number of signals
- `signals_validated` (Integer) - Validated signals
- `signals_approved` (Integer) - Approved signals
- `signals_executed` (Integer) - Executed signals
- `duration_seconds` (Float) - Iteration duration
- `errors` (Text, JSON) - List of errors
- `created_at` (DateTime) - Record creation time

### 5. AuditReportLog Table

Stores audit reports:

- `id` (String, PK) - Unique ID
- `report_type` (String) - "iteration", "daily", "weekly"
- `timestamp` (DateTime) - Report time
- `summary` (Text) - Report summary
- `recommendations` (Text) - Recommendations
- `metrics` (Text, JSON) - Report metrics
- `created_at` (DateTime) - Record creation time

## DatabaseManager Class

### Initialization

```python
from utils.database import DatabaseManager
from config.settings import get_config

config = get_config()
db = DatabaseManager(config=config)
```

### Configuration

Database configuration is optional. If not configured, uses in-memory SQLite:

```bash
# .env
DATABASE_URL=sqlite:///trading_system.db
DATABASE_ECHO=false
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10
```

For PostgreSQL:
```bash
DATABASE_URL=postgresql://user:password@localhost/trading_system
```

For MySQL:
```bash
DATABASE_URL=mysql://user:password@localhost/trading_system
```

### Methods

#### Logging

- `log_trade(signal, execution_result, correlation_id)` - Log a trade
- `log_iteration(iteration_summary)` - Log iteration summary
- `log_audit_report(audit_report)` - Log audit report
- `log_equity_snapshot(equity, cash, buying_power)` - Log equity snapshot
- `log_risk_metrics(...)` - Log risk metrics

#### Querying

- `get_trade_history(symbol, start_date, end_date, limit)` - Query trade history
- `get_equity_curve(start_date, end_date, limit)` - Query equity curve

#### Utility

- `generate_correlation_id()` - Generate correlation ID
- `health_check()` - Check database health
- `session()` - Context manager for database sessions

## Integration

### AuditAgent Integration

The `AuditAgent` automatically persists data when database is configured:

```python
from agents.audit_agent import AuditAgent
from config.settings import get_config

config = get_config()
agent = AuditAgent(config=config)

# Process iteration (automatically persists to DB if configured)
report = agent.process(iteration_summary, execution_results)
```

**What gets persisted:**
- Iteration summaries
- Audit reports
- Trades (from execution results)

### Manual Persistence

You can also persist data manually:

```python
from utils.database import DatabaseManager
from models.signal import TradingSignal
from models.audit import ExecutionResult

db = DatabaseManager(config=config)

# Log a trade
corr_id = db.log_trade(signal, execution_result)

# Log iteration
db.log_iteration(iteration_summary)

# Log equity snapshot
db.log_equity_snapshot(equity=100000.0, cash=50000.0)

# Log risk metrics
db.log_risk_metrics(
    portfolio_value=100000.0,
    daily_pnl=500.0,
    sharpe_ratio=1.5
)
```

## Usage Examples

### Query Trade History

```python
from datetime import datetime, timedelta
from utils.database import DatabaseManager

db = DatabaseManager(config=config)

# Get all trades for a symbol
trades = db.get_trade_history(symbol="AAPL")

# Get trades in date range
start_date = datetime.now() - timedelta(days=30)
trades = db.get_trade_history(
    symbol="AAPL",
    start_date=start_date,
    limit=100
)
```

### Query Equity Curve

```python
# Get equity curve data
curve = db.get_equity_curve(
    start_date=datetime.now() - timedelta(days=90),
    limit=1000
)

# Calculate returns
for snapshot in curve:
    print(f"{snapshot.timestamp}: ${snapshot.equity:.2f} ({snapshot.total_return:.2f}%)")
```

### Direct SQLAlchemy Queries

```python
from utils.database import TradeHistory, RiskMetrics

with db.session() as session:
    # Query with filters
    trades = session.query(TradeHistory).filter(
        TradeHistory.symbol == "AAPL",
        TradeHistory.executed == True
    ).all()
    
    # Aggregations
    from sqlalchemy import func
    total_trades = session.query(func.count(TradeHistory.id)).scalar()
    avg_confidence = session.query(func.avg(TradeHistory.confidence)).scalar()
```

## Error Handling

Database operations are wrapped in try/except blocks and use context managers for proper session handling:

```python
try:
    db.log_trade(signal, execution_result)
except Exception as e:
    logger.error(f"Failed to persist trade: {e}")
    # System continues even if persistence fails
```

## Testing

Comprehensive tests in `tests/test_database.py`:

```bash
# Run database tests
pytest tests/test_database.py -v

# Run with coverage
pytest tests/test_database.py --cov=utils.database --cov-report=html
```

**Test Coverage:**
- Database initialization
- Trade logging and querying
- Iteration logging
- Audit report logging
- Equity curve tracking
- Risk metrics logging
- Integration with AuditAgent
- Error handling

## Migration

### Initial Setup

1. Set `DATABASE_URL` in `.env`:
   ```bash
   DATABASE_URL=sqlite:///trading_system.db
   ```

2. Tables are created automatically on first use:
   ```python
   db = DatabaseManager(config=config)
   # Tables created automatically
   ```

### Manual Migration

For production databases, you can use Alembic:

```bash
pip install alembic
alembic init alembic
# Configure alembic.ini with DATABASE_URL
alembic revision --autogenerate -m "Initial schema"
alembic upgrade head
```

## Performance Considerations

1. **Indexes**: Key fields (`symbol`, `timestamp`, `order_id`) are indexed for fast queries
2. **Connection Pooling**: Configured via `DATABASE_POOL_SIZE` and `DATABASE_MAX_OVERFLOW`
3. **Batch Operations**: For bulk inserts, consider batching:
   ```python
   with db.session() as session:
       for signal in signals:
           # ... create trade objects
           session.add(trade)
       session.commit()  # Single commit for all
   ```
4. **Archiving**: For long-running systems, consider archiving old data:
   ```python
   # Archive trades older than 1 year
   cutoff = datetime.now() - timedelta(days=365)
   old_trades = db.get_trade_history(end_date=cutoff)
   # Archive to separate table or file
   ```

## Security

1. **Connection Strings**: Store `DATABASE_URL` securely (environment variables, secrets manager)
2. **SQL Injection**: SQLAlchemy handles parameterization automatically
3. **Access Control**: Use database-level permissions for production
4. **Encryption**: Use encrypted connections (SSL/TLS) for remote databases

## Monitoring

Monitor database health:

```python
health = db.health_check()
if health["status"] != "healthy":
    logger.error(f"Database unhealthy: {health.get('error')}")
```

## Future Enhancements

1. **Time Series Optimization**: Use time-series databases (TimescaleDB, InfluxDB) for metrics
2. **Partitioning**: Partition tables by date for better performance
3. **Caching**: Add Redis caching layer for frequently accessed data
4. **Backups**: Automated backup and restore procedures
5. **Analytics**: Pre-computed analytics tables for fast reporting
6. **Replication**: Read replicas for query performance

## Notes

- **Optional**: Database is optional - system works without it (uses in-memory SQLite for tests)
- **Backward Compatible**: Existing code continues to work without database
- **Graceful Degradation**: Persistence failures don't crash the system
- **Type Safe**: Uses SQLAlchemy models with full type hints

