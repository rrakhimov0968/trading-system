# Critical Fixes: Multiple Orders & Over-Leveraging

## Problem Summary

The system was placing **multiple orders per symbol** and **over-leveraging** (179% of account value), causing:
- 13 separate AAPL orders instead of 1
- $179,094 in purchases with only $100,000 account
- Multiple orders within minutes for the same symbol
- No position size limits

## Fixes Implemented

### ✅ Fix 1: Position Checking Before BUY Orders

**Location**: `core/orchestrator.py` and `core/async_orchestrator.py`

**What it does**:
- Checks if position already exists before placing BUY order
- Skips BUY if position exists (prevents duplicate orders)
- Only places SELL if position exists

**Code Added**:
```python
# Get current positions
current_positions = self.execution_agent.get_positions()
position_symbols = {pos.symbol for pos in current_positions}

if signal.action == SignalAction.BUY:
    if signal.symbol in position_symbols:
        logger.warning(f"🚫 SKIPPING BUY: Already have position in {signal.symbol}")
        continue  # Don't place order!
```

### ✅ Fix 2: Equal Weight Position Sizing

**Location**: `agents/risk_agent.py`

**What it does**:
- Allocates equal % of account to each symbol
- Max 20% per position (safety limit)
- Matches backtesting assumptions

**How it works**:
```
Account: $100,000
Symbols: 5 (SPY, QQQ, AAPL, GOOGL, NVDA)
Position per symbol: $100,000 / 5 = $20,000
Max per position: 20% = $20,000

AAPL @ $265: $20,000 / $265 = 75 shares ✅
```

**Before**: Unlimited position sizing (could buy $50K+ per symbol)
**After**: Equal weight, max 20% per symbol

### ✅ Fix 3: Order Cooldown / Deduplication

**Location**: `core/order_tracker.py` (new file)

**What it does**:
- Prevents multiple orders for same symbol within 24 hours
- Tracks last order time per symbol
- Clears tracking after SELL so symbol can be bought again

**How it works**:
```python
order_tracker = OrderTracker(cooldown_minutes=1440)  # 24 hours

if not order_tracker.can_place_order('AAPL'):
    # Skip - too soon since last order
    continue

# Place order
execute_order(...)
order_tracker.record_order('AAPL')  # Mark as placed
```

### ✅ Fix 4: Position Size Validation

**Location**: Both orchestrators

**What it does**:
- Validates position size doesn't exceed 25% of account
- Safety check even if RiskAgent calculated it correctly
- Double-layer protection

**Code Added**:
```python
position_value = signal.qty * signal.price
if position_value > account_value * 0.25:
    logger.error(f"🚫 Position size too large: ${position_value} > 25% of account")
    continue  # Skip order
```

## How It Works Now

### Correct Flow

1. **Daily Check** (system runs every iteration):
   ```
   Get current positions from Alpaca
   
   For each symbol:
     - If BUY signal AND no position exists:
         * Check cooldown (24 hours since last order)
         * Calculate position size: account / num_symbols (max 20%)
         * Validate position size < 25% of account
         * Place ONE order
         * Record order time
     
     - If SELL signal AND position exists:
         * Get position quantity
         * Place SELL order
         * Clear cooldown tracking (can buy again later)
   ```

2. **Position Sizing Example**:
   ```
   Account: $100,000
   Symbols: 5
   
   AAPL BUY signal:
     Position value = $100,000 / 5 = $20,000
     AAPL price = $265
     Shares = $20,000 / $265 = 75 shares ✅
   
   Result: ONE order for 75 shares (not 190!)
   ```

3. **Order Deduplication**:
   ```
   09:33:09 - BUY 75 AAPL ✅ (first order)
   09:34:12 - BUY signal for AAPL → SKIPPED (cooldown active) ✅
   09:35:00 - BUY signal for AAPL → SKIPPED (position exists) ✅
   ```

## Additional Safety Checks

### ✅ Fix 5: Daily Execution Limit

**Location**: `core/order_tracker.py`

**What it does**:
- Limits total orders per day across all symbols (default: 10 orders/day)
- Prevents runaway trading even if multiple signals fire
- Resets at midnight automatically

**How it works**:
```python
order_tracker = OrderTracker(
    cooldown_minutes=1440,  # 24 hours
    max_orders_per_day=10   # Maximum 10 orders per day
)

if total_orders_today >= 10:
    # Block all new orders until tomorrow
    return False, "Daily limit reached"
```

### ✅ Fix 6: Account Health Validation

**Location**: `core/orchestrator.py` and `core/async_orchestrator.py`

**What it does**:
- Validates account health before each iteration
- Checks for excessive drawdown (>20% from initial)
- Warns on low cash (<15% of equity)
- Blocks trading if buying power is negative (margin call risk)

**Checks Performed**:
1. **Drawdown Check**: Equity hasn't dropped >20% from initial
2. **Cash Reserve**: At least 15% cash available (warning only)
3. **Buying Power**: Must be positive (blocks if negative)

### ✅ Fix 7: Position Reconciliation

**Location**: `core/orchestrator.py` and `core/async_orchestrator.py`

**What it does**:
- Compares tracked positions with actual broker positions
- Detects manual closes/opens
- Synchronizes order tracker with reality
- Runs once per iteration

**How it works**:
```
Get positions from broker → Compare with tracker → Sync discrepancies

If broker has position but tracker doesn't:
  → Add to tracker (prevent duplicate orders)

If tracker has position but broker doesn't:
  → Clear from tracker (was manually closed)
```

### ✅ Fix 8: Emergency Kill Switch

**Location**: `core/orchestrator.py` and `core/async_orchestrator.py`

**What it does**:
- Checks for `EMERGENCY_STOP` file before each iteration
- Immediately stops all trading if file exists
- Prevents orders even if system continues running

**Usage**:
```bash
# To stop trading immediately:
touch EMERGENCY_STOP

# To resume trading:
rm EMERGENCY_STOP
```

**When to use**:
- You notice unexpected behavior
- Account drawdown exceeds comfort zone
- Need to pause for investigation
- Market conditions change dramatically

## Results

### Before Fixes:
- ❌ 13 separate AAPL orders
- ❌ $179,094 purchased with $100K account
- ❌ 179% leveraged
- ❌ Multiple orders per minute
- ❌ No daily limits
- ❌ No account health checks
- ❌ No position reconciliation
- ❌ No emergency stop

### After Fixes:
- ✅ 1 order per symbol (max)
- ✅ Equal weight allocation ($20K per symbol)
- ✅ Max 20% per position (safety limit)
- ✅ 24-hour cooldown prevents duplicates
- ✅ Position checking prevents over-buying
- ✅ Daily limit: 10 orders/day maximum
- ✅ Account health validation (drawdown, cash, buying power)
- ✅ Position reconciliation (syncs with broker)
- ✅ Emergency kill switch (touch EMERGENCY_STOP to stop)

## Testing the Fixes

1. **Verify Position Checking**:
   ```bash
   # Run system and check logs
   # Should see: "🚫 SKIPPING BUY: Already have position" messages
   ```

2. **Verify Position Sizing**:
   ```bash
   # Check RiskAgent logs
   # Should see: "Equal weight position sizing: $20,000 / $price = X shares"
   ```

3. **Verify Cooldown**:
   ```bash
   # Place one order, then try another within 24 hours
   # Should see: "🚫 SKIPPING BUY: Cooldown active"
   ```

## Configuration

You can adjust these settings via environment variables:

```bash
# Order cooldown (in minutes)
# Default: 1440 (24 hours)
# Set to 60 for 1-hour cooldown, or 43200 for 30 days
export ORDER_COOLDOWN_MINUTES=1440

# Maximum orders per day (across all symbols)
# Default: 10
export MAX_ORDERS_PER_DAY=10

# Initial account equity (for drawdown calculation)
# Default: 100000
export INITIAL_ACCOUNT_EQUITY=100000

# Max position % (in RiskAgent)
# Already set to 20% (0.20) - see agents/risk_agent.py
```

## Safety Checks Summary

| Check | Location | Blocks Trading | When |
|-------|----------|----------------|------|
| Position Check | Orchestrator | Yes | If position already exists |
| Order Cooldown | OrderTracker | Yes | If < 24 hours since last order |
| Daily Limit | OrderTracker | Yes | If >= 10 orders today |
| Account Drawdown | Orchestrator | Yes | If >20% from initial |
| Buying Power | Orchestrator | Yes | If negative (margin call) |
| Emergency Stop | Orchestrator | Yes | If EMERGENCY_STOP file exists |
| Position Reconciliation | Orchestrator | No | Runs every iteration (info only) |
| Cash Reserve | Orchestrator | No | Warning only if <15% cash |

## Important Notes

1. **Existing Positions**: The fixes work going forward. You may need to manually close duplicate positions from before the fix.

2. **Cooldown Reset**: Cooldown clears after SELL, so you can buy again after exiting.

3. **Position Sync**: The system fetches positions from Alpaca on each iteration, so it knows about all positions.

4. **Backtesting Match**: Equal weight position sizing now matches backtesting assumptions (each symbol gets equal allocation).

5. **Daily Limit Reset**: Daily order count resets automatically at midnight (based on system time).

6. **Emergency Stop**: The `EMERGENCY_STOP` file check happens at the start of each iteration, so you can create it anytime to immediately stop trading.

7. **Account Health**: If account health check fails, the iteration is skipped entirely. Check logs to see why (drawdown, buying power, etc.).

8. **Position Reconciliation**: This runs every iteration and automatically syncs tracked positions with broker positions. Manual trades are detected and handled gracefully.

## Monitoring

Watch the logs for:
- `🚫 SKIPPING BUY` messages (good - preventing duplicates)
- `✅ Order executed` messages (orders that passed all checks)
- `Equal weight position sizing` messages (showing calculated sizes)

Run the dashboard to monitor in real-time:
```bash
streamlit run monitoring.py
```

