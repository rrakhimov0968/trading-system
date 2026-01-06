# Why Backtest Strategies Show Returns But FAIL

## The Issue

Your backtest results show positive returns (e.g., 25.83% for AAPL, 52.05% for GOOGL), but the status is marked as **❌ FAIL**. This is confusing because returns look good, but there's more to the story.

## Validation Criteria (ALL Must Pass)

A strategy **PASSES** only if **ALL THREE** criteria are met:

1. ✅ **Sharpe Ratio ≥ 1.5** (risk-adjusted return)
2. ✅ **Max Drawdown ≤ 8%** (maximum loss from peak)
3. ✅ **Win Rate ≥ 45%** (percentage of profitable trades)

## Why Your Strategies Are Failing

Looking at your results:

```
Symbol     Return       Sharpe     Max DD       Win Rate     Trades     Status
--------------------------------------------------------------------------------
AAPL            25.83%      0.23      -22.03%        0.00%         0  ❌ FAIL
GOOGL           52.05%      0.52      -21.77%        0.00%         0  ❌ FAIL
```

### Failure Breakdown:

1. **Sharpe Ratio: 0.23, 0.52** ❌
   - **Required:** ≥ 1.5
   - **Actual:** 0.23, 0.52 (WAY below threshold)
   - **Meaning:** Risk-adjusted returns are poor. A Sharpe of 0.23 means the strategy barely beats cash, with high volatility.

2. **Max Drawdown: -22.03%, -21.77%** ❌
   - **Required:** ≤ 8%
   - **Actual:** -22.03%, -21.77% (WAY above threshold)
   - **Meaning:** The strategy lost 22% from its peak. This is too risky - you could wipe out 22% of your capital in a drawdown.

3. **Win Rate: 0.00%** ❌
   - **Required:** ≥ 45%
   - **Actual:** 0.00% (below threshold)
   - **Meaning:** No winning trades (or no trades at all).

4. **Trades: 0** ⚠️ **CRITICAL ISSUE**
   - **Meaning:** **NO TRADES WERE EXECUTED!**
   - This is the root cause. If there are 0 trades, the strategy isn't generating actionable signals.

## Why Returns Are Misleading

The positive returns shown (25.83%, 52.05%) are **misleading** because:

1. **No Trades Executed**: When VectorBT has 0 trades, it may calculate returns based on:
   - Buy-and-hold (if you bought at start and held)
   - Calculation artifacts (portfolio metrics when no trades occur)
   - Price appreciation without actual trading

2. **These Returns Don't Reflect Strategy Performance**: Since no trades were executed, the returns don't represent how the strategy would actually perform.

## The Real Problem: 0 Trades

The **"Trades: 0"** is the critical issue. This means:

- ✅ Signals are being generated (you see "Generated 2512 entry signals and 1296 exit signals")
- ❌ But VectorBT isn't executing any trades from those signals
- ❌ This could be because:
  - Entry/exit signals aren't properly paired
  - Signals aren't generating actionable trade pairs
  - Strategy logic isn't creating valid trades
  - VectorBT requires proper entry/exit sequences

## What to Do

### Immediate Actions:

1. **Debug Signal Generation**: Check why signals aren't converting to trades
   - Verify entry/exit signal pairing
   - Check if signals are on the correct dates
   - Ensure signals are in the right format for VectorBT

2. **Lower Thresholds (Temporarily)**: If you want to test with lower standards:
   ```bash
   # Set environment variables
   export QUANT_MIN_SHARPE=0.5  # Lower from 1.5 to 0.5
   export QUANT_MAX_DRAWDOWN=0.25  # Increase from 8% to 25%
   ```
   **Note:** This is just for testing. Don't trade strategies that fail these thresholds.

3. **Fix Strategy Logic**: The strategies need to generate proper entry/exit pairs that VectorBT can execute.

### Long-Term Fixes:

1. **Review Strategy Implementation**: Ensure strategies generate valid BUY/SELL signals
2. **Check Signal Format**: Verify signals are properly formatted for VectorBT
3. **Test with Simple Strategy**: Try a simple buy-and-hold strategy first to verify the framework works
4. **Add Debug Logging**: Log when signals are generated vs when trades are executed

## Example: What "PASS" Looks Like

A passing strategy would look like:

```
Symbol     Return       Sharpe     Max DD       Win Rate     Trades     Status
--------------------------------------------------------------------------------
AAPL            15.50%      1.65      -6.50%       52.00%        45  ✅ PASS
```

Notice:
- Sharpe: 1.65 ≥ 1.5 ✅
- Max DD: -6.50% ≤ 8% ✅
- Win Rate: 52% ≥ 45% ✅
- Trades: 45 (actual trades executed) ✅

## Conclusion

**Your strategies are failing because:**
1. Sharpe ratio is too low (< 1.5)
2. Max drawdown is too high (> 8%)
3. Win rate is 0% (< 45%)
4. **Most importantly: 0 trades were executed**

The positive returns are misleading - they don't reflect actual strategy performance since no trades occurred. Focus on fixing the signal generation to create actionable trades first, then worry about optimizing returns.

