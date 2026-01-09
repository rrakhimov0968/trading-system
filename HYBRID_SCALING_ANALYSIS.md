# Hybrid Stock Selection - System Capacity & Implementation Analysis

## Executive Summary

**Can your trading system handle 29 symbols?** ✅ **YES, with proper implementation**

**Will it exhaust API limits?** ❌ **NO, if using batch fetching and caching**

This document analyzes the system capacity, API limits, and provides a production-ready implementation plan for scaling from 5 to 29 symbols using a hybrid tiered approach.

---

## 1. API Rate Limit Analysis

### Current Limits

**Alpaca Data API:**
- **Standard Limit**: 200 requests/minute
- **Rate Limiter**: Already implemented in `DataAgent` (shared across instances)
- **Current Behavior**: Sequential fetching (1 symbol = 1 API call)

### API Usage Projections

**Current Setup (5 symbols):**
- 1 iteration = 5 API calls
- Loop interval: 60 seconds
- **Usage**: ~5 req/min (2.5% of limit) ✅ Very safe

**Proposed Setup (29 symbols) - Sequential:**
- 1 iteration = 29 API calls
- Loop interval: 60 seconds
- **Usage**: ~29 req/min (14.5% of limit) ✅ Still safe, but inefficient

**Proposed Setup (29 symbols) - Batch Fetching:**
- 1 iteration = 1 API call (Alpaca supports multi-symbol requests)
- Loop interval: 60 seconds
- **Usage**: ~1 req/min (0.5% of limit) ✅ Optimal

**Verdict**: With batch fetching, API limits are not a concern.

---

## 2. System Capacity Assessment

### Current Architecture Strengths

✅ **Rate Limiting**: Already implemented  
✅ **Caching**: In-memory cache reduces redundant calls  
✅ **Error Handling**: Robust retry logic  
✅ **Async Support**: Async orchestrator available  
✅ **Position Tracking**: OrderTracker prevents duplicates  

### Current Architecture Limitations

❌ **Sequential Fetching**: Loops through symbols one by one  
❌ **No Batch Support**: Each symbol = separate API call  
❌ **Equal Weight Only**: No tiered allocation  
❌ **No Fractional Shares**: Can't handle expensive stocks in Tier 3  
❌ **Fixed Symbol List**: No dynamic symbol selection  

### Capacity Estimate

**With Optimizations (Batch Fetching + Caching):**
- ✅ Can easily handle 29 symbols
- ✅ Can scale to 50-100 symbols if needed
- ✅ Memory usage: ~50-100MB for cached data
- ✅ CPU usage: Minimal (I/O bound, not CPU bound)

**Bottlenecks (if any):**
1. Strategy calculation (per symbol) - minimal CPU time
2. Database writes - optimized with batching
3. Network latency - mitigated by batching

---

## 3. Tiered Position Sizing Analysis

### The "Expensive Stock Trap" (Critical Issue)

**Problem Identified:**
```
Account: $10,000
Tier 3 Allocation: 30% = $3,000
Tier 3 Symbols: 20
Per-symbol allocation: $3,000 / 20 = $150

COST (Costco): ~$900/share
MSFT (Microsoft): ~$420/share

Result: 
  int($150 / $900) = 0 shares ❌
  int($150 / $420) = 0 shares ❌
```

**Impact:**
- Silent failure (no errors logged)
- Biased exposure (only cheap stocks get bought)
- Distorted diversification
- False backtest optimism

**Solution:**
1. **Fractional Shares** (Required for < $25k accounts)
2. **Minimum Notional** ($10-25 minimum order size)
3. **Dynamic Tier Compression** (Reduce active Tier 3 symbols if prices are high)

---

## 4. Implementation Plan

### Phase 1: Batch Data Fetching (Critical for API Efficiency)

**Implementation:**
- Modify `DataAgent.process()` to support batch requests
- Alpaca supports comma-separated symbols: `symbol_or_symbols=['AAPL','MSFT','GOOGL']`
- Fetch all 29 symbols in 1 API call instead of 29 calls

**Code Changes:**
```python
# agents/data_agent.py
def process(self, symbols: List[str], ...) -> Dict[str, MarketData]:
    if self.provider == DataProvider.ALPACA:
        # BATCH FETCH (1 API call for all symbols)
        return self._fetch_alpaca_batch(symbols, timeframe, ...)
    else:
        # Fallback to sequential for other providers
        ...
```

**Benefits:**
- 29x reduction in API calls
- Faster data fetching (parallel processing)
- More reliable (single request vs 29)

---

### Phase 2: Tiered Position Sizing (Addresses Expensive Stock Trap)

**Tier Structure:**
```
TIER 1 (Index ETFs): 40% of capital, 4 symbols
  - SPY, QQQ, DIA, IWM
  - Per symbol: 40% / 4 = 10% each

TIER 2 (Sector ETFs): 30% of capital, 5 symbols  
  - XLK, XLF, XLV, XLE, XLY
  - Per symbol: 30% / 5 = 6% each

TIER 3 (Individual Stocks): 30% of capital, 20 symbols
  - AAPL, MSFT, GOOGL, etc.
  - Per symbol: 30% / 20 = 1.5% each
  - **Problem**: Can result in 0 shares for expensive stocks
```

**Fixes:**
1. **Fractional Shares Support** (Required)
2. **Dynamic Tier Compression** (Reduce active symbols if prices too high)
3. **Minimum Notional Enforcement** ($10-25 minimum)
4. **Immutable Config** (Never mutate tier allocation at runtime)

---

### Phase 3: Scanner-Driven Monitoring (Reduces API Load Further)

**Workflow:**
```
4:00 PM ET: Market Closes
4:05 PM ET: Scanner runs, generates candidates.json
  - Scans all 29 symbols for oversold conditions
  - Selects top 5-10 opportunities
  - Saves to candidates.json

9:30 AM ET (Next Day): Main system starts
  - Reads candidates.json
  - Monitors only 5-10 symbols (instead of all 29)
  - Places trades on signals
```

**Benefits:**
- Focus on high-probability opportunities
- Reduced API calls (5-10 symbols vs 29)
- Better signal-to-noise ratio
- Mandatory baseline monitoring (SPY/QQQ always monitored)

---

### Phase 4: Safety Fixes (Critical for Production)

**Issue 1: Mutating Tier Allocation at Runtime**
```python
# ❌ DANGEROUS (mutating config)
config['allocation'] = (self.min_position * n_symbols) / self.account_value

# ✅ SAFE (compute effective, don't mutate)
effective_allocation = min(
    config['allocation'],
    (self.min_position * n_symbols) / self.account_value
)
```

**Issue 2: Over-Allocation Risk**
```python
# Always enforce tier cap AFTER sizing
tier_cap = self.account_value * config['allocation']
position_amount = min(position_amount, tier_cap * 0.20)  # Max 20% per symbol in tier
```

**Issue 3: Fractional Shares ≠ Infinite Liquidity**
```python
# Enforce minimum notional
MIN_NOTIONAL = 10.0  # $10 minimum order
if position_amount < MIN_NOTIONAL:
    return {'skipped': True, 'reason': 'Notional too small'}
```

**Issue 4: Scanner Bias Feedback Loop**
```python
# Always monitor baseline symbols
self.focus_symbols = (
    scanner_results[:5]  # Top scanner picks
    + ['SPY', 'QQQ']     # Mandatory baseline
    + self.random_control_symbol()  # Random control
)
```

---

## 5. Trade Frequency Projections

### Current (5 symbols):
- Trades per year: ~35 (7 per symbol)
- Typical positions: 1-2 at a time
- Days with activity: ~50/year

### With 29 symbols (Expected):
- Trades per year: ~120-150 (4-5 per symbol)
- Typical positions: 5-10 at a time
- Days with activity: ~100/year
- **Position holding period**: 7-14 days (unchanged)

### With Scanner (Optimized):
- Scanner selects: 5-10 opportunities daily
- Trades per year: ~80-100 (focused on best signals)
- Typical positions: 3-7 at a time
- Days with activity: ~80/year
- **Higher win rate** (better signal quality)

---

## 6. Configuration Requirements

### Minimum Account Size

**Without Fractional Shares:**
- Recommended: $25,000+
- Tier 3 allocation: $7,500 (30% of $25k)
- Per-symbol: $375 (can buy most stocks)

**With Fractional Shares:**
- Minimum: $5,000
- Tier 3 allocation: $1,500 (30% of $5k)
- Per-symbol: $75 (fractional shares required)

**Recommended:**
- Start with $10,000+ (comfortable buffer)
- Enable fractional shares for safety

### Environment Variables

```bash
# Enable tiered allocation
ENABLE_TIERED_ALLOCATION=true

# Tier allocations (must sum to 1.0)
TIER1_ALLOCATION=0.40
TIER2_ALLOCATION=0.30
TIER3_ALLOCATION=0.30

# Fractional shares (required for < $25k)
ENABLE_FRACTIONAL_SHARES=true
MIN_ORDER_NOTIONAL=10.0

# Scanner integration
USE_SCANNER=true
SCANNER_CANDIDATES_FILE=candidates.json

# Baseline monitoring (always monitor these)
BASELINE_SYMBOLS=SPY,QQQ
```

---

## 7. Risk Considerations

### Position Concentration Risk

**With 29 symbols:**
- Max per symbol: 10% (Tier 1) or 1.5% (Tier 3)
- Max tier exposure: 40% (Tier 1), 30% (Tier 2/3)
- **Diversification**: Excellent (29 symbols across sectors)

**With Scanner (5-10 symbols):**
- Max per symbol: 20% (if all Tier 1)
- Typical per symbol: 10-15%
- **Diversification**: Good (5-10 symbols)

### Execution Risk

**Batch Orders:**
- Risk: All 29 symbols trigger at once
- Mitigation: OrderTracker prevents rapid-fire orders
- Daily limit: 10 orders/day (safety cap)

**Fractional Shares:**
- Risk: Execution slippage on small orders
- Mitigation: Minimum notional ($10-25)
- Market impact: Minimal (small order sizes)

---

## 8. Implementation Checklist

### Pre-Deployment

- [ ] Implement batch data fetching
- [ ] Add tiered position sizing (with safety fixes)
- [ ] Enable fractional shares support
- [ ] Create scanner script
- [ ] Add mandatory baseline monitoring
- [ ] Test with paper trading ($10k account)
- [ ] Verify API rate limits not exceeded
- [ ] Test Tier 3 with expensive stocks (COST, MSFT)

### Monitoring

- [ ] Track API request rate (should be < 10 req/min)
- [ ] Monitor position sizes (verify tier allocations)
- [ ] Check for zero-share allocations (Tier 3 trap)
- [ ] Validate scanner effectiveness (win rate)
- [ ] Ensure baseline symbols always monitored

### Rollout Plan

1. **Week 1**: Test with 10 symbols (Tier 1 + Tier 2 only)
2. **Week 2**: Add Tier 3 (5 stocks first, test fractional shares)
3. **Week 3**: Full 29 symbols + scanner integration
4. **Week 4**: Monitor and optimize

---

## 9. Success Metrics

**API Efficiency:**
- Target: < 5 API calls per iteration (with batch + scanner)
- Current: 29 calls per iteration (sequential)
- Improvement: 83% reduction

**Position Sizing:**
- Target: All Tier 3 symbols can be bought (fractional shares)
- No zero-share allocations
- Tier allocations maintained (immutable config)

**Trade Quality:**
- Win rate: Maintain or improve (scanner selects better signals)
- Position count: 5-10 positions typical (diversified)
- Days with activity: 80-100/year (vs 50 with 5 symbols)

---

## 10. Conclusion

**Your trading system CAN handle 29 symbols** with proper implementation:

1. ✅ **API Limits**: Not a concern with batch fetching
2. ✅ **System Capacity**: More than adequate (can scale to 100+)
3. ⚠️ **Tier 3 Trap**: Must fix (fractional shares required)
4. ✅ **Performance**: Improved with scanner + batch fetching

**Recommended Path:**
1. Implement batch fetching (Phase 1)
2. Add tiered sizing with safety fixes (Phase 2)
3. Test with 10 symbols first (Week 1)
4. Add scanner + full 29 symbols (Week 2-3)

**Critical Warnings:**
- ❌ Never mutate tier allocation at runtime
- ❌ Always enforce minimum notional for fractional shares
- ❌ Always monitor baseline symbols (prevent scanner bias)
- ❌ Test Tier 3 with expensive stocks before production

---

**Last Updated:** 2025-01-06
