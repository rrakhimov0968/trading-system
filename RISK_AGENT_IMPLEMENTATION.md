# RiskAgent Implementation Summary

## Overview

Implemented a comprehensive RiskAgent that enforces hard-coded risk rules and calculates position sizing for trading signals. The agent ensures no trade exceeds risk limits and provides optional LLM advisory explanations for rejections.

## ✅ What Was Implemented

### 1. **RiskAgent** (`agents/risk_agent.py`)

A deterministic-first agent for risk management:

**Key Features:**
- ✅ Inherits from `BaseAgent` for consistency
- ✅ Hard-coded rule enforcement (code enforces, never LLM)
- ✅ Position sizing based on confidence and risk
- ✅ Optional LLM advisor for explaining rejections (not in enforcement path)
- ✅ Account balance integration
- ✅ Daily loss tracking
- ✅ Comprehensive error handling
- ✅ Full type hints and documentation

**Key Methods:**
- `process()`: Main entry point - validates and sizes signals
- `enforce_rules()`: Hard-coded risk rule checks
- `calculate_position_sizing()`: Calculates qty and risk_amount
- `reset_daily_loss()`: Reset daily loss tracking
- `health_check()`: Verifies agent status

### 2. **Signal Model Extension** (`models/signal.py`)

Added risk-specific fields:
- `qty`: Calculated position size (in shares)
- `risk_amount`: Risk per trade in dollars
- `approved`: Whether signal passed risk checks

### 3. **Configuration** (`config/settings.py`)

Added RiskAgent configuration:
- `risk_max_per_trade`: Maximum risk per trade (default: 0.02 = 2%)
- `risk_max_daily_loss`: Maximum daily loss (default: 0.05 = 5%)
- `risk_min_confidence`: Minimum confidence to approve (default: 0.3)
- `risk_max_qty`: Maximum shares per trade (default: 1000)
- `risk_default_account_balance`: Default balance if can't fetch (default: 10000.0)
- `risk_use_llm_advisor`: Enable LLM advisor (default: false)

### 4. **Orchestrator Integration** (`core/orchestrator.py`)

Integrated RiskAgent into the pipeline:
- Step 4: Validates signals and calculates position sizing
- Step 5: Executes approved trades only
- Logs approved vs rejected signals
- Continues execution even if RiskAgent fails

### 5. **Tests** (`tests/test_risk_agent.py`)

Comprehensive test coverage:
- Initialization tests
- Rule enforcement tests
- Position sizing tests
- Process method tests
- Daily loss limit tests
- Health check tests

## 📊 How It Works

### Flow:

1. **QuantAgent** validates signals → `List[TradingSignal]`
2. **RiskAgent.process()** receives signals and ExecutionAgent
3. For each signal:
   - **Enforce Rules**:
     - Check minimum confidence threshold
     - Validate risk per trade doesn't exceed 2%
     - Validate daily loss limit
   - **Calculate Position Sizing**:
     - Calculate risk amount (scaled by confidence)
     - Calculate stop distance (ATR-based)
     - Calculate quantity: risk_amount / stop_distance
     - Validate against all rules
4. Returns approved signals with qty and risk_amount

### Risk Rules:

**Rule 1: Minimum Confidence**
- Signals with confidence < 0.3 are rejected
- Prevents low-quality signals from being executed

**Rule 2: Maximum Risk Per Trade (2%)**
- Risk amount = max_risk_per_trade * account_balance * confidence
- Absolute maximum = 2% of account balance
- Position sizing ensures risk doesn't exceed this

**Rule 3: Daily Loss Limit (5%)**
- Cumulative daily risk tracked
- New trades rejected if would exceed 5% daily limit
- Reset at start of trading day

### Position Sizing Logic:

```python
# Risk amount (scaled by confidence)
risk_amount = max_risk_per_trade * account_balance * confidence

# Stop distance (ATR-based, conservative 1.5x)
stop_distance = ATR * 1.5

# Quantity calculation
qty = risk_amount / stop_distance

# Clamp to bounds
qty = max(1, min(int(qty), max_qty))
```

### LLM Advisor (Optional):

If `risk_use_llm_advisor=true`:
- Explains rejections (why signal was rejected)
- Provides sizing rationale for approved signals
- Suggests alternatives for rejected signals
- Uses OpenAI or Claude (prefers OpenAI)

## 🔧 Configuration

### Environment Variables

```bash
# RiskAgent thresholds
RISK_MAX_PER_TRADE=0.02          # Maximum risk per trade (default: 2%)
RISK_MAX_DAILY_LOSS=0.05         # Maximum daily loss (default: 5%)
RISK_MIN_CONFIDENCE=0.3          # Minimum confidence (default: 0.3)
RISK_MAX_QTY=1000                # Maximum shares per trade (default: 1000)
RISK_DEFAULT_ACCOUNT_BALANCE=10000.0  # Default balance (default: $10,000)

# Optional LLM advisor (requires OPENAI_API_KEY or ANTHROPIC_API_KEY)
RISK_USE_LLM_ADVISOR=false       # Enable LLM for explanations (default: false)
```

## 📝 Usage Example

### Manual Usage

```python
from agents.risk_agent import RiskAgent
from models.signal import TradingSignal, SignalAction
from config.settings import get_config

config = get_config()
agent = RiskAgent(config=config)

# Signals from QuantAgent
signals = [
    TradingSignal(
        symbol="AAPL",
        action=SignalAction.BUY,
        strategy_name="TrendFollowing",
        confidence=0.75,
        timestamp=datetime.now(),
        price=150.0,
        historical_data=market_data["AAPL"].to_dataframe()
    )
]

# Execution agent for account balance
execution_agent = ExecutionAgent(config=config)

# Validate and size positions
approved_signals = agent.process(signals, execution_agent=execution_agent)

# Only approved signals have qty and risk_amount
for signal in approved_signals:
    if signal.approved:
        print(f"{signal.symbol}: {signal.qty} shares, risk=${signal.risk_amount:.2f}")
```

### In Orchestrator

The orchestrator automatically:
1. Fetches market data (DataAgent)
2. Generates signals (StrategyAgent)
3. Validates signals (QuantAgent)
4. **Risk checks and position sizing (RiskAgent)** ✅ **NEW**
5. Executes approved trades (ExecutionAgent)
6. (Future) Audit logs (AuditAgent)

## 🧪 Testing

Run tests with:

```bash
# Run all RiskAgent tests
pytest tests/test_risk_agent.py -v

# Run with coverage
pytest tests/test_risk_agent.py --cov=agents.risk_agent

# Run specific test
pytest tests/test_risk_agent.py::TestPositionSizing::test_calculate_position_sizing
```

## 📋 Risk Rules Details

### Rule Enforcement:

```python
# Rule 1: Minimum Confidence
if confidence < 0.3:
    raise RiskCheckError("Confidence too low")

# Rule 2: Max Risk Per Trade (after sizing)
if actual_risk > 2% of account:
    raise RiskCheckError("Exceeds max risk per trade")

# Rule 3: Daily Loss Limit
if current_daily_loss + new_risk > 5% of account:
    raise RiskCheckError("Would exceed daily loss limit")
```

### Position Sizing Example:

```
Account Balance: $10,000
Signal Confidence: 0.75
Max Risk Per Trade: 2% = $200

Risk Amount: $200 * 0.75 = $150
ATR: $2.00
Stop Distance: $2.00 * 1.5 = $3.00

Quantity: $150 / $3.00 = 50 shares
Actual Risk: 50 * $3.00 = $150 (1.5% of account)
```

## 🎯 Architecture Alignment

The implementation follows the design specifications:

✅ **Role**: Hard-coded rule enforcement  
✅ **LLM Role**: Optional advisor (explains rejections, not in enforcement path)  
✅ **Provider**: OpenAI or Claude (if enabled)  
✅ **Flow**: Risk checks → Position sizing → Approved signals  
✅ **Focus**: Unbreakable rules (code enforces, never LLM)

## ⚠️ Important Notes

1. **Account Balance**: Fetched from ExecutionAgent if available
2. **Daily Loss Tracking**: In-memory (reset on restart). In production, persist to DB
3. **Position Sizing**: Based on ATR stop distance (conservative 1.5x)
4. **HOLD Signals**: Automatically approved (no position sizing needed)
5. **LLM Optional**: Works perfectly without LLM (code-only enforcement)

## 🚀 Benefits

1. **Risk Control**: Hard limits prevent excessive risk
2. **Position Sizing**: Systematic sizing based on confidence and volatility
3. **Unbreakable Rules**: Code enforces, LLM only advises
4. **Account Protection**: Daily loss limits protect account
5. **Deterministic**: Same input always produces same validation

## 📚 Integration

### Complete Pipeline Flow:

```
DataAgent → MarketData
    ↓
StrategyAgent → TradingSignal[]
    ↓
QuantAgent → Validated TradingSignal[] (adjusted confidence)
    ↓
RiskAgent → Approved TradingSignal[] (with qty, risk_amount) ✅ NEW
    ↓
ExecutionAgent → Order Execution
```

### Execution Integration:

ExecutionAgent now receives signals with:
- `qty`: Calculated position size
- `risk_amount`: Risk for this trade
- `approved`: Risk validation status
- `stop_loss`: Calculated stop loss level

## ✅ Success Criteria Met

- ✅ Accepts `List[TradingSignal]` from QuantAgent
- ✅ Enforces hard-coded rules (2% per trade, daily limits)
- ✅ Calculates position sizing based on confidence and risk
- ✅ Optional LLM advisor for explanations (not in enforcement)
- ✅ Deterministic-first (code enforces, LLM advises)
- ✅ Integrated with orchestration loop
- ✅ Executes approved trades only
- ✅ Comprehensive test coverage

The RiskAgent is now fully functional and integrated into the trading system pipeline!

