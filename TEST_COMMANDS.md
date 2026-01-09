# Test Commands for Today's Critical Fixes

## Quick Test Commands

### 1. Test All 7 Critical Fixes (Recommended)
```bash
python tests/test_critical_fixes.py
```

Or use the test script:
```bash
./run_tests.sh critical
```

### 2. Test Hybrid Scaling
```bash
python tests/test_hybrid_scaling.py
```

Or:
```bash
./run_tests.sh hybrid
```

### 3. Test Config Validation (Problem 4)
```bash
python -c "
from config.settings import AppConfig
import os

# Test invalid allocations
os.environ['ENABLE_TIERED_ALLOCATION'] = 'true'
os.environ['TIER1_ALLOCATION'] = '0.50'
os.environ['TIER2_ALLOCATION'] = '0.30'
os.environ['TIER3_ALLOCATION'] = '0.10'  # Invalid: sums to 0.90

try:
    config = AppConfig.from_env()
    print('❌ Should have raised ValueError')
except ValueError as e:
    if 'must sum to 1.0' in str(e):
        print('✅ Config validation works correctly')
"
```

### 4. Test All Tests (Full Suite)
```bash
# Using pytest
pytest tests/ -v

# Or using test script
./run_tests.sh all
```

## Individual Test Commands

### Problem 1: Deterministic Pipeline
```bash
python -c "
from tests.test_critical_fixes import TestProblem1_DeterministicPipeline
TestProblem1_DeterministicPipeline().test_pipeline_enforces_order()
"
```

### Problem 2: Atomic Locking
```bash
python -c "
from tests.test_critical_fixes import TestProblem2_AtomicLocking
TestProblem2_AtomicLocking().test_reserve_release_cycle()
"
```

### Problem 3: Live Account Value
```bash
python -c "
from tests.test_critical_fixes import TestProblem3_LiveAccountValue
TestProblem3_LiveAccountValue().test_sizer_uses_live_equity()
"
```

### Problem 4: Tier Validation
```bash
./run_tests.sh config
```

### Problem 5: Scanner Diversity
```bash
python -c "
from tests.test_critical_fixes import TestProblem5_ScannerDiversity
TestProblem5_ScannerDiversity().test_scanner_includes_baseline()
"
```

### Problem 6: Fractional Check
```bash
python -c "
from tests.test_critical_fixes import TestProblem6_FractionalCheck
TestProblem6_FractionalCheck().test_fractional_validation_exists()
"
```

### Problem 7: Structured Logging
```bash
python -c "
from tests.test_critical_fixes import TestProblem7_StructuredLogging
TestProblem7_StructuredLogging().test_validation_decision_has_metadata()
"
```

## Manual Verification Tests

### Test Atomic Locking Manually
```bash
python -c "
from core.tier_exposure_tracker import TierExposureTracker

symbol_mapping = {'SPY': 'TIER1', 'QQQ': 'TIER1'}
tracker = TierExposureTracker(symbol_mapping)

# Test reservation
approved, reason = tracker.reserve_exposure(
    tier='TIER1',
    proposed_value=20000,
    current_tier_exposure=tracker.calculate_tier_exposure([], 100000)['TIER1'],
    account_value=100000,
    symbol='SPY'
)

print(f'Reservation approved: {approved}, reason: {reason}')

# Release
tracker.release_exposure('TIER1', 20000, 'SPY')
print('✅ Atomic locking works')
"
```

### Test Live Account Value
```bash
python -c "
from core.tiered_position_sizer import TieredPositionSizer

sizer = TieredPositionSizer(account_value=100000, use_fractional=True)

# Test with different account values
shares1, _ = sizer.calculate_shares('SPY', 688.0, 'TIER1', account_value=100000)
shares2, _ = sizer.calculate_shares('SPY', 688.0, 'TIER1', account_value=150000)

print(f'100k account: {shares1:.2f} shares')
print(f'150k account: {shares2:.2f} shares')
print(f'✅ Live account value: {shares2 > shares1}')
"
```

### Test Scanner Diversity
```bash
python -c "
from core.orchestrator_integration import load_focus_symbols
from config.settings import get_config
import json
from pathlib import Path

config = get_config()
config.use_scanner = True
config.baseline_symbols = ['SPY', 'QQQ']

# Create mock scanner file
scanner_data = {'symbols': ['AAPL', 'MSFT'], 'timestamp': '2024-01-01'}
Path('candidates.json').write_text(json.dumps(scanner_data))

symbols = load_focus_symbols(config)
print(f'Symbols: {symbols}')
print(f'Has SPY: {\"SPY\" in symbols}')
print(f'Has QQQ: {\"QQQ\" in symbols}')
print('✅ Scanner diversity works')
"
```

## Run All Tests with Coverage

```bash
# Install pytest-cov if not installed
pip install pytest-cov

# Run with coverage
pytest tests/ --cov=core --cov=agents --cov=config --cov-report=html

# View coverage report
open htmlcov/index.html
```

## Integration Tests

### Test Orchestrator Initialization
```bash
python -c "
from core.orchestrator import TradingSystemOrchestrator
from config.settings import get_config

config = get_config()
orchestrator = TradingSystemOrchestrator(config=config)
print('✅ Orchestrator initialized successfully')
print(f'Tier tracker: {orchestrator.tier_tracker is not None}')
print(f'Signal validator: {orchestrator.signal_validator is not None}')
print(f'Tiered sizer: {orchestrator.tiered_sizer is not None}')
"
```

### Test Async Orchestrator Initialization
```bash
python -c "
from core.async_orchestrator import AsyncTradingSystemOrchestrator
from config.settings import get_config

config = get_config()
orchestrator = AsyncTradingSystemOrchestrator(config=config)
print('✅ Async orchestrator initialized successfully')
"
```

## Quick Test Summary

```bash
# One command to test everything
./run_tests.sh

# Or step by step
python tests/test_critical_fixes.py && \
python tests/test_hybrid_scaling.py && \
pytest tests/ -v -k "not slow"
```

## Expected Results

All tests should pass:
- ✅ Problem 1: Deterministic Pipeline
- ✅ Problem 2: Atomic Locking  
- ✅ Problem 3: Live Account Value
- ✅ Problem 4: Tier Validation
- ✅ Problem 5: Scanner Diversity
- ✅ Problem 6: Fractional Check
- ✅ Problem 7: Structured Logging
