#!/bin/bash
# Run all tests for today's critical fixes
# Usage: ./run_tests.sh [test_name]

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "🧪 Running Trading System Tests"
echo "================================"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test functions
test_critical_fixes() {
    echo -e "\n${YELLOW}Testing Critical Fixes (All 7 Problems)...${NC}"
    python tests/test_critical_fixes.py
}

test_hybrid_scaling() {
    echo -e "\n${YELLOW}Testing Hybrid Scaling...${NC}"
    python tests/test_hybrid_scaling.py
}

test_safety_checks() {
    echo -e "\n${YELLOW}Testing Safety Checks...${NC}"
    python test_safety_checks.py
}

test_all_unit() {
    echo -e "\n${YELLOW}Running All Unit Tests...${NC}"
    pytest tests/ -v --tb=short
}

test_orchestrator() {
    echo -e "\n${YELLOW}Testing Orchestrator Integration...${NC}"
    python -c "
from core.orchestrator_integration import *
from config.settings import get_config
config = get_config()
print('✅ Orchestrator integration imports successful')
"
}

test_config_validation() {
    echo -e "\n${YELLOW}Testing Config Validation...${NC}"
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
    if config.enable_tiered_allocation:
        print('❌ Should have raised ValueError')
        exit(1)
except ValueError as e:
    if 'must sum to 1.0' in str(e):
        print('✅ Config validation works correctly')
    else:
        print(f'❌ Wrong error: {e}')
        exit(1)
"
}

# Run specific test or all tests
if [ $# -eq 0 ]; then
    # Run all tests
    test_critical_fixes
    test_hybrid_scaling
    test_config_validation
    test_orchestrator
    
    echo -e "\n${GREEN}✅ All basic tests passed!${NC}"
    echo -e "${YELLOW}To run full pytest suite: pytest tests/ -v${NC}"
else
    case "$1" in
        critical)
            test_critical_fixes
            ;;
        hybrid)
            test_hybrid_scaling
            ;;
        safety)
            test_safety_checks
            ;;
        config)
            test_config_validation
            ;;
        orchestrator)
            test_orchestrator
            ;;
        all|unit)
            test_all_unit
            ;;
        *)
            echo "Unknown test: $1"
            echo "Usage: $0 [critical|hybrid|safety|config|orchestrator|all]"
            exit 1
            ;;
    esac
fi
