# Step-by-Step Guide: Installing Dependencies and Running E2E Tests

## Prerequisites

- Python 3.11 or 3.12 (recommended; Python 3.13 may have compatibility issues)
- pip (Python package manager)
- Virtual environment (venv)

## Step 1: Set Up Python Environment

### Option A: Using System Python (Recommended)

```bash
# Check Python version
python3 --version  # Should be 3.11 or 3.12

# Create a new virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### Option B: Using pyenv (If you have it installed)

```bash
# Install Python 3.12 (recommended)
pyenv install 3.12.7

# Set local Python version
pyenv local 3.12.7

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # macOS/Linux
```

**Verify activation**: You should see `(venv)` in your terminal prompt.

## Step 2: Upgrade pip and Install Build Tools

```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install build tools (helps with packages that need compilation)
pip install --upgrade setuptools wheel
```

## Step 3: Install Dependencies (Step by Step)

If you encounter issues, try installing in stages:

### Stage 1: Core Dependencies

```bash
pip install pandas==2.1.4 numpy python-dotenv==1.0.0
```

### Stage 2: API Integrations

```bash
pip install alpaca-py==0.21.0 anthropic==0.18.1 openai==1.12.0 groq==0.4.2
```

### Stage 3: Data and Analysis

```bash
pip install yfinance==0.2.36 pandas-ta scipy statsmodels
```

**Note**: If `pandas-ta` fails, try:
```bash
pip install pandas-ta --no-cache-dir
```

### Stage 4: Database (Optional - skip if not using PostgreSQL)

```bash
# Skip this if you're not using a database
pip install psycopg2-binary==2.9.9 sqlalchemy==2.0.25
```

### Stage 5: Utilities

```bash
pip install requests==2.31.0 aiohttp==3.9.1 python-dateutil==2.8.2 pytz==2023.3 pydantic==2.5.0
```

### Stage 6: Testing

```bash
pip install pytest==7.4.4 pytest-asyncio==0.23.3 pytest-cov==4.1.0 pytest-mock==3.12.0 requests-mock==1.11.1
```

### Stage 7: Monitoring (Optional)

```bash
pip install prometheus-client==0.19.0
```

### Alternative: Install All at Once

```bash
# If the staged approach worked, try installing everything:
pip install -r requirements.txt
```

## Step 4: Troubleshooting Common Installation Issues

### Issue 1: Package compilation fails (numpy, scipy, pandas-ta)

**Solution**: Install build dependencies first:
```bash
pip install --upgrade pip setuptools wheel
pip install numpy scipy  # Install these first as they're dependencies
pip install -r requirements.txt
```

### Issue 2: "No module named '_tkinter'" (macOS)

**Solution**: Install Python with tkinter:
```bash
# Using Homebrew:
brew install python-tk
```

### Issue 3: "ERROR: Failed building wheel" for pandas-ta

**Solution**: Try installing without cache:
```bash
pip install pandas-ta --no-cache-dir
# Or install from source:
pip install git+https://github.com/twopirllc/pandas-ta.git
```

### Issue 4: Python version incompatibility

**Solution**: Use Python 3.11 or 3.12:
```bash
# Check version
python --version

# If it's 3.13 or higher, create venv with specific version:
python3.12 -m venv venv  # If you have 3.12 installed
```

### Issue 5: SSL/TLS errors during installation

**Solution**: Upgrade certificates:
```bash
pip install --upgrade certifi
```

### Issue 6: Permission errors

**Solution**: Use `--user` flag or ensure venv is activated:
```bash
# Make sure venv is activated (check for (venv) in prompt)
source venv/bin/activate
pip install -r requirements.txt
```

## Step 5: Verify Installation

```bash
# Check if pytest is installed
pytest --version

# Check if all key packages are available
python -c "import pandas, numpy, pytest, yfinance; print('All packages imported successfully!')"

# If you see errors, install missing packages individually
```

## Step 6: Set Up Test Environment (Optional)

```bash
# Create .env.test file for test configuration
cat > .env.test << EOF
# Trading mode
TRADING_MODE=paper
LOG_LEVEL=DEBUG

# Alpaca (paper trading mode, mock keys are fine for tests)
ALPACA_API_KEY=test_api_key
ALPACA_SECRET_KEY=test_secret_key
ALPACA_PAPER=true

# Groq (mock key is fine for tests with mocks)
GROQ_API_KEY=test_groq_key

# Anthropic (mock key is fine for tests with mocks)
ANTHROPIC_API_KEY=test_anthropic_key

# Symbols for testing
SYMBOLS=SPY,QQQ,AAPL

# Agent configurations
QUANT_MIN_SHARPE=1.5
RISK_MAX_RISK_PER_TRADE=0.02
EOF
```

**Note**: E2E tests use mocks, so real API keys aren't required, but the config still needs these variables.

## Step 7: Run E2E Tests

### Run All Integration Tests

```bash
# Make sure you're in the project directory and venv is activated
cd /Users/ravshan/trading-system
source venv/bin/activate  # If not already activated

# Run all integration tests
pytest -m integration -v
```

### Run Specific E2E Test

```bash
# Run a specific test
pytest tests/test_pipeline.py::TestPipelineEndToEnd::test_full_pipeline_with_mocks -v
```

### Run with Detailed Output

```bash
# Verbose output with print statements
pytest -m integration -v -s
```

### Run with Coverage Report

```bash
# Generate coverage report
pytest -m integration --cov=. --cov-report=html

# View coverage report (opens in browser)
open htmlcov/index.html  # macOS
# or
xdg-open htmlcov/index.html  # Linux
```

### Run with Debugging

```bash
# Drop into debugger on failure
pytest -m integration --pdb

# Show local variables on failure
pytest -m integration -l
```

## Step 8: Quick Test Verification

Run this to verify your setup works:

```bash
# Quick smoke test
python -c "
import sys
print(f'Python version: {sys.version}')

try:
    import pandas as pd
    import numpy as np
    import pytest
    import yfinance as yf
    import pydantic
    print('✓ All core packages imported successfully')
except ImportError as e:
    print(f'✗ Import error: {e}')
    sys.exit(1)

# Test pytest can find tests
import subprocess
result = subprocess.run(['pytest', '--collect-only', '-m', 'integration'], 
                       capture_output=True, text=True)
if result.returncode == 0:
    print('✓ Pytest can collect integration tests')
else:
    print('✗ Pytest collection failed')
    print(result.stderr)
"
```

## Expected Test Output

When tests run successfully, you should see:

```
======================== test session starts ========================
platform darwin -- Python 3.12.x, pytest-7.4.4, ...
collected 5 items

tests/test_pipeline.py::TestPipelineEndToEnd::test_full_pipeline_with_mocks PASSED
tests/test_pipeline.py::TestPipelineEndToEnd::test_pipeline_with_no_signals PASSED
...

======================= 5 passed in 2.34s ========================
```

## Common Test Issues and Fixes

### Issue: "No tests collected"

**Fix**: Ensure you're in the project root directory:
```bash
cd /Users/ravshan/trading-system
pytest -m integration
```

### Issue: Import errors in tests

**Fix**: Ensure all dependencies are installed and venv is activated:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### Issue: Mock not working

**Fix**: Check that pytest-mock is installed:
```bash
pip install pytest-mock==3.12.0
```

### Issue: Tests timeout or hang

**Fix**: Check for un-mocked API calls:
```bash
# Run with timeout
pytest -m integration --timeout=30
```

## Next Steps

1. **Run full test suite**: `pytest` (runs all tests)
2. **Check test coverage**: `pytest --cov=. --cov-report=html`
3. **View test documentation**: See `E2E_TESTING.md` for detailed documentation

## Getting Help

If you're still having issues:

1. **Check Python version**: `python --version` (should be 3.11 or 3.12)
2. **Check pip version**: `pip --version`
3. **Check virtual environment**: `which python` (should show venv path)
4. **Review error messages**: Copy full error output for debugging
5. **Try minimal test**: `pytest tests/test_pipeline.py -v -s`

## Summary Checklist

- [ ] Python 3.11 or 3.12 installed
- [ ] Virtual environment created and activated
- [ ] pip upgraded to latest version
- [ ] Dependencies installed successfully
- [ ] pytest works: `pytest --version`
- [ ] E2E tests run: `pytest -m integration -v`

