#!/bin/bash

# Setup and Test Script for Trading System E2E Tests
# This script helps set up the environment and run tests

set -e  # Exit on error

echo "🚀 Trading System - Setup and Test Script"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check Python version
echo "📋 Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Found Python: $PYTHON_VERSION"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found. Creating one...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate venv
echo ""
echo "🔧 Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo ""
echo "⬆️  Upgrading pip..."
pip install --upgrade pip setuptools wheel --quiet
echo -e "${GREEN}✓ pip upgraded${NC}"

# Install dependencies
echo ""
echo "📦 Installing dependencies..."
echo "   This may take a few minutes..."

# Install in stages for better error reporting
if pip install pandas==2.1.4 numpy python-dotenv==1.0.0 --quiet; then
    echo -e "${GREEN}✓ Core dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install core dependencies${NC}"
    exit 1
fi

if pip install alpaca-py==0.21.0 anthropic==0.18.1 openai==1.12.0 groq==0.4.2 --quiet; then
    echo -e "${GREEN}✓ API dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install API dependencies${NC}"
    exit 1
fi

if pip install yfinance==0.2.36 scipy statsmodels --quiet; then
    echo -e "${GREEN}✓ Data analysis dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install data analysis dependencies${NC}"
    exit 1
fi

# Try pandas-ta with error handling
if pip install pandas-ta --no-cache-dir --quiet 2>/dev/null; then
    echo -e "${GREEN}✓ pandas-ta installed${NC}"
else
    echo -e "${YELLOW}⚠️  pandas-ta installation had issues, trying alternative...${NC}"
    pip install pandas-ta --quiet || echo -e "${YELLOW}⚠️  pandas-ta may have issues, continuing anyway...${NC}"
fi

if pip install requests==2.31.0 aiohttp==3.9.1 python-dateutil==2.8.2 pytz==2023.3 pydantic==2.5.0 --quiet; then
    echo -e "${GREEN}✓ Utility dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install utility dependencies${NC}"
    exit 1
fi

if pip install pytest==7.4.4 pytest-asyncio==0.23.3 pytest-cov==4.1.0 pytest-mock==3.12.0 requests-mock==1.11.1 --quiet; then
    echo -e "${GREEN}✓ Testing dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install testing dependencies${NC}"
    exit 1
fi

# Verify installation
echo ""
echo "✅ Verifying installation..."
python3 -c "import pandas, numpy, pytest, yfinance, pydantic; print('   All core packages imported successfully!')" 2>/dev/null || {
    echo -e "${RED}✗ Package verification failed${NC}"
    exit 1
}

# Check if pytest can collect tests
echo ""
echo "🧪 Checking test collection..."
if pytest --collect-only -m integration --quiet > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Tests can be collected${NC}"
else
    echo -e "${YELLOW}⚠️  Could not collect integration tests (may need to check test files)${NC}"
fi

# Run tests if requested
if [ "$1" == "--test" ]; then
    echo ""
    echo "🏃 Running E2E tests..."
    echo ""
    pytest -m integration -v
else
    echo ""
    echo -e "${GREEN}✅ Setup complete!${NC}"
    echo ""
    echo "To run E2E tests, use:"
    echo "  source venv/bin/activate"
    echo "  pytest -m integration -v"
    echo ""
    echo "Or run this script with --test flag:"
    echo "  ./setup_and_test.sh --test"
fi

