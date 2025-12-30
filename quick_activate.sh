#!/bin/zsh
# Quick activation script for trading-system venv

cd /Users/ravshan/trading-system

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Error: venv directory not found!"
    echo "Creating new venv..."
    python3 -m venv venv
fi

# Check if activate script exists
if [ ! -f "venv/bin/activate" ]; then
    echo "❌ Error: activate script not found!"
    exit 1
fi

# Activate venv
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Verify activation
if [ "$VIRTUAL_ENV" != "" ]; then
    echo "✅ Virtual environment activated!"
    echo "   Python: $(which python)"
    echo "   Version: $(python --version)"
    echo "   Venv: $VIRTUAL_ENV"
else
    echo "⚠️  Warning: Activation may not have worked"
    echo "   Checking Python location..."
    which python
fi

