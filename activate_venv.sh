#!/bin/zsh
# Proper venv activation that works with pyenv
# IMPORTANT: You must SOURCE this script, not execute it!
# Usage: source activate_venv.sh
#        OR: . activate_venv.sh

cd /Users/ravshan/trading-system

# Source the venv activate script
source venv/bin/activate

# Verify and fix PATH if pyenv is interfering
if [[ "$(command -v python 2>/dev/null)" == *"pyenv"* ]] || [[ "$(which python 2>/dev/null)" == *"pyenv"* ]]; then
    echo "⚠️  pyenv shims detected, fixing PATH..."
    # Ensure venv/bin is first in PATH (before pyenv shims)
    export PATH="/Users/ravshan/trading-system/venv/bin:$(echo $PATH | tr ':' '\n' | grep -v "/Users/ravshan/trading-system/venv/bin" | tr '\n' ':' | sed 's/:$//')"
    
    # Verify the fix worked
    if [[ "$(command -v python 2>/dev/null)" == *"venv"* ]]; then
        echo "✅ PATH fixed - venv Python is now first"
    fi
fi

# Verify activation worked
if [[ -n "$VIRTUAL_ENV" ]]; then
    echo "✅ Virtual environment activated!"
    echo "   Python: $(command -v python)"
    echo "   Version: $(python --version 2>&1)"
    echo "   Location: $VIRTUAL_ENV"
    echo ""
    echo "To verify: run 'python -c \"import sys; print(sys.executable)\"'"
else
    echo "❌ Activation failed"
    return 1 2>/dev/null || exit 1
fi

