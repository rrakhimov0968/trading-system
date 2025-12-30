#!/bin/zsh
# Fix for pyenv interfering with venv activation

cd /Users/ravshan/trading-system

echo "🔧 Fixing pyenv/venv conflict..."

# Activate venv
source venv/bin/activate

# Disable pyenv for this shell session
# This prevents pyenv shims from intercepting python commands
if command -v pyenv >/dev/null 2>&1; then
    # Unset pyenv functions
    unset -f pyenv
    unset -f pyenv-sh-activate
    unset -f pyenv-sh-deactivate
    
    # Remove pyenv shims from PATH temporarily
    # (venv/bin should already be first, but let's ensure it)
    export PATH=$(echo $PATH | tr ':' '\n' | grep -v pyenv | tr '\n' ':' | sed 's/:$//')
    export PATH="/Users/ravshan/trading-system/venv/bin:$PATH"
fi

echo "✅ Virtual environment activated (pyenv disabled)"
echo ""
echo "Verification:"
echo "  VIRTUAL_ENV: $VIRTUAL_ENV"
echo "  Python: $(which python)"
echo "  Version: $(python --version 2>&1)"
echo ""
echo "To deactivate: run 'deactivate'"

