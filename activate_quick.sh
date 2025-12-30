#!/bin/zsh
# Quick activation - SOURCE this file: source activate_quick.sh
cd /Users/ravshan/trading-system
source venv/bin/activate
echo "✅ Venv activated: $(python -c 'import sys; print(sys.executable)')"
