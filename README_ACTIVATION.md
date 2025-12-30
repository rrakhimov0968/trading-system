# How to Activate Virtual Environment

## The Problem
When you run `./activate_venv.sh`, it runs in a subshell and the activation doesn't persist. You need to **SOURCE** it instead.

## ✅ Correct Way (Source the script)

```bash
cd /Users/ravshan/trading-system

# Method 1: Source the helper script
source activate_venv.sh

# Method 2: Source venv directly (simpler)
source venv/bin/activate
```

## ❌ Wrong Way (Executing the script)

```bash
./activate_venv.sh  # ❌ This won't work - changes are lost
```

## Verification

After activation, check if it worked:

```bash
# Check 1: VIRTUAL_ENV should be set
echo $VIRTUAL_ENV
# Should show: /Users/ravshan/trading-system/venv

# Check 2: Python executable location (MOST RELIABLE)
python -c "import sys; print(sys.executable)"
# Should show: /Users/ravshan/trading-system/venv/bin/python

# Check 3: Python version
python --version
# Should show: Python 3.11.14
```

**Note:** `which python` might still show pyenv shims, but that's okay if `sys.executable` shows the venv path. The venv Python will be used.

## Quick Commands

```bash
# Activate
cd /Users/ravshan/trading-system && source venv/bin/activate

# Deactivate
deactivate

# Or use the helper (remember to SOURCE it)
source activate_venv.sh
```

## Alternative: Use venv Python Directly (No Activation Needed)

You can skip activation entirely and use the venv Python directly:

```bash
# Run Python
venv/bin/python your_script.py

# Run pytest
venv/bin/pytest -m integration

# Install packages
venv/bin/pip install package_name
```

## Add to ~/.zshrc (Optional)

Add this alias to your `~/.zshrc` for convenience:

```bash
alias venv-trading='cd /Users/ravshan/trading-system && source venv/bin/activate'
```

Then just run:
```bash
venv-trading
```

