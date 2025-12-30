# Virtual Environment Activation Troubleshooting Guide

## Quick Fixes

### 1. Verify You're in the Correct Directory
```bash
cd /Users/ravshan/trading-system
pwd  # Should show: /Users/ravshan/trading-system
```

### 2. Try Different Activation Methods

**For zsh (your shell):**
```bash
# Method 1: Using source (should work)
source venv/bin/activate

# Method 2: Using dot notation (equivalent to source)
. venv/bin/activate

# Method 3: Full path
source /Users/ravshan/trading-system/venv/bin/activate
```

### 3. Check if Activation Worked
After running the activation command, you should see `(venv)` at the start of your prompt:
```bash
(venv) ravshan@macbook trading-system %
```

Also verify:
```bash
which python  # Should point to venv/bin/python
python --version  # Should show Python version
```

### 4. If Activation Seems to Work But Doesn't

**Check if you're in a subprocess:**
- Some IDEs and terminal multiplexers create subshells
- Try running activation in a fresh terminal window

**Check shell configuration:**
```bash
# See if there are any shell config issues
echo $SHELL  # Should show /bin/zsh

# Check if zsh config is interfering
cat ~/.zshrc | grep -i venv  # Look for any venv-related config
```

### 5. Recreate Virtual Environment (If Corrupted)

If the venv is corrupted or created with wrong Python version:

```bash
# Remove old venv
rm -rf venv

# Create new venv with specific Python version
python3 -m venv venv

# OR if you have multiple Python versions
python3.11 -m venv venv  # Use Python 3.11 specifically
python3.12 -m venv venv  # Use Python 3.12 specifically

# Activate
source venv/bin/activate

# Verify
which python
python --version

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 6. Common Issues and Solutions

#### Issue: "No such file or directory"
**Solution:** Venv doesn't exist. Recreate it:
```bash
python3 -m venv venv
```

#### Issue: "Permission denied"
**Solution:** Fix permissions:
```bash
chmod +x venv/bin/activate
```

#### Issue: "Bad interpreter"
**Solution:** Venv was created with different Python. Recreate:
```bash
rm -rf venv
python3 -m venv venv
```

#### Issue: Activation runs but prompt doesn't change
**Possible causes:**
- You're in a subprocess
- Shell config is overriding PS1
- Terminal doesn't support prompt changes

**Solution:** Manually verify:
```bash
source venv/bin/activate
which python  # Should show venv path
python -c "import sys; print(sys.prefix)"  # Should show venv path
```

#### Issue: "Command not found: source"
**Solution:** You might be in sh/bash without source. Use:
```bash
. venv/bin/activate
```

### 7. Using Virtual Environment Without Activation

You can also use the venv Python directly without activating:
```bash
# Run Python directly
venv/bin/python --version

# Run scripts
venv/bin/python your_script.py

# Run pytest
venv/bin/pytest -m integration

# Install packages
venv/bin/pip install package_name
```

### 8. Verify Virtual Environment is Working

After activation (or using venv/bin/python):
```bash
# Check Python path
python -c "import sys; print(sys.executable)"
# Should show: /Users/ravshan/trading-system/venv/bin/python

# Check installed packages
pip list | head -10

# Check if key packages are installed
python -c "import pandas, numpy, pytest; print('All packages OK')"
```

### 9. IDE/Editor Integration

**VS Code / Cursor:**
- Select Python interpreter: `Cmd+Shift+P` → "Python: Select Interpreter"
- Choose: `./venv/bin/python`

**PyCharm:**
- Settings → Project → Python Interpreter
- Add → Existing Environment → Select `venv/bin/python`

**Terminal in IDE:**
- Some IDEs need you to activate venv in their integrated terminal
- Or configure IDE to use venv Python automatically

### 10. Check for Conflicting Python Environments

```bash
# Check if pyenv is interfering
which python
pyenv which python  # If pyenv is active

# Check if conda is active
conda env list  # If conda is installed

# Check PATH
echo $PATH | tr ':' '\n' | grep -i python
```

### 11. Create an Activation Script (Alternative)

Create a simple activation helper:
```bash
cat > activate_venv.sh << 'EOF'
#!/bin/bash
cd /Users/ravshan/trading-system
source venv/bin/activate
echo "Virtual environment activated!"
echo "Python: $(which python)"
echo "Version: $(python --version)"
EOF

chmod +x activate_venv.sh

# Use it
./activate_venv.sh
```

### 12. Debug Activation Script

If activation isn't working, check what the script does:
```bash
# See what activate script sets
head -20 venv/bin/activate

# Manually set variables (for testing)
export VIRTUAL_ENV="/Users/ravshan/trading-system/venv"
export PATH="$VIRTUAL_ENV/bin:$PATH"
unset PYTHON_HOME

# Test
which python  # Should show venv path
```

## Quick Diagnostic Commands

Run these to diagnose:
```bash
# 1. Check venv exists
test -d venv && echo "✓ venv directory exists" || echo "✗ venv missing"

# 2. Check activate script exists
test -f venv/bin/activate && echo "✓ activate script exists" || echo "✗ activate missing"

# 3. Check Python in venv
test -f venv/bin/python && echo "✓ venv Python exists" || echo "✗ venv Python missing"

# 4. Check Python version in venv
venv/bin/python --version 2>&1 && echo "✓ venv Python works" || echo "✗ venv Python broken"

# 5. Check current Python
which python
python --version

# 6. After activation, check
source venv/bin/activate 2>&1
which python
echo $VIRTUAL_ENV
```

## Most Likely Solutions

1. **If you see no error but prompt doesn't change:** Venv is working, just manually verify with `which python`
2. **If you get "command not found":** Use `. venv/bin/activate` instead of `source`
3. **If you get permission errors:** Recreate venv with `rm -rf venv && python3 -m venv venv`
4. **If Python version is wrong:** Recreate venv with specific Python: `python3.11 -m venv venv`

## Still Having Issues?

1. Check the exact error message you're seeing
2. Run the diagnostic commands above
3. Try recreating the venv
4. Check if you have multiple Python installations conflicting

## Alternative: Use the venv Python Directly

You can always use the venv Python without activation:
```bash
# Instead of: python script.py
venv/bin/python script.py

# Instead of: pytest
venv/bin/pytest

# Instead of: pip install
venv/bin/pip install
```

This works from any directory and doesn't require activation.

