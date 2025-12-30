# Configuration Setup Guide

## ✅ System Status

**Great news!** Your trading system is now fully functional:
- ✅ All agents initialize successfully
- ✅ All LLM clients are compatible
- ✅ System starts and runs properly

The health check failures you see are due to **missing or invalid API keys**, not code issues.

## 🔧 Required Configuration

### 1. API Keys Setup

Create a `.env` file in the project root:

```bash
cd /Users/ravshan/trading-system
cp .env.example .env  # If you have an example file
# Or create .env manually
```

### 2. Required Environment Variables

#### Alpaca API (Required)
```bash
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
ALPACA_PAPER=true  # Use paper trading (recommended for testing)
# Note: Don't set ALPACA_BASE_URL - let the library handle it automatically
```

**Get Alpaca API Keys:**
1. Sign up at https://alpaca.markets/
2. Go to Paper Trading dashboard
3. Generate API keys

#### Groq API (Required for StrategyAgent)
```bash
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=mixtral-8x7b-32768  # Optional, has default
```

**Get Groq API Key:**
1. Sign up at https://console.groq.com/
2. Create an API key

#### Anthropic API (Required for AuditAgent)
```bash
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022  # Optional, updated default
```

**Get Anthropic API Key:**
1. Sign up at https://console.anthropic.com/
2. Create an API key

#### OpenAI API (Optional - only if using OpenAI for RiskAgent LLM advisor)
```bash
OPENAI_API_KEY=your_openai_api_key_here  # Optional
OPENAI_MODEL=gpt-4  # Optional
```

### 3. Optional Configuration

#### Symbols to Trade
```bash
SYMBOLS=AAPL,MSFT,GOOGL  # Comma-separated list
```

#### Trading Mode
```bash
TRADING_MODE=paper  # or "live" (be careful!)
```

#### Logging
```bash
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR
```

#### Data Provider
```bash
# Use Alpaca for data (requires Alpaca API keys)
DATA_PROVIDER=alpaca

# OR use Yahoo Finance (free, no API key needed)
DATA_PROVIDER=yahoo

# OR use Polygon (requires POLYGON_API_KEY)
DATA_PROVIDER=polygon
POLYGON_API_KEY=your_polygon_key
```

## 📝 Example .env File

```bash
# Trading Configuration
TRADING_MODE=paper
LOG_LEVEL=INFO
SYMBOLS=AAPL,MSFT,GOOGL,SPY,QQQ

# Alpaca (Required)
ALPACA_API_KEY=PKXXXXXXXXXXXXXXXXXXXXXXXX
ALPACA_SECRET_KEY=XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
ALPACA_PAPER=true

# Groq (Required for StrategyAgent)
GROQ_API_KEY=gsk_XXXXXXXXXXXXXXXXXXXXXXXX
GROQ_MODEL=mixtral-8x7b-32768

# Anthropic (Required for AuditAgent)
ANTHROPIC_API_KEY=sk-ant-XXXXXXXXXXXXXXXXXXXXXXXX
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Data Provider
DATA_PROVIDER=alpaca  # or yahoo for free data

# Optional: OpenAI (if using for RiskAgent advisor)
# OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXXXX
```

## 🐛 Fixed Issues

### Claude Model Name
- **Fixed**: Updated default model from `claude-3-opus-20240229` (deprecated) to `claude-3-5-sonnet-20241022` (current)
- **Note**: You can override this with `ANTHROPIC_MODEL` environment variable

### Alpaca URL Issue
- **Issue**: URL showing `/v2/v2/account` (double `/v2/`)
- **Solution**: Don't set `ALPACA_BASE_URL` - let the library handle URLs automatically
- **If you must set it**, use: `ALPACA_BASE_URL=https://paper-api.alpaca.markets` (no trailing `/v2`)

## 🚀 Running the System

Once you have API keys configured:

```bash
# Activate venv
source venv/bin/activate

# Run the system
python main.py
```

## ✅ Health Check Status

After configuring API keys, you should see:
```
✓ DataAgent: healthy
✓ StrategyAgent: healthy
✓ QuantAgent: healthy
✓ RiskAgent: healthy
✓ ExecutionAgent: healthy
✓ AuditAgent: healthy
```

## 🧪 Testing Without Real API Keys

For testing, you can use the E2E tests (they use mocks):

```bash
source venv/bin/activate
pytest -m integration -v
```

## 📚 Additional Resources

- **Alpaca Documentation**: https://alpaca.markets/docs/
- **Groq Documentation**: https://console.groq.com/docs
- **Anthropic Documentation**: https://docs.anthropic.com/
- **Project README**: See `README.md`

## ⚠️ Important Notes

1. **Paper Trading First**: Always test with `ALPACA_PAPER=true` before going live
2. **API Key Security**: Never commit `.env` file to git (it's in `.gitignore`)
3. **Rate Limits**: Be aware of API rate limits for each provider
4. **Costs**: LLM API calls have costs - monitor usage
5. **Model Updates**: Model names change over time - check provider docs for latest

## 🆘 Troubleshooting

### Health Checks Fail
- Check that API keys are correctly set in `.env`
- Verify API keys are valid and active
- Check API provider status pages

### Alpaca 404 Errors
- Ensure `ALPACA_BASE_URL` is not set (or set correctly)
- Verify you're using paper trading keys with `ALPACA_PAPER=true`

### Claude Model Not Found
- Use `claude-3-5-sonnet-20241022` or check Anthropic docs for current models
- Override with `ANTHROPIC_MODEL` env var

### Groq 401 Errors
- Verify `GROQ_API_KEY` is correct
- Check that the API key has proper permissions

