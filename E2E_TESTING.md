# End-to-End Testing Guide

This document describes how to run end-to-end (E2E) integration tests for the complete trading system pipeline.

## Overview

E2E tests verify that the entire pipeline works correctly from data fetching through audit reporting:
1. **DataAgent** → Fetches market data
2. **StrategyAgent** → Generates trading signals
3. **QuantAgent** → Validates signals quantitatively
4. **RiskAgent** → Enforces risk rules and sizes positions
5. **ExecutionAgent** → Executes approved trades
6. **AuditAgent** → Generates audit reports

## Prerequisites

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Activate virtual environment:**
   ```bash
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Create test environment file** (optional):
   ```bash
   cp .env.example .env.test
   # Edit .env.test with test API keys (mock keys are fine for most tests)
   ```

## Test Structure

### Test Markers

Tests are categorized using pytest markers:
- `@pytest.mark.unit` - Fast unit tests (isolated)
- `@pytest.mark.integration` - Integration/E2E tests
- `@pytest.mark.slow` - Slow-running tests
- `@pytest.mark.api` - Tests requiring API access
- `@pytest.mark.paper_trading` - Paper trading tests

### Test Files

- `tests/test_pipeline.py` - E2E pipeline tests
- `tests/conftest.py` - Shared fixtures and configuration

## Running Tests

### Run All Integration Tests

```bash
pytest -m integration
```

### Run Specific E2E Test

```bash
pytest tests/test_pipeline.py::TestPipelineEndToEnd::test_full_pipeline_with_mocks -v
```

### Run with Coverage

```bash
pytest -m integration --cov=. --cov-report=html
```

### Run with Debugging

```bash
pytest -m integration --pdb  # Drop into debugger on failure
pytest -m integration -v -s  # Verbose output with print statements
```

## Test Configuration

### Environment Variables for Testing

Create a `.env.test` file with these settings:

```bash
# Trading mode
TRADING_MODE=paper
LOG_LEVEL=DEBUG

# Alpaca (paper trading mode, mock keys)
ALPACA_API_KEY=test_api_key
ALPACA_SECRET_KEY=test_secret_key
ALPACA_PAPER=true

# Groq (mock key)
GROQ_API_KEY=test_groq_key

# Anthropic (mock key)
ANTHROPIC_API_KEY=test_anthropic_key

# Symbols for testing
SYMBOLS=SPY,QQQ,AAPL

# Agent configurations
QUANT_MIN_SHARPE=1.5
RISK_MAX_RISK_PER_TRADE=0.02
```

### Mocking Strategy

All external API calls are mocked in E2E tests:
- **DataAgent**: Mocked `yfinance` or Alpaca data client
- **StrategyAgent**: Mocked Groq LLM responses
- **ExecutionAgent**: Mocked Alpaca TradingClient
- **AuditAgent**: Mocked Anthropic Claude client

This ensures:
- Tests run fast (no network calls)
- Tests are reliable (no API rate limits)
- Tests are free (no API costs)
- Tests are deterministic (consistent results)

## Test Fixtures

### Available Fixtures

1. **`mock_config`** - Basic test configuration
2. **`test_config_with_llms`** - Full config with all LLM providers
3. **`mock_market_data`** - Sample market data for multiple symbols
4. **`mock_groq_client`** - Mocked Groq client
5. **`mock_anthropic_client`** - Mocked Anthropic client
6. **`sample_trading_signal`** - Sample trading signal

### Using Fixtures

```python
def test_my_feature(mock_config, mock_market_data):
    # Use fixtures in your test
    data_agent = DataAgent(config=mock_config)
    result = data_agent.process(symbols=["SPY"])
```

## E2E Test Examples

### Basic Pipeline Test

```python
@pytest.mark.integration
def test_full_pipeline_with_mocks(
    test_config_with_llms,
    mock_market_data,
    mock_groq_client,
    mock_anthropic_client
):
    """Test the complete pipeline flow."""
    report = run_pipeline(
        symbols=["SPY", "QQQ"],
        config=test_config_with_llms,
        market_data=mock_market_data
    )
    
    assert isinstance(report, AuditReport)
    assert len(report.summary) > 0
```

### Test with Specific Mocks

```python
@patch('agents.strategy_agent.Groq')
def test_pipeline_with_custom_strategy(mock_groq):
    # Setup custom mock behavior
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = ...
    
    # Run test
    ...
```

## Pipeline Function

The `run_pipeline()` function in `tests/test_pipeline.py` executes the complete pipeline:

```python
def run_pipeline(
    symbols: List[str],
    config,
    market_data: Dict[str, MarketData] = None
) -> AuditReport:
    """
    Run the complete trading pipeline end-to-end.
    
    Returns:
        AuditReport from the AuditAgent
    """
    # Step 1: DataAgent
    # Step 2: StrategyAgent
    # Step 3: QuantAgent
    # Step 4: RiskAgent
    # Step 5: ExecutionAgent
    # Step 6: AuditAgent
```

## Assertions

E2E tests verify:

1. **Pipeline Completes**: All agents execute without crashing
2. **Data Flow**: Data flows correctly between agents
3. **Metrics Accuracy**: Calculated metrics are correct
4. **Report Generation**: Audit reports are generated
5. **Error Handling**: Failures are handled gracefully

### Example Assertions

```python
# Verify report structure
assert isinstance(report, AuditReport)
assert report.report_type == "iteration"

# Verify metrics
assert "signals_generated" in report.metrics
assert report.metrics["signals_generated"] >= 0

# Verify relationships
assert report.metrics["signals_approved"] <= report.metrics["signals_generated"]
assert 0.0 <= report.metrics["approval_rate"] <= 1.0
```

## Troubleshooting

### Tests Failing

1. **Check imports**: Ensure all dependencies are installed
2. **Check mocks**: Verify mock objects are properly configured
3. **Check logs**: Use `-v -s` flags to see detailed output
4. **Check environment**: Ensure test environment variables are set

### Common Issues

1. **Import Errors**: Run `pip install -r requirements.txt`
2. **Mock Not Working**: Check patch paths match actual import paths
3. **Missing Fixtures**: Ensure `conftest.py` is in the tests directory

### Debug Mode

```bash
# Drop into debugger on failure
pytest -m integration --pdb

# Show print statements
pytest -m integration -s

# Show local variables on failure
pytest -m integration -l
```

## Continuous Integration

### GitHub Actions Example

```yaml
name: E2E Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt
      - run: pytest -m integration --cov=. --cov-report=xml
```

## Best Practices

1. **Mock External Services**: Always mock API calls in E2E tests
2. **Use Fixtures**: Reuse fixtures from `conftest.py`
3. **Test Edge Cases**: Include tests for failures and edge cases
4. **Keep Tests Fast**: E2E tests should complete in seconds
5. **Clear Assertions**: Use descriptive assertion messages
6. **Isolate Tests**: Each test should be independent

## Next Steps

1. **Add More Test Cases**: Cover more scenarios
2. **Performance Tests**: Add timing benchmarks
3. **Load Tests**: Test with large datasets
4. **Integration with CI/CD**: Automate test runs

## References

- [pytest Documentation](https://docs.pytest.org/)
- [pytest-mock Documentation](https://pytest-mock.readthedocs.io/)
- [Project README](README.md)

