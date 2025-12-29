# Trading System

A multi-agent trading system with strategy, quantitative analysis, risk management, execution, and audit capabilities.

## Architecture

The system is built around five specialized agents following a deterministic-first approach with LLMs used only for interpretation and advisory roles:

1. **Strategy Agent**: Interprets market context and selects from predefined strategy templates (Groq)
2. **Quant Agent**: Performs deterministic quantitative analysis; LLM reviews results (Claude)
3. **Risk Agent**: Enforces hard-coded risk rules; LLM provides advisory feedback (OpenAI/Claude)
4. **Execution Agent**: Pure Python code for order execution - no LLM in execution path (Code-only)
5. **Audit Agent**: Generates reports and narratives (Claude)

See [ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md) for detailed feedback and improvement roadmap.

## Features

- ✅ Type-safe with comprehensive type hints
- ✅ Comprehensive error handling with retry logic
- ✅ Structured logging with correlation IDs
- ✅ Centralized configuration management
- ✅ Input validation
- ✅ Health checks
- ✅ Comprehensive test suite
- ✅ Base agent architecture for consistency

## Project Structure

```
trading-system/
├── agents/              # Agent implementations
│   ├── base.py         # Base agent class
│   ├── execution_agent.py
│   ├── strategy_agent.py
│   ├── quant_agent.py
│   ├── risk_agent.py
│   └── audit_agent.py
├── config/             # Configuration management
│   └── settings.py
├── models/             # Data models and validation
│   ├── enums.py
│   ├── validation.py
│   └── trade.py
├── utils/              # Utilities
│   ├── exceptions.py   # Custom exceptions
│   ├── logging.py      # Logging setup
│   ├── retry.py        # Retry utilities
│   └── database.py
├── tests/              # Test suite
│   ├── conftest.py     # Pytest fixtures
│   ├── test_execution_agent.py
│   └── test_validation.py
├── main.py             # Entry point
└── requirements.txt    # Dependencies
```

## Setup

### Prerequisites

- Python 3.11+
- Virtual environment (recommended)

### Installation

1. **Create and activate virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables:**
Create a `.env` file in the root directory:
```bash
# Required: Alpaca API credentials
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
ALPACA_PAPER=true  # Set to false for live trading

# Optional: Trading mode and logging
TRADING_MODE=paper  # or "live"
LOG_LEVEL=INFO      # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=text     # or "json" for structured logging

# Optional: LLM provider keys (for Strategy, Quant, Risk, Audit agents)
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GROQ_API_KEY=your_groq_key

# Optional: Database
DATABASE_URL=postgresql://user:password@localhost/trading_db
```

## Usage

### Running the System

```bash
python main.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov

# Run specific test file
pytest tests/test_execution_agent.py

# Run specific test
pytest tests/test_execution_agent.py::TestExecutionAgentOrders::test_process_successful_order

# Run only unit tests
pytest -m unit
```

### Example: Using Execution Agent

```python
from config.settings import get_config
from agents.execution_agent import ExecutionAgent

# Load config
config = get_config()

# Initialize agent
agent = ExecutionAgent(config=config)

# Check health
health = agent.health_check()
print(health)

# Get account info
account = agent.get_account()
print(f"Balance: ${account.cash}")

# Place an order
order_request = {
    "symbol": "AAPL",
    "quantity": 10,
    "side": "buy",
    "order_type": "market"
}

result = agent.process(order_request)
print(f"Order placed: {result['order_id']}")
```

## Testing

The project includes comprehensive tests with the following coverage:

- **Unit Tests**: Fast, isolated tests for individual components
- **Integration Tests**: Tests for component interactions
- **Fixtures**: Reusable test data and mocks in `conftest.py`

Test markers:
- `@pytest.mark.unit` - Unit tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.api` - Tests requiring API access
- `@pytest.mark.slow` - Slow-running tests

## Configuration

Configuration is managed centrally through `config/settings.py`:

- Environment-based configuration
- Type-safe configuration objects
- Validation of required settings
- Support for multiple environments (paper/live trading)

## Error Handling

The system uses a hierarchical exception structure:

- `TradingSystemError` - Base exception
- `AgentError` - Agent-related errors
- `ExecutionError` - Trade execution failures
- `ValidationError` - Input validation failures
- `APIError` - External API failures

All errors include correlation IDs for request tracing.

## Logging

Structured logging with correlation IDs enables:

- Request tracing across agents
- Debugging production issues
- Performance monitoring
- Audit trails

Logs can be output as:
- **Text format**: Human-readable format (default)
- **JSON format**: Structured JSON for log aggregation tools

Set `LOG_FORMAT=json` in `.env` for JSON output.

## Development Guidelines

### Adding a New Agent

1. Inherit from `BaseAgent`
2. Implement the `process()` method
3. Add comprehensive error handling
4. Include type hints
5. Write unit tests
6. Document with docstrings

Example:
```python
from agents.base import BaseAgent

class MyAgent(BaseAgent):
    def process(self, *args, **kwargs):
        """Process method implementation."""
        self.log_info("Processing...")
        # Your logic here
        return result
```

### Code Quality

- Use type hints for all function parameters and returns
- Include docstrings for all classes and methods
- Follow PEP 8 style guidelines
- Run `pytest` before committing
- Use meaningful variable names

## Security Notes

- ⚠️ Never commit `.env` files (already in `.gitignore`)
- ⚠️ Use paper trading for development and testing
- ⚠️ Validate all inputs before processing
- ⚠️ Use environment variables for all secrets
- ⚠️ Review code before switching to live trading

## Roadmap

See [ARCHITECTURE_REVIEW.md](ARCHITECTURE_REVIEW.md) for:
- Current implementation status
- Identified issues and improvements
- Phased improvement plan
- Scalability considerations

## License

[Your License Here]

## Contributing

[Contributing Guidelines]

