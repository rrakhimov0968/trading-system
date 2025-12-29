"""Main entry point for the trading system."""
import sys
from utils.logging import setup_logging
from config.settings import get_config
from agents.execution_agent import ExecutionAgent
from utils.exceptions import TradingSystemError

# Setup logging first (before other imports that may log)
setup_logging()

# Import logging after setup
import logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for the trading system."""
    try:
        logger.info("Trading System Starting...")
        
        # Load configuration
        config = get_config()
        logger.info(
            f"Configuration loaded: mode={config.trading_mode.value}, "
            f"log_level={config.log_level.value}"
        )
        
        # Initialize agents
        execution_agent = ExecutionAgent(config=config)
        
        # Health check
        health = execution_agent.health_check()
        logger.info(f"ExecutionAgent health: {health['status']}")
        
        if health["status"] != "healthy":
            logger.error(f"ExecutionAgent is unhealthy: {health}")
            sys.exit(1)
        
        # Get account info (example operation)
        account = execution_agent.get_account()
        logger.info(
            f"Account Balance: ${account.cash}, "
            f"Buying Power: ${account.buying_power}"
        )
        
        logger.info("Trading System Ready")
        
    except TradingSystemError as e:
        logger.error(
            f"Trading system error: {e.message}",
            extra={"correlation_id": e.correlation_id, "details": e.details}
        )
        sys.exit(1)
    except Exception as e:
        logger.exception("Unexpected error starting trading system")
        sys.exit(1)


if __name__ == "__main__":
    main()