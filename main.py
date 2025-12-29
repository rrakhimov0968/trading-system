"""Main entry point for the trading system."""
import sys
from utils.logging import setup_logging
from config.settings import get_config
from core.orchestrator import TradingSystemOrchestrator
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
            f"log_level={config.log_level.value}, "
            f"data_provider={config.data_provider.provider.value if config.data_provider else 'None'}"
        )
        
        # Create and start orchestrator
        orchestrator = TradingSystemOrchestrator(config=config)
        orchestrator.start()
        
    except TradingSystemError as e:
        logger.error(
            f"Trading system error: {e.message}",
            extra={"correlation_id": e.correlation_id, "details": e.details}
        )
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
        sys.exit(0)
    except Exception as e:
        logger.exception("Unexpected error starting trading system")
        sys.exit(1)


if __name__ == "__main__":
    main()