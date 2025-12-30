"""Main entry point for the trading system."""
import sys
import asyncio
import os
from utils.logging import setup_logging
from config.settings import get_config
from core.orchestrator import TradingSystemOrchestrator
from core.async_orchestrator import AsyncTradingSystemOrchestrator
from utils.exceptions import TradingSystemError

# Setup logging first (before other imports that may log)
setup_logging()

# Import logging after setup
import logging

logger = logging.getLogger(__name__)


async def run_async_pipeline(symbols: list[str] = None):
    """
    Run a single async iteration of the trading pipeline.
    
    This is a convenience function for testing or running single iterations.
    
    Args:
        symbols: Optional list of symbols to process
    
    Returns:
        Dictionary with iteration results
    """
    config = get_config()
    symbols = symbols or config.symbols
    
    orchestrator = AsyncTradingSystemOrchestrator(config=config)
    result = await orchestrator.run_async_pipeline(symbols)
    
    # Cleanup
    await orchestrator.stop()
    
    return result


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
        
        # Check if async mode is enabled (default to true now)
        use_async = os.getenv("USE_ASYNC_ORCHESTRATOR", "true").lower() == "true"
        
        if use_async:
            logger.info("Using async event-driven orchestrator (default)")
            orchestrator = AsyncTradingSystemOrchestrator(config=config)
            asyncio.run(orchestrator.start())
        else:
            logger.info("Using synchronous orchestrator (USE_ASYNC_ORCHESTRATOR=false)")
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