"""Orchestration loop for running the trading system."""
import time
import signal
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import logging

from config.settings import AppConfig
from agents.data_agent import DataAgent
from agents.execution_agent import ExecutionAgent
from agents.strategy_agent import StrategyAgent
from agents.quant_agent import QuantAgent
from agents.risk_agent import RiskAgent
from utils.exceptions import TradingSystemError

logger = logging.getLogger(__name__)


class TradingSystemOrchestrator:
    """
    Orchestrates the trading system agents in a continuous loop.
    
    Flow:
    1. DataAgent fetches market data
    2. Strategy Agent evaluates data and generates signals ✅
    3. Quant Agent validates signals and adjusts confidence ✅
    4. Risk Agent validates trades and calculates position sizing ✅
    5. Execution Agent executes approved trades (future)
    6. Audit Agent logs and reports (future)
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize the orchestrator.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.running = False
        self.iteration = 0
        
        # Initialize agents
        logger.info("Initializing agents...")
        try:
            self.data_agent = DataAgent(config=config)
            self.strategy_agent = StrategyAgent(config=config)
            self.quant_agent = QuantAgent(config=config)
            self.risk_agent = RiskAgent(config=config)
            self.execution_agent = ExecutionAgent(config=config)
            # TODO: Initialize other agents as they're implemented
            # self.audit_agent = AuditAgent(config=config)
            
            logger.info("All agents initialized successfully")
        except Exception as e:
            logger.exception("Failed to initialize agents")
            raise TradingSystemError(
                f"Failed to initialize orchestrator: {str(e)}"
            ) from e
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.stop()
    
    def start(self) -> None:
        """Start the orchestration loop."""
        logger.info("=" * 60)
        logger.info("Trading System Orchestrator Starting")
        logger.info(f"Mode: {self.config.trading_mode.value}")
        logger.info(f"Symbols: {', '.join(self.config.symbols)}")
        logger.info(f"Loop interval: {self.config.loop_interval_seconds} seconds")
        logger.info("=" * 60)
        
        # Health check all agents
        if not self._health_check_all():
            logger.error("Health checks failed, exiting")
            sys.exit(1)
        
        self.running = True
        logger.info("Orchestrator started, entering main loop...")
        
        try:
            while self.running:
                self._run_iteration()
                
                if self.running:  # Check again in case stop() was called
                    time.sleep(self.config.loop_interval_seconds)
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            self.stop()
        except Exception as e:
            logger.exception("Unexpected error in orchestrator loop")
            self.stop()
            raise
    
    def stop(self) -> None:
        """Stop the orchestration loop."""
        if not self.running:
            return
        
        logger.info("Stopping orchestrator...")
        self.running = False
        
        # Finalize any pending operations
        logger.info(f"Orchestrator stopped after {self.iteration} iterations")
    
    def _health_check_all(self) -> bool:
        """Perform health checks on all agents."""
        logger.info("Performing health checks...")
        
        checks = {
            "DataAgent": self.data_agent.health_check(),
            "StrategyAgent": self.strategy_agent.health_check(),
            "QuantAgent": self.quant_agent.health_check(),
            "RiskAgent": self.risk_agent.health_check(),
            "ExecutionAgent": self.execution_agent.health_check(),
        }
        
        all_healthy = True
        for agent_name, health in checks.items():
            status = health.get("status", "unknown")
            if status == "healthy":
                logger.info(f"✓ {agent_name}: {status}")
            else:
                logger.error(f"✗ {agent_name}: {status} - {health.get('error', 'Unknown error')}")
                all_healthy = False
        
        return all_healthy
    
    def _run_iteration(self) -> None:
        """Run a single iteration of the trading loop."""
        self.iteration += 1
        iteration_start = datetime.now()
        
        logger.info("-" * 60)
        logger.info(f"Iteration {self.iteration} started at {iteration_start.isoformat()}")
        logger.info("-" * 60)
        
        try:
            # Step 1: Fetch market data
            logger.info("Step 1: Fetching market data...")
            try:
                market_data = self.data_agent.process(
                    symbols=self.config.symbols,
                    timeframe="1Day",
                    limit=100
                )
            except Exception as e:
                logger.error(f"DataAgent failed: {e}", exc_info=True)
                logger.warning("Skipping iteration due to data fetch failure")
                return
            
            if not market_data:
                logger.warning("No market data received, skipping iteration")
                return
            
            logger.info(f"Fetched data for {len(market_data)} symbols")
            
            # Log summary of fetched data
            for symbol, data in market_data.items():
                if data.bars:
                    latest_bar = data.bars[-1]
                    logger.info(
                        f"  {symbol}: {len(data.bars)} bars, "
                        f"latest close=${latest_bar.close:.2f}, volume={latest_bar.volume:,}"
                    )
            
            # Step 2: Strategy Agent evaluates data and generates signals
            logger.info("Step 2: Evaluating market data and generating signals...")
            try:
                signals = self.strategy_agent.process(market_data)
            except Exception as e:
                logger.error(f"StrategyAgent failed: {e}", exc_info=True)
                logger.warning("Continuing iteration with empty signals due to StrategyAgent failure")
                signals = []
            
            if signals:
                logger.info(f"Generated {len(signals)} trading signals")
                for signal in signals:
                    logger.info(
                        f"  {signal.symbol}: {signal.action.value} using {signal.strategy_name} "
                        f"(confidence: {signal.confidence:.2f}, price: ${signal.price:.2f})"
                    )
            else:
                logger.info("No signals generated")
            
            # Step 3: Quant Agent validates signals and adjusts confidence
            if signals:
                logger.info("Step 3: Validating signals with quantitative analysis...")
                try:
                    validated_signals = self.quant_agent.process(signals, market_data)
                    
                    # Log confidence adjustments
                    for original, validated in zip(signals, validated_signals):
                        if original.confidence != validated.confidence:
                            logger.info(
                                f"  {validated.symbol}: Confidence adjusted "
                                f"{original.confidence:.2f} → {validated.confidence:.2f}"
                            )
                    
                    signals = validated_signals
                    logger.info(f"Quantitative validation completed: {len(signals)} signals validated")
                except Exception as e:
                    logger.error(f"QuantAgent failed: {e}", exc_info=True)
                    logger.warning("Continuing with unvalidated signals due to QuantAgent failure")
                    # Continue with original signals
            
            # Step 4: Risk Agent validates trades and calculates position sizing
            if signals:
                logger.info("Step 4: Risk validation and position sizing...")
                try:
                    approved_signals = self.risk_agent.process(signals, execution_agent=self.execution_agent)
                    
                    # Log approved vs rejected
                    approved = [s for s in approved_signals if s.approved]
                    rejected = [s for s in approved_signals if not s.approved]
                    
                    if approved:
                        logger.info(f"Approved {len(approved)} signals with position sizing:")
                        for signal in approved:
                            logger.info(
                                f"  {signal.symbol}: {signal.action.value} {signal.qty} shares, "
                                f"risk=${signal.risk_amount:.2f}, stop_loss=${signal.stop_loss:.2f if signal.stop_loss else 'N/A'}"
                            )
                    
                    if rejected:
                        logger.warning(f"Rejected {len(rejected)} signals due to risk violations")
                        for signal in rejected:
                            logger.warning(f"  {signal.symbol}: Risk check failed")
                    
                    # Filter to only approved signals
                    signals = [s for s in approved_signals if s.approved]
                    logger.info(f"Risk validation completed: {len(signals)} approved, {len(rejected)} rejected")
                except Exception as e:
                    logger.error(f"RiskAgent failed: {e}", exc_info=True)
                    logger.warning("Continuing with unvalidated signals due to RiskAgent failure")
                    # Continue with signals (but mark as not approved)
                    for signal in signals:
                        signal.approved = False
            
            # Step 5: Execution Agent executes approved trades
            if signals:
                logger.info("Step 5: Executing approved trades...")
                executed_count = 0
                for signal in signals:
                    if signal.action == SignalAction.HOLD:
                        continue  # Skip HOLD signals
                    
                    try:
                        order_request = {
                            "symbol": signal.symbol,
                            "quantity": signal.qty or 1,
                            "side": signal.action.value.lower(),
                            "order_type": "market"
                        }
                        
                        result = self.execution_agent.process(order_request)
                        executed_count += 1
                        logger.info(
                            f"Order executed for {signal.symbol}: {result.get('order_id', 'N/A')}"
                        )
                    except Exception as e:
                        logger.error(
                            f"ExecutionAgent failed for {signal.symbol}: {e}",
                            exc_info=True
                        )
                        # Continue with next trade
                
                if executed_count == 0:
                    logger.info("No trades executed (all signals were HOLD or execution failed)")
            else:
                logger.info("No approved signals to execute")
            
            # Step 6: Audit Agent would log and report
            # TODO: Implement Audit Agent
            # try:
            #     self.audit_agent.process(iteration_results)
            # except Exception as e:
            #     logger.error(f"AuditAgent failed: {e}", exc_info=True)
            
            iteration_duration = (datetime.now() - iteration_start).total_seconds()
            logger.info(f"Iteration {self.iteration} completed in {iteration_duration:.2f} seconds")
            
        except TradingSystemError as e:
            logger.error(
                f"Trading system error in iteration {self.iteration}: {e.message}",
                extra={"correlation_id": e.correlation_id}
            )
        except Exception as e:
            logger.exception(f"Unexpected error in iteration {self.iteration}")
            # Continue running - don't crash on single iteration failure

