"""Orchestration loop for running the trading system."""
import os
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
from agents.audit_agent import AuditAgent
from models.audit import IterationSummary, ExecutionResult
from models.signal import SignalAction
from utils.exceptions import TradingSystemError
from utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from core.market_hours import is_market_open, get_market_status_message

logger = logging.getLogger(__name__)


class TradingSystemOrchestrator:
    """
    Orchestrates the trading system agents in a continuous loop.
    
    Flow:
    1. DataAgent fetches market data ✅
    2. Strategy Agent evaluates data and generates signals ✅
    3. Quant Agent validates signals and adjusts confidence ✅
    4. Risk Agent validates trades and calculates position sizing ✅
    5. Execution Agent executes approved trades ✅
    6. Audit Agent logs and reports ✅
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
        
        # Memory management for signals history (prevent unbounded growth)
        self.all_signals: list = []
        self.MAX_SIGNALS_STORED = 1000  # Keep only last 1000 signals
        
        # Circuit breaker for system protection
        circuit_config = CircuitBreakerConfig(
            max_failures=int(os.getenv("CIRCUIT_BREAKER_MAX_FAILURES", "5")),
            timeout_seconds=int(os.getenv("CIRCUIT_BREAKER_TIMEOUT", "300")),
            data_quality_threshold=float(os.getenv("CIRCUIT_BREAKER_DATA_QUALITY_THRESHOLD", "0.8")),
            equity_drop_threshold=float(os.getenv("CIRCUIT_BREAKER_EQUITY_DROP_THRESHOLD", "0.10"))
        )
        self.circuit_breaker = CircuitBreaker(config=circuit_config)
        
        # Monitoring metrics
        self.monitoring_metrics = {
            "llm_success_count": 0,
            "llm_failure_count": 0,
            "total_iterations": 0,
            "successful_iterations": 0,
            "failed_iterations": 0,
            "recent_signals": [],
            "recent_equity_values": []
        }
        
        # Initialize agents
        logger.info("Initializing agents...")
        try:
            self.data_agent = DataAgent(config=config)
            self.strategy_agent = StrategyAgent(config=config)
            self.quant_agent = QuantAgent(config=config)
            self.risk_agent = RiskAgent(config=config)
            self.execution_agent = ExecutionAgent(config=config)
            self.audit_agent = AuditAgent(config=config)
            
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
                # Check market hours - if closed, sleep longer
                if not is_market_open():
                    status_msg = get_market_status_message()
                    logger.info(f"Market is closed. {status_msg}")
                    logger.info("Sleeping for 5 minutes until next check...")
                    # Sleep 5 minutes instead of 60 seconds when market is closed
                    time.sleep(300)
                    continue
                
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
            "AuditAgent": self.audit_agent.health_check(),
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
    
    def _cleanup_old_signals(self) -> None:
        """Prevent memory buildup by removing old signals."""
        if len(self.all_signals) > self.MAX_SIGNALS_STORED:
            # Remove oldest 10% or enough to get under limit
            remove_count = max(
                len(self.all_signals) - self.MAX_SIGNALS_STORED,
                len(self.all_signals) // 10
            )
            self.all_signals = self.all_signals[remove_count:]
            logger.debug(
                f"Cleaned up {remove_count} old signals "
                f"(kept {len(self.all_signals)}/{self.MAX_SIGNALS_STORED} max)"
            )
    
    def _run_iteration(self) -> None:
        """Run a single iteration of the trading loop."""
        # Check circuit breaker before starting iteration
        if self.circuit_breaker.is_open():
            logger.error(
                f"CIRCUIT BREAKER IS OPEN - Skipping iteration {self.iteration + 1}. "
                f"Reason: {self.circuit_breaker.get_metrics()}"
            )
            return
        
        self.iteration += 1
        self.monitoring_metrics["total_iterations"] += 1
        iteration_start = datetime.now()
        
        logger.info("-" * 60)
        logger.info(f"Iteration {self.iteration} started at {iteration_start.isoformat()}")
        logger.info("-" * 60)
        
        # Check if market is open before running full pipeline
        if not is_market_open():
            status_msg = get_market_status_message()
            logger.info(f"Market is closed. {status_msg}")
            logger.info("Skipping iteration until market opens")
            return
        
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
                self.monitoring_metrics["failed_iterations"] += 1
                # Record as data quality failure
                self.circuit_breaker.record_data_quality(0.0)
                return
            
            if not market_data:
                logger.warning("No market data received, skipping iteration")
                self.monitoring_metrics["failed_iterations"] += 1
                self.circuit_breaker.record_data_quality(0.0)
                return
            
            # Calculate data quality score (simple heuristic: % of symbols with data)
            expected_symbols = len(self.config.symbols)
            received_symbols = len(market_data)
            data_quality = received_symbols / expected_symbols if expected_symbols > 0 else 0.0
            self.circuit_breaker.record_data_quality(data_quality)
            
            logger.info(f"Fetched data for {len(market_data)} symbols")
            
            # Log summary of fetched data
            for symbol, data in market_data.items():
                if data.bars:
                    latest_bar = data.bars[-1]
                    logger.info(
                        f"  {symbol}: {len(data.bars)} bars, "
                        f"latest close=${latest_bar.close:.2f}, volume={latest_bar.volume:,}"
                    )
            
            # Initialize variables for iteration summary
            signals = []
            execution_results = []
            executed_count = 0
            
            # Cleanup old signals to prevent memory buildup
            self._cleanup_old_signals()
            
            # Step 2: Strategy Agent evaluates data and generates signals
            logger.info("Step 2: Evaluating market data and generating signals...")
            try:
                signals = self.strategy_agent.process(market_data)
                # Record LLM success (StrategyAgent uses LLM)
                self.circuit_breaker.record_llm_success()
                self.monitoring_metrics["llm_success_count"] += 1
            except Exception as e:
                logger.error(f"StrategyAgent failed: {e}", exc_info=True)
                logger.warning("Continuing iteration with empty signals due to StrategyAgent failure")
                signals = []
                # Record LLM failure
                self.circuit_breaker.record_llm_failure()
                self.monitoring_metrics["llm_failure_count"] += 1
                
                # Check circuit breaker after LLM failure
                if self.circuit_breaker.is_open():
                    logger.error("Circuit breaker opened due to LLM failures - stopping iteration")
                    return
            
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
            
            # Check account equity before execution (circuit breaker protection)
            try:
                account = self.execution_agent.get_account()
                if account and hasattr(account, 'equity'):
                    current_equity = float(account.equity)
                    self.circuit_breaker.check_equity_drop(current_equity)
                    # Track equity for monitoring
                    self.monitoring_metrics["recent_equity_values"].append({
                        "timestamp": datetime.now(),
                        "equity": current_equity
                    })
                    # Keep only last 100 equity values
                    if len(self.monitoring_metrics["recent_equity_values"]) > 100:
                        self.monitoring_metrics["recent_equity_values"] = \
                            self.monitoring_metrics["recent_equity_values"][-100:]
                    
                    if self.circuit_breaker.is_open():
                        logger.error("Circuit breaker opened due to equity drop - stopping execution")
                        signals = []  # Prevent execution
            except Exception as e:
                logger.warning(f"Failed to check account equity: {e}")
            
            # Step 5: Execution Agent executes approved trades
            if signals:
                logger.info("Step 5: Executing approved trades...")
                executed_count = 0
                for signal in signals:
                    if signal.action == SignalAction.HOLD:
                        continue  # Skip HOLD signals
                    
                    execution_result = ExecutionResult(
                        signal=signal,
                        execution_time=datetime.now()
                    )
                    
                    try:
                        order_request = {
                            "symbol": signal.symbol,
                            "quantity": signal.qty or 1,
                            "side": signal.action.value.lower(),
                            "order_type": "market"
                        }
                        
                        result = self.execution_agent.process(order_request)
                        execution_result.order_id = result.get('order_id')
                        execution_result.executed = True
                        execution_result.fill_price = signal.price
                        executed_count += 1
                        
                        logger.info(
                            f"Order executed for {signal.symbol}: {result.get('order_id', 'N/A')}"
                        )
                    except Exception as e:
                        execution_result.executed = False
                        execution_result.error = str(e)
                        logger.error(
                            f"ExecutionAgent failed for {signal.symbol}: {e}",
                            exc_info=True
                        )
                        # Continue with next trade
                    
                    execution_results.append(execution_result)
                
                if executed_count == 0:
                    logger.info("No trades executed (all signals were HOLD or execution failed)")
            else:
                logger.info("No approved signals to execute")
            
            # Step 6: Audit Agent generates report
            iteration_duration = (datetime.now() - iteration_start).total_seconds()
            
            # Create iteration summary
            iteration_summary = IterationSummary(
                iteration_number=self.iteration,
                timestamp=iteration_start,
                symbols_processed=list(market_data.keys()) if market_data else [],
                signals_generated=len(signals),
                signals_validated=len(signals),
                signals_approved=len([s for s in signals if getattr(s, 'approved', False)]),
                signals_executed=executed_count,
                execution_results=execution_results,
                errors=[],  # Could collect errors from each step
                duration_seconds=iteration_duration
            )
            
            try:
                audit_report = self.audit_agent.process(iteration_summary, execution_results)
                logger.info("Step 6: Audit report generated")
                logger.info(f"Audit Summary: {audit_report.summary[:200]}...")  # First 200 chars
            except Exception as e:
                logger.error(f"AuditAgent failed: {e}", exc_info=True)
                # Continue - audit failure shouldn't stop the system
            
            # Store signals for potential analysis (with cleanup)
            if signals:
                self.all_signals.extend(signals)
                self._cleanup_old_signals()
                
                # Update monitoring metrics
                self.monitoring_metrics["recent_signals"].extend([
                    {
                        "symbol": s.symbol,
                        "action": s.action.value,
                        "confidence": s.confidence,
                        "strategy": s.strategy_name,
                        "timestamp": datetime.now()
                    }
                    for s in signals
                ])
                # Keep only last 100 signals
                if len(self.monitoring_metrics["recent_signals"]) > 100:
                    self.monitoring_metrics["recent_signals"] = \
                        self.monitoring_metrics["recent_signals"][-100:]
            
            # Mark iteration as successful
            self.monitoring_metrics["successful_iterations"] += 1
            
            logger.info(f"Iteration {self.iteration} completed in {iteration_duration:.2f} seconds")
            
        except TradingSystemError as e:
            logger.error(
                f"Trading system error in iteration {self.iteration}: {e.message}",
                extra={"correlation_id": e.correlation_id}
            )
            self.monitoring_metrics["failed_iterations"] += 1
        except Exception as e:
            logger.exception(f"Unexpected error in iteration {self.iteration}")
            self.monitoring_metrics["failed_iterations"] += 1
    
    def get_monitoring_metrics(self) -> Dict[str, Any]:
        """
        Get monitoring metrics for dashboard.
        
        Returns:
            Dictionary with all monitoring metrics
        """
        llm_total = (
            self.monitoring_metrics["llm_success_count"] + 
            self.monitoring_metrics["llm_failure_count"]
        )
        llm_success_rate = (
            (self.monitoring_metrics["llm_success_count"] / llm_total * 100)
            if llm_total > 0 else 0.0
        )
        
        return {
            "iteration_count": self.iteration,
            "total_iterations": self.monitoring_metrics["total_iterations"],
            "successful_iterations": self.monitoring_metrics["successful_iterations"],
            "failed_iterations": self.monitoring_metrics["failed_iterations"],
            "llm_success_count": self.monitoring_metrics["llm_success_count"],
            "llm_failure_count": self.monitoring_metrics["llm_failure_count"],
            "llm_success_rate": round(llm_success_rate, 2),
            "recent_signals": self.monitoring_metrics["recent_signals"],
            "recent_equity_values": self.monitoring_metrics["recent_equity_values"],
            "circuit_breaker": self.circuit_breaker.get_metrics()
        }
            # Continue running - don't crash on single iteration failure

