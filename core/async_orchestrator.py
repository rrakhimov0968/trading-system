"""Async event-driven orchestration for the trading system."""
import asyncio
import signal
import sys
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

from config.settings import AppConfig
from agents.data_agent import DataAgent
from agents.execution_agent import ExecutionAgent
from agents.strategy_agent import StrategyAgent
from agents.quant_agent import QuantAgent
from agents.risk_agent import RiskAgent
from agents.audit_agent import AuditAgent
from models.audit import IterationSummary, ExecutionResult
from models.signal import TradingSignal, SignalAction
from models.market_data import MarketData
from utils.event_bus import EventBus
from utils.exceptions import TradingSystemError
from utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from core.market_hours import is_market_open, get_market_status_message

logger = logging.getLogger(__name__)


class AsyncTradingSystemOrchestrator:
    """
    Async event-driven orchestrator for the trading system.
    
    Uses an event bus for decoupled, non-blocking agent communication.
    Agents subscribe to events and publish new events when they complete processing.
    
    Flow:
    1. DataAgent fetches data → publishes 'data_ready'
    2. StrategyAgent subscribes to 'data_ready' → processes → publishes 'signals_ready'
    3. QuantAgent subscribes to 'signals_ready' → validates → publishes 'validated_ready'
    4. RiskAgent subscribes to 'validated_ready' → approves → publishes 'approved_ready'
    5. ExecutionAgent subscribes to 'approved_ready' → executes → publishes 'executed_ready'
    6. AuditAgent subscribes to 'executed_ready' → generates report
    """
    
    def __init__(self, config: AppConfig):
        """
        Initialize the async orchestrator.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.running = False
        self.iteration = 0
        self.event_bus = EventBus()
        
        # Circuit breaker for system protection
        import os
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
                f"Failed to initialize async orchestrator: {str(e)}"
            ) from e
        
        # Setup event subscriptions
        self._setup_subscriptions()
        
        # Store iteration state
        self._current_iteration_data: Dict[str, Any] = {}
        self._iteration_complete_event = asyncio.Event()
    
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
    
    def _setup_subscriptions(self) -> None:
        """Setup event subscriptions for agent communication."""
        # StrategyAgent subscribes to data_ready
        self.event_bus.subscribe('data_ready', self._handle_data_ready)
        
        # QuantAgent subscribes to signals_ready
        self.event_bus.subscribe('signals_ready', self._handle_signals_ready)
        
        # RiskAgent subscribes to validated_ready
        self.event_bus.subscribe('validated_ready', self._handle_validated_ready)
        
        # ExecutionAgent subscribes to approved_ready
        self.event_bus.subscribe('approved_ready', self._handle_approved_ready)
        
        # AuditAgent subscribes to executed_ready
        self.event_bus.subscribe('executed_ready', self._handle_executed_ready)
        
        logger.info("Event subscriptions setup complete")
    
    async def _handle_data_ready(self, market_data: Dict[str, MarketData]) -> None:
        """Handle data_ready event - trigger StrategyAgent."""
        try:
            logger.info(f"Processing market data for {len(market_data)} symbols")
            signals = await self._async_process(self.strategy_agent.process, market_data)
            self._current_iteration_data['signals'] = signals
            await self.event_bus.publish('signals_ready', signals)
        except Exception as e:
            logger.exception(f"Error in StrategyAgent: {e}")
            await self.event_bus.publish('signals_ready', [])
    
    async def _handle_signals_ready(self, signals: List[TradingSignal]) -> None:
        """Handle signals_ready event - trigger QuantAgent."""
        if not signals:
            logger.info("No signals to validate, skipping QuantAgent")
            await self.event_bus.publish('validated_ready', [])
            return
        
        try:
            market_data = self._current_iteration_data.get('market_data', {})
            validated_signals = await self._async_process(
                self.quant_agent.process, 
                signals, 
                market_data
            )
            self._current_iteration_data['validated_signals'] = validated_signals
            await self.event_bus.publish('validated_ready', validated_signals)
        except Exception as e:
            logger.exception(f"Error in QuantAgent: {e}")
            await self.event_bus.publish('validated_ready', signals)  # Continue with unvalidated
    
    async def _handle_validated_ready(self, signals: List[TradingSignal]) -> None:
        """Handle validated_ready event - trigger RiskAgent."""
        if not signals:
            logger.info("No signals to approve, skipping RiskAgent")
            await self.event_bus.publish('approved_ready', [])
            return
        
        try:
            approved_signals = await self._async_process(
                self.risk_agent.process,
                signals,
                execution_agent=self.execution_agent
            )
            self._current_iteration_data['approved_signals'] = approved_signals
            await self.event_bus.publish('approved_ready', approved_signals)
        except Exception as e:
            logger.exception(f"Error in RiskAgent: {e}")
            await self.event_bus.publish('approved_ready', [])
    
    async def _handle_approved_ready(self, signals: List[TradingSignal]) -> None:
        """Handle approved_ready event - trigger ExecutionAgent."""
        # Check account equity before execution (circuit breaker protection)
        try:
            account = await self._async_process(self.execution_agent.get_account)
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
                    await self.event_bus.publish('executed_ready', [])
                    return
        except Exception as e:
            logger.warning(f"Failed to check account equity: {e}")
        
        if not signals:
            logger.info("No signals to execute, skipping ExecutionAgent")
            await self.event_bus.publish('executed_ready', [])
            return
        
        approved = [s for s in signals if s.approved]
        if not approved:
            logger.info("No approved signals to execute")
            await self.event_bus.publish('executed_ready', [])
            return
        
        execution_results = []
        
        for signal in approved:
            if signal.action != SignalAction.HOLD:
                try:
                    order_request = {
                        "symbol": signal.symbol,
                        "quantity": signal.qty or 1,
                        "side": signal.action.value.lower(),
                        "order_type": "market"
                    }
                    result = await self._async_process(self.execution_agent.process, order_request)
                    execution_results.append(ExecutionResult(
                        signal=signal,
                        order_id=result.get('order_id'),
                        executed=True,
                        execution_time=datetime.now(),
                        fill_price=result.get('fill_price') or signal.price
                    ))
                except Exception as e:
                    logger.exception(f"Execution failed for {signal.symbol}: {e}")
                    execution_results.append(ExecutionResult(
                        signal=signal,
                        order_id=None,
                        executed=False,
                        execution_time=datetime.now(),
                        error=str(e),
                        fill_price=None
                    ))
        
        self._current_iteration_data['execution_results'] = execution_results
        await self.event_bus.publish('executed_ready', execution_results)
    
    async def _handle_executed_ready(self, execution_results: List[ExecutionResult]) -> None:
        """Handle executed_ready event - trigger AuditAgent."""
        try:
            # Create iteration summary
            iteration_summary = IterationSummary(
                iteration_number=self.iteration,
                timestamp=datetime.now(),
                symbols_processed=self.config.symbols,
                signals_generated=len(self._current_iteration_data.get('signals', [])),
                signals_validated=len(self._current_iteration_data.get('validated_signals', [])),
                signals_approved=len([s for s in self._current_iteration_data.get('approved_signals', []) if s.approved]),
                signals_executed=len(execution_results),
                execution_results=execution_results,
                errors=[],
                duration_seconds=0.0
            )
            
            report = await self._async_process(
                self.audit_agent.process,
                iteration_summary,
                execution_results
            )
            
            self._current_iteration_data['audit_report'] = report
            logger.info("Iteration complete, audit report generated")
            
            # Update monitoring metrics
            signals = self._current_iteration_data.get('signals', [])
            if signals:
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
            
        except Exception as e:
            logger.exception(f"Error in AuditAgent: {e}")
            self.monitoring_metrics["failed_iterations"] += 1
        finally:
            # Signal iteration complete
            self._iteration_complete_event.set()
    
    async def _async_process(self, func, *args, **kwargs):
        """
        Helper to run sync or async functions.
        
        Args:
            func: The function to call (sync or async)
            *args: Positional arguments
            **kwargs: Keyword arguments
        """
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            # Run sync function in executor
            loop = asyncio.get_event_loop()
            # Create a closure to capture args and kwargs
            def sync_wrapper():
                return func(*args, **kwargs)
            return await loop.run_in_executor(None, sync_wrapper)
    
    async def run_async_pipeline(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run a single async iteration of the trading pipeline.
        
        Args:
            symbols: Optional list of symbols to process (uses config symbols if not provided)
        
        Returns:
            Dictionary with iteration results
        """
        symbols = symbols or self.config.symbols
        self.iteration += 1
        iteration_start = datetime.now()
        
        logger.info("-" * 60)
        logger.info(f"Async Iteration {self.iteration} started at {iteration_start.isoformat()}")
        logger.info("-" * 60)
        
        # Check if market is open before running full pipeline
        if not is_market_open():
            status_msg = get_market_status_message()
            logger.info(f"Market is closed. {status_msg}")
            logger.info("Skipping iteration until market opens")
            return {"status": "market_closed", "message": status_msg}
        
        # Reset iteration state
        self._current_iteration_data = {}
        self._iteration_complete_event.clear()
        
        try:
            # Step 1: Fetch market data asynchronously
            logger.info(f"Fetching market data for {len(symbols)} symbols...")
            market_data = await self.data_agent.process_async(
                symbols=symbols,
                timeframe="1Day",
                limit=252  # 1 year of trading days (252) to satisfy strategies that need 200+ bars
            )
            
            if not market_data:
                logger.warning("No market data received, skipping iteration")
                return {"error": "No market data"}
            
            logger.info(f"Fetched data for {len(market_data)} symbols")
            self._current_iteration_data['market_data'] = market_data
            
            # Publish data_ready event to trigger pipeline
            await self.event_bus.publish('data_ready', market_data)
            
            # Wait for pipeline to complete (with timeout)
            try:
                await asyncio.wait_for(
                    self._iteration_complete_event.wait(),
                    timeout=300.0  # 5 minute timeout
                )
            except asyncio.TimeoutError:
                logger.error("Pipeline execution timed out after 5 minutes")
                return {"error": "Pipeline timeout"}
            
            # Calculate duration
            duration = (datetime.now() - iteration_start).total_seconds()
            self._current_iteration_data['duration_seconds'] = duration
            
            logger.info(f"Iteration {self.iteration} completed in {duration:.2f} seconds")
            
            return self._current_iteration_data
            
        except Exception as e:
            logger.exception(f"Error in async pipeline iteration: {e}")
            return {"error": str(e)}
    
    async def start(self) -> None:
        """Start the async orchestration loop."""
        logger.info("=" * 60)
        logger.info("Async Trading System Orchestrator Starting")
        logger.info(f"Mode: {self.config.trading_mode.value}")
        logger.info(f"Symbols: {', '.join(self.config.symbols)}")
        logger.info(f"Loop interval: {self.config.loop_interval_seconds} seconds")
        logger.info("=" * 60)
        
        # Health check all agents
        if not await self._health_check_all():
            logger.error("Health checks failed, exiting")
            sys.exit(1)
        
        self.running = True
        logger.info("Async orchestrator started, entering main loop...")
        
        try:
            while self.running:
                # Check circuit breaker before iteration
                if self.circuit_breaker.is_open():
                    logger.error(
                        f"CIRCUIT BREAKER IS OPEN - Skipping iteration. "
                        f"Reason: {self.circuit_breaker.get_metrics()}"
                    )
                    await asyncio.sleep(self.config.loop_interval_seconds)
                    continue
                
                # Check market hours - if closed, sleep longer
                if not is_market_open():
                    status_msg = get_market_status_message()
                    logger.info(f"Market is closed. {status_msg}")
                    logger.info("Sleeping for 5 minutes until next check...")
                    # Sleep 5 minutes instead of 60 seconds when market is closed
                    await asyncio.sleep(300)
                    continue
                
                self.monitoring_metrics["total_iterations"] += 1
                await self.run_async_pipeline()
                
                if self.running:
                    await asyncio.sleep(self.config.loop_interval_seconds)
        
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            await self.stop()
        except Exception as e:
            logger.exception("Unexpected error in async orchestrator loop")
            await self.stop()
            raise
    
    async def stop(self) -> None:
        """Stop the async orchestration loop."""
        if not self.running:
            return
        
        logger.info("Stopping async orchestrator...")
        self.running = False
        
        # Cleanup async resources
        if hasattr(self.data_agent, 'cleanup_async_resources'):
            await self.data_agent.cleanup_async_resources()
        
        logger.info(f"Async orchestrator stopped after {self.iteration} iterations")
    
    async def _health_check_all(self) -> bool:
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

