"""Async event-driven orchestration for the trading system."""
import asyncio
import signal
import sys
import json
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
from models.enums import OrderSide
from models.market_data import MarketData
from utils.event_bus import EventBus
from utils.exceptions import TradingSystemError
from utils.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from core.market_hours import is_market_open, get_market_status_message
from core.position_manager import PositionManager
from core.orchestrator_integration import (
    initialize_tier_tracker,
    initialize_signal_validator,
    initialize_tiered_sizer,
    load_focus_symbols,
    load_symbol_tier_mapping
)
from agents.market_regime_agent import MarketRegimeAgent
from core.market_regime import MarketRegime
from pathlib import Path
import os

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
        
        # Initialize order tracker to prevent duplicate orders
        from core.order_tracker import OrderTracker
        self.order_tracker = OrderTracker(
            cooldown_minutes=1440,  # 24 hour cooldown
            max_orders_per_day=int(os.getenv("MAX_ORDERS_PER_DAY", "10"))
        )
        
        # Initialize agents
        logger.info("Initializing agents...")
        try:
            self.data_agent = DataAgent(config=config)
            self.strategy_agent = StrategyAgent(config=config)
            self.quant_agent = QuantAgent(config=config)
            self.risk_agent = RiskAgent(config=config)
            self.execution_agent = ExecutionAgent(config=config)
            self.audit_agent = AuditAgent(config=config)
            
            # Initialize database manager for position context
            from utils.database import DatabaseManager
            db_manager = DatabaseManager(config=config)
            
            # Initialize PositionManager for stop-loss management
            self.position_manager = PositionManager(
                config=config,
                execution_agent=self.execution_agent,
                data_agent=self.data_agent,
                database_manager=db_manager
            )
            
            # Initialize tier exposure tracker (if tiered allocation enabled)
            self.tier_tracker = None
            if config.enable_tiered_allocation:
                try:
                    self.tier_tracker = initialize_tier_tracker(config)
                except Exception as e:
                    logger.warning(f"Failed to initialize tier tracker: {e}")
            
            # Initialize signal validator (if scanner enabled)
            self.signal_validator = None
            if config.use_scanner:
                try:
                    self.signal_validator = initialize_signal_validator(config)
                except Exception as e:
                    logger.warning(f"Failed to initialize signal validator: {e}")
            
            # Tiered position sizer will be initialized after account value is known
            self.tiered_sizer = None
            
            # Store symbol tier mapping for quick lookup
            self.symbol_tier_mapping = load_symbol_tier_mapping(config)
            
            # Initialize market regime agent (if enabled)
            self.market_regime_agent = None
            if config.enable_regime_filter:
                self.market_regime_agent = MarketRegimeAgent(
                    config=config,
                    data_agent=self.data_agent
                )
                mode_str = "strict (hard gate)" if config.strict_regime else "soft (scalar)"
                logger.info(f"✅ Market regime agent initialized: {config.regime_benchmark} > SMA{config.regime_sma_period} ({mode_str} mode)")
            
            logger.info("All agents and PositionManager initialized successfully")
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
    
    def check_emergency_stop(self) -> bool:
        """
        Check if emergency stop file exists.
        
        Returns:
            True if emergency stop is active, False otherwise
        """
        stop_file = Path("EMERGENCY_STOP")
        
        if stop_file.exists():
            logger.error(
                "🚨 EMERGENCY STOP ACTIVE - NOT PLACING ANY ORDERS! "
                "Remove EMERGENCY_STOP file to resume trading."
            )
            return True
        
        return False
    
    def validate_account_health(self, account) -> bool:
        """
        Validate account is in good state before trading.
        
        Args:
            account: Account object from broker
        
        Returns:
            True if account is healthy, False otherwise
        """
        try:
            equity = float(account.equity) if hasattr(account, 'equity') else 0.0
            cash = float(account.cash) if hasattr(account, 'cash') else 0.0
            buying_power = float(account.buying_power) if hasattr(account, 'buying_power') else 0.0
            
            # Check 1: Equity hasn't dropped more than 20% from initial
            initial_equity = float(os.getenv("INITIAL_ACCOUNT_EQUITY", "100000"))
            if equity > 0 and initial_equity > 0:
                current_drawdown = (initial_equity - equity) / initial_equity
                
                if current_drawdown > 0.20:
                    logger.error(
                        f"🚫 ACCOUNT DRAWDOWN TOO HIGH: {current_drawdown:.1%} "
                        f"(Equity: ${equity:,.0f} vs Initial: ${initial_equity:,.0f})"
                    )
                    return False
            
            # Check 2: Have enough cash for at least 1 position
            min_cash_needed = equity * 0.15 if equity > 0 else 0.0  # Need at least 15% in cash
            if cash < min_cash_needed:
                logger.warning(
                    f"⚠️  LOW CASH: ${cash:,.0f} < ${min_cash_needed:,.0f} "
                    f"(15% of equity)"
                )
                # Don't block, just warn
            
            # Check 3: Not on margin call
            if buying_power < 0:
                logger.error("🚫 NEGATIVE BUYING POWER - MARGIN CALL RISK!")
                return False
            
            logger.info(
                f"✅ Account Health: Equity=${equity:,.0f}, Cash=${cash:,.0f}, "
                f"Buying Power=${buying_power:,.0f}"
            )
            return True
            
        except Exception as e:
            logger.error(f"Error validating account health: {e}", exc_info=True)
            return False
    
    async def reconcile_positions(self) -> None:
        """
        Verify system's understanding of positions matches broker.
        Run this once per iteration to detect manual closes/opens.
        """
        logger.info("🔍 Reconciling positions with broker...")
        
        try:
            # Get positions from broker
            alpaca_positions = await self._async_process(self.execution_agent.get_positions)
            alpaca_symbols = {pos.symbol for pos in alpaca_positions}
            
            # Get positions from order tracker (symbols we've placed orders for)
            tracked_symbols = set(self.order_tracker.last_order_time.keys())
            
            # Check for mismatches
            missing_in_alpaca = tracked_symbols - alpaca_symbols
            missing_in_tracker = alpaca_symbols - tracked_symbols
            
            if missing_in_alpaca:
                logger.warning(
                    f"⚠️  Tracked positions not in broker: {missing_in_alpaca} "
                    f"(May have been manually closed)"
                )
                # Clear from tracker
                for symbol in missing_in_alpaca:
                    self.order_tracker.clear_tracking(symbol)
            
            if missing_in_tracker:
                logger.warning(
                    f"⚠️  Broker positions not tracked: {missing_in_tracker} "
                    f"(May have been manually opened)"
                )
                # Add to tracker to prevent duplicate orders
                for symbol in missing_in_tracker:
                    self.order_tracker.record_order(symbol)
            
            # Log position values
            total_position_value = sum(
                float(pos.qty) * float(pos.current_price)
                for pos in alpaca_positions
                if hasattr(pos, 'qty') and hasattr(pos, 'current_price')
            )
            
            account = await self._async_process(self.execution_agent.get_account)
            equity = float(account.equity) if account and hasattr(account, 'equity') else 0.0
            position_pct = (total_position_value / equity * 100) if equity > 0 else 0
            
            logger.info(
                f"✅ Positions reconciled: {len(alpaca_positions)} positions, "
                f"${total_position_value:,.0f} ({position_pct:.1f}% of equity)"
            )
            
        except Exception as e:
            logger.error(f"Error reconciling positions: {e}", exc_info=True)
    
    async def _handle_data_ready(self, market_data: Dict[str, MarketData]) -> None:
        """Handle data_ready event - trigger StrategyAgent (after regime check)."""
        try:
            logger.info(f"Processing market data for {len(market_data)} symbols")
            
            # MARKET REGIME CHECK (system-level protection)
            current_regime: Optional[MarketRegime] = None
            if self.market_regime_agent and market_data:
                try:
                    current_regime = self.market_regime_agent.process(market_data)
                    logger.info(
                        "Market regime evaluated",
                        extra={
                            "allowed": current_regime.allowed,
                            "risk_scalar": current_regime.risk_scalar,
                            "reason": current_regime.reason
                        }
                    )
                    
                    if not current_regime.allowed:
                        logger.warning(f"🚫 MARKET REGIME: {current_regime.reason}")
                        logger.warning("Trading halted due to market regime (strict mode enabled)")
                        # Skip signal generation
                        await self.event_bus.publish('signals_ready', [])
                        return
                    else:
                        logger.info(f"✅ MARKET REGIME: {current_regime.reason}")
                except Exception as e:
                    logger.error(f"Market regime evaluation failed: {e}", exc_info=True)
                    logger.warning("Allowing trading despite regime evaluation error")
                    current_regime = MarketRegime(allowed=True, risk_scalar=1.0, reason="Evaluation_error")
            else:
                current_regime = MarketRegime(allowed=True, risk_scalar=1.0, reason="Regime_filter_disabled")
            
            # Store regime for use in position sizing
            self._current_regime = current_regime
            
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
        
        # CRITICAL: Get current positions to prevent duplicate orders
        try:
            current_positions = await self._async_process(self.execution_agent.get_positions)
            position_symbols = {pos.symbol for pos in current_positions}
            logger.info(f"Current open positions: {', '.join(position_symbols) if position_symbols else 'None'}")
        except Exception as e:
            logger.warning(f"Failed to fetch positions: {e}. Proceeding with caution.")
            current_positions = []
            position_symbols = set()
        
        # Get account value for tier tracking and sizing
        account_value = None
        try:
            account_check = await self._async_process(self.execution_agent.get_account)
            account_value = float(account_check.equity) if hasattr(account_check, 'equity') else float(account_check.cash)
            
            # Initialize tiered sizer if needed (and not already initialized)
            if self.config.enable_tiered_allocation and not self.tiered_sizer:
                self.tiered_sizer = initialize_tiered_sizer(self.config, account_value)
        except Exception as e:
            logger.warning(f"Failed to get account value: {e}")
        
        # Calculate tier exposures if tier tracking enabled
        tier_exposures = None
        if self.tier_tracker and account_value:
            try:
                tier_exposures = self.tier_tracker.calculate_tier_exposure(current_positions, account_value)
                self.tier_tracker.log_tier_status(tier_exposures)
            except Exception as e:
                logger.warning(f"Failed to calculate tier exposures: {e}")
        
        # Load scanner data if using scanner
        scanner_data = None
        if self.config.use_scanner and self.signal_validator:
            scanner_file = Path(self.config.scanner_file)
            if scanner_file.exists():
                try:
                    with open(scanner_file) as f:
                        scanner_data = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load scanner data: {e}")
        
        execution_results = []
        
        for signal in approved:
            if signal.action != SignalAction.HOLD:
                # CRITICAL FIX 1: Check for existing positions before BUY
                if signal.action == SignalAction.BUY:
                    # Check if we already have a position
                    if signal.symbol in position_symbols:
                        logger.warning(
                            f"🚫 SKIPPING BUY order for {signal.symbol}: "
                            f"Already have an open position. "
                            f"This prevents duplicate orders and over-leveraging."
                        )
                        execution_results.append(ExecutionResult(
                            signal=signal,
                            executed=False,
                            execution_time=datetime.now(),
                            error="Position already exists"
                        ))
                        continue
                    
                    # CRITICAL FIX 4: Check order cooldown (prevent multiple orders per day)
                    can_place, reason = self.order_tracker.can_place_order(signal.symbol)
                    if not can_place:
                        logger.warning(
                            f"🚫 SKIPPING BUY order for {signal.symbol}: {reason}"
                        )
                        execution_results.append(ExecutionResult(
                            signal=signal,
                            executed=False,
                            execution_time=datetime.now(),
                            error=reason
                        ))
                        continue
                    
                    # NEW: Tier exposure check (CRITICAL - prevents drift)
                    if self.tier_tracker and tier_exposures and account_value:
                        tier = self.tier_tracker.get_tier_for_symbol(signal.symbol)
                        if tier:
                            # Estimate proposed position value (will be recalculated with actual sizing)
                            proposed_value = (signal.qty or 0) * (signal.price or 0)
                            if proposed_value == 0:
                                # Get current price for estimation
                                try:
                                    from agents.data_agent import DataAgent
                                    market_data_dict = await self.data_agent.process_async([signal.symbol], limit=1)
                                    if signal.symbol in market_data_dict and market_data_dict[signal.symbol].bars:
                                        current_price = market_data_dict[signal.symbol].bars[-1].close
                                        proposed_value = (signal.qty or 0) * current_price
                                except:
                                    pass
                            
                            # ATOMIC: Reserve exposure (Problem 2 Fix)
                            approved_tier, tier_reason = self.tier_tracker.reserve_exposure(
                                tier=tier,
                                proposed_value=proposed_value,
                                current_tier_exposure=tier_exposures[tier],
                                account_value=account_value,
                                symbol=signal.symbol
                            )
                            
                            if not approved_tier:
                                logger.warning(
                                    f"🚫 SKIPPING BUY order for {signal.symbol}: Tier exposure check failed - {tier_reason}",
                                    extra={
                                        "symbol": signal.symbol,
                                        "reason": "tier_exposure",
                                        "tier": tier,
                                        "tier_reason": tier_reason
                                    }
                                )
                                execution_results.append(ExecutionResult(
                                    signal=signal,
                                    executed=False,
                                    execution_time=datetime.now(),
                                    error=f"Tier exposure: {tier_reason}"
                                ))
                                continue
                    
                    # NEW: Signal freshness check (if using scanner)
                    if self.signal_validator and scanner_data:
                        try:
                            # Get current price
                            current_price = signal.price
                            if not current_price:
                                try:
                                    market_data_dict = await self.data_agent.process_async([signal.symbol], limit=1)
                                    if signal.symbol in market_data_dict and market_data_dict[signal.symbol].bars:
                                        current_price = market_data_dict[signal.symbol].bars[-1].close
                                except:
                                    pass
                            
                            if current_price:
                                validation = self.signal_validator.validate_from_scanner_data(
                                    signal.symbol,
                                    current_price,
                                    scanner_data
                                )
                                
                                if not validation.valid:
                                    logger.warning(
                                        f"🚫 SKIPPING BUY order for {signal.symbol}: Signal freshness check failed - {validation.reason}"
                                    )
                                    execution_results.append(ExecutionResult(
                                        signal=signal,
                                        executed=False,
                                        execution_time=datetime.now(),
                                        error=f"Signal freshness: {validation.reason}"
                                    ))
                                    continue
                        except Exception as e:
                            logger.warning(f"Signal freshness check failed for {signal.symbol}: {e}")
                    
                    # NEW: Tiered position sizing (if enabled)
                    if self.tiered_sizer and account_value:
                        tier = self.symbol_tier_mapping.get(signal.symbol)
                        if tier:
                            try:
                                # Get current price if not set
                                current_price = signal.price
                                if not current_price:
                                    market_data_dict = await self.data_agent.process_async([signal.symbol], limit=1)
                                    if signal.symbol in market_data_dict and market_data_dict[signal.symbol].bars:
                                        current_price = market_data_dict[signal.symbol].bars[-1].close
                                
                                if current_price:
                                    # ALWAYS pass live account_value (Problem 3 Fix)
                                    # Apply market regime scalar to position sizing
                                    regime_scalar = self._current_regime.risk_scalar if hasattr(self, '_current_regime') and self._current_regime else 1.0
                                    shares, meta = self.tiered_sizer.calculate_shares(
                                        signal.symbol,
                                        current_price,
                                        tier,
                                        account_value=account_value,  # Live equity, not cached
                                        regime_scalar=regime_scalar  # Market regime risk scalar
                                    )
                                    
                                    if shares and shares > 0:
                                        signal.qty = shares if self.tiered_sizer.use_fractional else int(shares)
                                        signal.price = current_price
                                        logger.info(
                                            f"📊 Tiered sizing for {signal.symbol}: {signal.qty:.4f} shares "
                                            f"(${meta.get('position_notional', 0):,.2f} notional)"
                                        )
                                    else:
                                        logger.warning(
                                            f"🚫 SKIPPING BUY order for {signal.symbol}: Position sizing failed - {meta.get('reason', 'Unknown')}",
                                            extra={
                                                "symbol": signal.symbol,
                                                "reason": "position_sizing",
                                                "tier": tier,
                                                "sizing_reason": meta.get('reason')
                                            }
                                        )
                                        # Release reserved exposure on failure (Problem 2 Fix)
                                        if self.tier_tracker:
                                            tier_for_release = self.tier_tracker.get_tier_for_symbol(signal.symbol)
                                            if tier_for_release:
                                                self.tier_tracker.release_exposure(tier_for_release, proposed_value, signal.symbol)
                                        execution_results.append(ExecutionResult(
                                            signal=signal,
                                            executed=False,
                                            execution_time=datetime.now(),
                                            error=f"Position sizing: {meta.get('reason', 'Failed')}"
                                        ))
                                        continue
                            except Exception as e:
                                logger.warning(f"Tiered sizing failed for {signal.symbol}: {e}")
                
                # CRITICAL FIX 2: Check we have position before SELL
                elif signal.action == SignalAction.SELL:
                    if signal.symbol not in position_symbols:
                        logger.warning(
                            f"🚫 SKIPPING SELL order for {signal.symbol}: "
                            f"No open position found. Cannot sell what we don't own."
                        )
                        execution_results.append(ExecutionResult(
                            signal=signal,
                            executed=False,
                            execution_time=datetime.now(),
                            error="No position to sell"
                        ))
                        continue
                
                # CRITICAL FIX 3: Validate position sizing
                if signal.action == SignalAction.BUY and signal.qty:
                    try:
                        # Account is already fetched earlier in this method
                        account_check = await self._async_process(self.execution_agent.get_account)
                        account_value = float(account_check.equity) if hasattr(account_check, 'equity') else float(account_check.cash)
                        position_value = signal.qty * (signal.price or 0)
                        
                        if position_value > account_value * 0.25:  # Max 25% per position
                            logger.error(
                                f"🚫 SKIPPING BUY order for {signal.symbol}: "
                                f"Position size ${position_value:,.2f} exceeds 25% of account "
                                f"(${account_value:,.2f}). This is a safety check."
                            )
                            execution_results.append(ExecutionResult(
                                signal=signal,
                                executed=False,
                                execution_time=datetime.now(),
                                error=f"Position size too large: ${position_value:,.2f} > ${account_value * 0.25:,.2f}"
                            ))
                            continue
                    except Exception as e:
                        logger.warning(f"Could not validate position size for {signal.symbol}: {e}")
                
                order_id = None
                fill_price = None
                executed = False
                error = None
                
                try:
                    order_request = {
                        "symbol": signal.symbol,
                        "quantity": signal.qty or 1,
                        "side": signal.action.value.lower(),
                        "order_type": "market"
                    }
                    
                    logger.info(
                        f"Placing {signal.action.value} order: {signal.qty or 1} shares of {signal.symbol}"
                    )
                    
                    # Use fractional fallback if fractional shares enabled
                    if self.config.enable_fractional_shares and signal.qty and signal.qty != int(signal.qty):
                        # Use place_order_with_fallback for fractional shares
                        try:
                            order_side = OrderSide.BUY if signal.action == SignalAction.BUY else OrderSide.SELL
                            
                            order = await self._async_process(
                                self.execution_agent.place_order_with_fallback,
                                signal.symbol,
                                signal.qty,
                                order_side,
                                'market'
                            )
                            
                            if order:
                                order_id = order.id if hasattr(order, 'id') else str(order)
                                fill_price = getattr(order, 'filled_avg_price', None) or signal.price
                                executed = True
                            else:
                                raise Exception("Order returned None")
                        except Exception as e:
                            logger.error(f"Fractional order failed for {signal.symbol}: {e}")
                            result = await self._async_process(self.execution_agent.process, order_request)
                            order_id = result.get('order_id')
                            fill_price = result.get('fill_price') or signal.price
                            executed = True
                    else:
                        result = await self._async_process(self.execution_agent.process, order_request)
                        order_id = result.get('order_id')
                        fill_price = result.get('fill_price') or signal.price
                        executed = True  # Order was placed successfully
                    
                    logger.info(
                        f"✅ Order executed for {signal.symbol}: {order_id or 'N/A'}"
                    )
                    
                    # Update position tracking after successful order
                    if signal.action == SignalAction.BUY:
                        position_symbols.add(signal.symbol)
                        # Record order to prevent duplicates within cooldown period
                        self.order_tracker.record_order(signal.symbol)
                    elif signal.action == SignalAction.SELL:
                        position_symbols.discard(signal.symbol)
                        # Clear tracking after SELL so we can buy again later
                        self.order_tracker.clear_tracking(signal.symbol)
                    
                except Exception as e:
                    error_str = str(e)
                    logger.exception(
                        f"Execution failed for {signal.symbol}: {e}",
                        extra={
                            "symbol": signal.symbol,
                            "reason": "execution_error",
                            "error": str(e),
                            "action": signal.action.value
                        }
                    )
                    executed = False
                    error = error_str
                    
                    # Release reserved exposure on execution failure (Problem 2 Fix)
                    if signal.action == SignalAction.BUY and self.tier_tracker:
                        tier_for_release = self.tier_tracker.get_tier_for_symbol(signal.symbol)
                        if tier_for_release:
                            proposed_value = (signal.qty or 0) * (signal.price or 0)
                            if proposed_value > 0:
                                self.tier_tracker.release_exposure(tier_for_release, proposed_value, signal.symbol)
                                logger.info(f"🔓 Released reserved exposure for {signal.symbol} after execution failure")
                
                # Create ExecutionResult regardless of success/failure
                # This ensures all trades are logged to DB
                execution_results.append(ExecutionResult(
                    signal=signal,
                    order_id=order_id,
                    executed=executed,
                    execution_time=datetime.now(),
                    error=error,
                    fill_price=fill_price
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
        
        Uses focus symbols (scanner-driven or all configured) if symbols not provided.
        """
        # Use focus symbols if not explicitly provided
        if symbols is None:
            symbols = load_focus_symbols(self.config)
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
        
        # Check emergency stop first
        if self.check_emergency_stop():
            logger.error("🚨 EMERGENCY STOP ACTIVE - Skipping iteration")
            return {"status": "emergency_stop", "message": "Emergency stop file exists"}
        
        # Reconcile positions with broker (once per iteration)
        await self.reconcile_positions()
        
        # Validate account health before proceeding
        try:
            account = await self._async_process(self.execution_agent.get_account)
            if not self.validate_account_health(account):
                logger.error("🚫 Account health check failed - STOPPING TRADING")
                self.monitoring_metrics["failed_iterations"] += 1
                return {"status": "account_unhealthy", "message": "Account health check failed"}
        except Exception as e:
            logger.error(f"Failed to validate account health: {e}", exc_info=True)
            self.monitoring_metrics["failed_iterations"] += 1
            return {"status": "error", "message": f"Account health check error: {e}"}
        
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
                
                # Check positions for stop-loss triggers (runs every iteration)
                # This is the "hard brake" to prevent holding losing positions
                try:
                    stop_results = self.position_manager.check_stops()
                    if stop_results.get("stop_orders_placed", 0) > 0:
                        logger.warning(
                            f"PositionManager exited {stop_results['stop_orders_placed']} position(s): "
                            f"{', '.join(stop_results.get('symbols_exited', []))}"
                        )
                except Exception as e:
                    logger.exception(f"Error in PositionManager.check_stops: {str(e)}")
                    # Don't stop the main loop if position manager fails
                
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

