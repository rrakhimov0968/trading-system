"""Database utilities for persistence and state management."""
import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from contextlib import contextmanager
import uuid

from sqlalchemy import (
    Column, String, Float, Integer, DateTime, Boolean, Text, Enum,
    create_engine, ForeignKey
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

from models.signal import SignalAction, TradingSignal
from models.audit import ExecutionResult, IterationSummary, AuditReport
from config.settings import DatabaseConfig, AppConfig
import json

logger = logging.getLogger(__name__)

Base = declarative_base()


class TradeHistory(Base):
    """Trade history table for persisting executed trades."""
    __tablename__ = 'trade_history'
    
    id = Column(String, primary_key=True)  # Correlation ID
    symbol = Column(String, nullable=False, index=True)
    action = Column(Enum(SignalAction), nullable=False)
    strategy_name = Column(String, nullable=False)
    qty = Column(Integer)
    price = Column(Float, nullable=False)
    fill_price = Column(Float)  # Actual execution price
    stop_loss = Column(Float)
    take_profit = Column(Float)
    risk_amount = Column(Float)
    confidence = Column(Float)
    timestamp = Column(DateTime, nullable=False, index=True)
    execution_time = Column(DateTime)
    order_id = Column(String, index=True)
    executed = Column(Boolean, default=False)
    error = Column(Text)
    
    # Quantitative metrics
    sharpe = Column(Float)
    drawdown = Column(Float)
    expectancy = Column(Float)
    
    # Additional metadata
    reasoning = Column(Text)
    correlation_id = Column(String, index=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class EquityCurve(Base):
    """Equity curve tracking table."""
    __tablename__ = 'equity_curve'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, nullable=False, index=True)
    equity = Column(Float, nullable=False)
    cash = Column(Float)
    buying_power = Column(Float)
    total_return = Column(Float)
    daily_return = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class RiskMetrics(Base):
    """Risk metrics tracking table."""
    __tablename__ = 'risk_metrics'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Position metrics
    total_positions = Column(Integer, default=0)
    total_exposure = Column(Float, default=0.0)
    max_position_size = Column(Float)
    
    # Risk metrics
    portfolio_value = Column(Float)
    daily_pnl = Column(Float, default=0.0)
    total_pnl = Column(Float, default=0.0)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    
    # Risk limits
    risk_per_trade = Column(Float)
    daily_loss_limit = Column(Float)
    current_daily_loss = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=datetime.utcnow)


class BacktestResults(Base):
    """Backtest results table for strategy validation."""
    __tablename__ = 'backtest_results'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    strategy_name = Column(String, nullable=False, index=True)
    symbol = Column(String, nullable=False, index=True)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    
    # Performance metrics
    total_return = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    win_rate = Column(Float)
    total_trades = Column(Integer)
    avg_trade_return = Column(Float)
    
    # Validation status
    passed = Column(Boolean, default=False, index=True)
    min_sharpe = Column(Float)
    max_drawdown_threshold = Column(Float)
    min_win_rate = Column(Float)
    
    # Additional metadata
    parameters = Column(Text)  # JSON string of strategy parameters
    risk_free_rate = Column(Float)  # Risk-free rate used for Sharpe
    initial_cash = Column(Float)
    commission = Column(Float)
    
    # Walk-forward info
    is_walk_forward = Column(Boolean, default=False)
    walk_forward_period = Column(Integer)  # Which period in walk-forward
    
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class IterationLog(Base):
    """Iteration summary log table."""
    __tablename__ = 'iteration_log'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    iteration_number = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    
    # Counts
    symbols_processed = Column(Text)  # JSON array
    signals_generated = Column(Integer, default=0)
    signals_validated = Column(Integer, default=0)
    signals_approved = Column(Integer, default=0)
    signals_executed = Column(Integer, default=0)
    
    # Performance
    duration_seconds = Column(Float, default=0.0)
    errors = Column(Text)  # JSON array
    
    created_at = Column(DateTime, default=datetime.utcnow)


class AuditReportLog(Base):
    """Audit report log table."""
    __tablename__ = 'audit_report_log'
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    report_type = Column(String, nullable=False, index=True)  # "iteration", "daily", "weekly"
    timestamp = Column(DateTime, nullable=False, index=True)
    summary = Column(Text, nullable=False)
    recommendations = Column(Text)
    metrics = Column(Text)  # JSON
    
    created_at = Column(DateTime, default=datetime.utcnow)


class DatabaseManager:
    """
    Database manager for persisting trading system state.
    
    Handles all database operations including trade history, equity curve,
    risk metrics, and iteration logs.
    """
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize database manager.
        
        Args:
            config: Application configuration. If None, loads from environment.
        """
        if config is None:
            from config.settings import get_config
            config = get_config()
        
        self.config = config
        self.db_config = config.database
        
        if not self.db_config:
            logger.warning("No database configuration found. Using default SQLite.")
            # Use default SQLite database in project root
            default_db_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "trading_system.db"
            )
            db_url = f"sqlite:///{default_db_path}"
        else:
            db_url = self.db_config.url
            
            # Check if PostgreSQL URL but psycopg2 not available - fallback to SQLite
            if db_url.startswith("postgresql://") or db_url.startswith("postgres://"):
                try:
                    import psycopg2  # noqa: F401
                except ImportError:
                    logger.warning(
                        f"PostgreSQL URL configured but psycopg2 not installed. "
                        f"Falling back to SQLite: {db_url} → trading_system.db"
                    )
                    # Use default SQLite database in project root
                    default_db_path = os.path.join(
                        os.path.dirname(os.path.dirname(__file__)),
                        "trading_system.db"
                    )
                    db_url = f"sqlite:///{default_db_path}"
        
        # Create engine
        # SQLite doesn't support pool_size and max_overflow
        engine_kwargs = {
            "echo": self.db_config.echo if self.db_config else False,
            "pool_pre_ping": True  # Verify connections before using
        }
        
        # Only add pool parameters for non-SQLite databases
        if not db_url.startswith("sqlite"):
            engine_kwargs["pool_size"] = self.db_config.pool_size if self.db_config else 5
            engine_kwargs["max_overflow"] = self.db_config.max_overflow if self.db_config else 10
        
        try:
            self.engine = create_engine(db_url, **engine_kwargs)
        except Exception as e:
            # If engine creation fails (e.g., missing driver), fallback to SQLite
            logger.error(f"Failed to create database engine for {db_url}: {e}")
            logger.warning("Falling back to SQLite database")
            default_db_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "trading_system.db"
            )
            db_url = f"sqlite:///{default_db_path}"
            engine_kwargs = {
                "echo": False,
                "pool_pre_ping": True
            }
            self.engine = create_engine(db_url, **engine_kwargs)
        
        # Create session factory
        self.Session = sessionmaker(bind=self.engine)
        
        # Create tables
        Base.metadata.create_all(self.engine)
        
        logger.info(f"DatabaseManager initialized with URL: {db_url}")
    
    @contextmanager
    def session(self):
        """
        Context manager for database sessions.
        
        Usage:
            with db.session() as session:
                # Use session
        """
        session = self.Session()
        try:
            yield session
            session.commit()
        except SQLAlchemyError as e:
            session.rollback()
            logger.exception(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    def generate_correlation_id(self) -> str:
        """Generate a correlation ID for tracking."""
        return str(uuid.uuid4())
    
    def log_trade(
        self,
        signal: TradingSignal,
        execution_result: Optional[ExecutionResult] = None,
        correlation_id: Optional[str] = None
    ) -> str:
        """
        Log a trade to the database.
        
        Args:
            signal: TradingSignal object
            execution_result: Optional ExecutionResult from execution
            correlation_id: Optional correlation ID (generated if not provided)
        
        Returns:
            Correlation ID for the logged trade
        """
        if correlation_id is None:
            correlation_id = self.generate_correlation_id()
        
        try:
            with self.session() as session:
                # Extract execution details if available
                fill_price = signal.price
                order_id = None
                executed = False
                execution_time = None
                error = None
                
                if execution_result:
                    fill_price = execution_result.fill_price or signal.price
                    order_id = execution_result.order_id
                    executed = execution_result.executed
                    execution_time = execution_result.execution_time or datetime.utcnow()
                    error = execution_result.error
                
                # Extract quantitative metrics (if available in signal)
                sharpe = getattr(signal, 'sharpe', None)
                drawdown = getattr(signal, 'drawdown', None)
                expectancy = getattr(signal, 'expectancy', None)
                
                # Check if trade with this ID already exists (upsert logic)
                existing_trade = session.query(TradeHistory).filter_by(id=correlation_id).first()
                
                if existing_trade:
                    # Update existing trade
                    existing_trade.symbol = signal.symbol
                    existing_trade.action = signal.action
                    existing_trade.strategy_name = signal.strategy_name
                    existing_trade.qty = signal.qty
                    existing_trade.price = signal.price
                    existing_trade.fill_price = fill_price
                    existing_trade.stop_loss = signal.stop_loss
                    existing_trade.take_profit = signal.take_profit
                    existing_trade.risk_amount = signal.risk_amount
                    existing_trade.confidence = signal.confidence
                    existing_trade.timestamp = signal.timestamp
                    existing_trade.execution_time = execution_time
                    existing_trade.order_id = order_id
                    existing_trade.executed = executed
                    existing_trade.error = error
                    existing_trade.sharpe = sharpe
                    existing_trade.drawdown = drawdown
                    existing_trade.expectancy = expectancy
                    existing_trade.reasoning = signal.reasoning
                    existing_trade.correlation_id = correlation_id
                    logger.debug(f"Updated existing trade for {signal.symbol} with correlation_id {correlation_id}")
                else:
                    # Insert new trade
                    trade = TradeHistory(
                        id=correlation_id,
                        symbol=signal.symbol,
                        action=signal.action,
                        strategy_name=signal.strategy_name,
                        qty=signal.qty,
                        price=signal.price,
                        fill_price=fill_price,
                        stop_loss=signal.stop_loss,
                        take_profit=signal.take_profit,
                        risk_amount=signal.risk_amount,
                        confidence=signal.confidence,
                        timestamp=signal.timestamp,
                        execution_time=execution_time,
                        order_id=order_id,
                        executed=executed,
                        error=error,
                        sharpe=sharpe,
                        drawdown=drawdown,
                        expectancy=expectancy,
                        reasoning=signal.reasoning,
                        correlation_id=correlation_id
                    )
                    session.add(trade)
                    logger.debug(f"Logged trade for {signal.symbol} with correlation_id {correlation_id}")
            
            return correlation_id
            
        except IntegrityError as e:
            # Handle race condition where another process inserted the same trade
            logger.warning(f"Trade with correlation_id {correlation_id} already exists, skipping duplicate insert")
            return correlation_id
        except Exception as e:
            logger.exception(f"Failed to log trade: {e}")
            raise
    
    def log_iteration(self, iteration_summary: IterationSummary) -> str:
        """
        Log an iteration summary to the database.
        
        Args:
            iteration_summary: IterationSummary object
        
        Returns:
            ID of the logged iteration
        """
        import json
        
        iteration_id = self.generate_correlation_id()
        
        try:
            with self.session() as session:
                iteration_log = IterationLog(
                    id=iteration_id,
                    iteration_number=iteration_summary.iteration_number,
                    timestamp=iteration_summary.timestamp,
                    symbols_processed=json.dumps(iteration_summary.symbols_processed),
                    signals_generated=iteration_summary.signals_generated,
                    signals_validated=iteration_summary.signals_validated,
                    signals_approved=iteration_summary.signals_approved,
                    signals_executed=iteration_summary.signals_executed,
                    duration_seconds=iteration_summary.duration_seconds,
                    errors=json.dumps(iteration_summary.errors) if iteration_summary.errors else None
                )
                
                session.add(iteration_log)
                logger.debug(f"Logged iteration {iteration_summary.iteration_number}")
            
            return iteration_id
            
        except Exception as e:
            logger.exception(f"Failed to log iteration: {e}")
            raise
    
    def log_audit_report(self, audit_report: AuditReport) -> str:
        """
        Log an audit report to the database.
        
        Args:
            audit_report: AuditReport object
        
        Returns:
            ID of the logged report
        """
        import json
        
        report_id = self.generate_correlation_id()
        
        try:
            with self.session() as session:
                report_log = AuditReportLog(
                    id=report_id,
                    report_type=audit_report.report_type,
                    timestamp=audit_report.timestamp,
                    summary=audit_report.summary,
                    recommendations=audit_report.recommendations,
                    metrics=json.dumps(audit_report.metrics) if audit_report.metrics else None
                )
                
                session.add(report_log)
                logger.debug(f"Logged audit report: {audit_report.report_type}")
            
            return report_id
            
        except Exception as e:
            logger.exception(f"Failed to log audit report: {e}")
            raise
    
    def log_equity_snapshot(
        self,
        equity: float,
        cash: Optional[float] = None,
        buying_power: Optional[float] = None,
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        Log an equity curve snapshot.
        
        Args:
            equity: Current equity value
            cash: Current cash balance
            buying_power: Current buying power
            timestamp: Optional timestamp (uses now if not provided)
        
        Returns:
            ID of the logged snapshot
        """
        snapshot_id = self.generate_correlation_id()
        
        try:
            with self.session() as session:
                # Calculate returns if previous snapshot exists
                prev_snapshot = session.query(EquityCurve).order_by(
                    EquityCurve.timestamp.desc()
                ).first()
                
                total_return = None
                daily_return = None
                
                if prev_snapshot:
                    total_return = ((equity - prev_snapshot.equity) / prev_snapshot.equity) * 100 if prev_snapshot.equity > 0 else 0
                    # Simple daily return (would need proper date comparison in production)
                    daily_return = total_return
                
                snapshot = EquityCurve(
                    id=snapshot_id,
                    timestamp=timestamp or datetime.utcnow(),
                    equity=equity,
                    cash=cash,
                    buying_power=buying_power,
                    total_return=total_return,
                    daily_return=daily_return
                )
                
                session.add(snapshot)
                logger.debug(f"Logged equity snapshot: ${equity:.2f}")
            
            return snapshot_id
            
        except Exception as e:
            logger.exception(f"Failed to log equity snapshot: {e}")
            raise
    
    def log_risk_metrics(
        self,
        portfolio_value: Optional[float] = None,
        daily_pnl: float = 0.0,
        total_pnl: float = 0.0,
        max_drawdown: Optional[float] = None,
        sharpe_ratio: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Log risk metrics snapshot.
        
        Args:
            portfolio_value: Current portfolio value
            daily_pnl: Daily profit/loss
            total_pnl: Total profit/loss
            max_drawdown: Maximum drawdown percentage
            sharpe_ratio: Sharpe ratio
            **kwargs: Additional risk metrics
        
        Returns:
            ID of the logged metrics
        """
        metrics_id = self.generate_correlation_id()
        
        try:
            with self.session() as session:
                metrics = RiskMetrics(
                    id=metrics_id,
                    timestamp=datetime.utcnow(),
                    total_positions=kwargs.get('total_positions', 0),
                    total_exposure=kwargs.get('total_exposure', 0.0),
                    max_position_size=kwargs.get('max_position_size'),
                    portfolio_value=portfolio_value,
                    daily_pnl=daily_pnl,
                    total_pnl=total_pnl,
                    max_drawdown=max_drawdown,
                    sharpe_ratio=sharpe_ratio,
                    risk_per_trade=kwargs.get('risk_per_trade'),
                    daily_loss_limit=kwargs.get('daily_loss_limit'),
                    current_daily_loss=kwargs.get('current_daily_loss', 0.0)
                )
                
                session.add(metrics)
                logger.debug(f"Logged risk metrics")
            
            return metrics_id
            
        except Exception as e:
            logger.exception(f"Failed to log risk metrics: {e}")
            raise
    
    def get_trade_history(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[TradeHistory]:
        """
        Query trade history.
        
        Args:
            symbol: Optional symbol filter
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum number of results
        
        Returns:
            List of TradeHistory records (expired from session, access attributes immediately)
        """
        try:
            with self.session() as session:
                query = session.query(TradeHistory)
                
                if symbol:
                    query = query.filter(TradeHistory.symbol == symbol)
                
                if start_date:
                    query = query.filter(TradeHistory.timestamp >= start_date)
                
                if end_date:
                    query = query.filter(TradeHistory.timestamp <= end_date)
                
                query = query.order_by(TradeHistory.timestamp.desc()).limit(limit)
                
                # Expire objects to avoid DetachedInstanceError - access attributes in tests
                results = query.all()
                session.expunge_all()  # Detach objects from session
                return results
                
        except Exception as e:
            logger.exception(f"Failed to query trade history: {e}")
            return []
    
    def get_open_positions_with_strategy(self) -> List[Dict[str, Any]]:
        """
        Get all open positions with their strategy information.
        
        An open position is a BUY order that hasn't been closed by a SELL order.
        This queries TradeHistory for the most recent executed BUY order per symbol
        that hasn't been offset by a SELL order.
        
        Returns:
            List of dictionaries with position info:
            {
                'symbol': str,
                'strategy_name': str,
                'entry_price': float,
                'qty': int,
                'stop_loss': float,
                'timestamp': datetime,
                'correlation_id': str
            }
        """
        try:
            with self.session() as session:
                # Get all executed BUY orders
                buy_orders = session.query(TradeHistory).filter(
                    TradeHistory.action == SignalAction.BUY,
                    TradeHistory.executed == True
                ).order_by(TradeHistory.timestamp.desc()).all()
                
                # Get all executed SELL orders for tracking closed positions
                sell_orders = session.query(TradeHistory).filter(
                    TradeHistory.action == SignalAction.SELL,
                    TradeHistory.executed == True
                ).order_by(TradeHistory.timestamp.desc()).all()
                
                # Calculate net position per symbol
                position_map = {}
                
                # Add all BUY orders
                for buy in buy_orders:
                    if buy.symbol not in position_map:
                        position_map[buy.symbol] = {
                            'symbol': buy.symbol,
                            'strategy_name': buy.strategy_name,
                            'entry_price': buy.fill_price or buy.price,
                            'qty': buy.qty or 0,
                            'stop_loss': buy.stop_loss,
                            'timestamp': buy.timestamp,
                            'correlation_id': buy.correlation_id or buy.id
                        }
                    else:
                        # Add to existing position (multiple buys)
                        existing = position_map[buy.symbol]
                        total_qty = existing['qty'] + (buy.qty or 0)
                        # Weighted average entry price
                        existing_value = existing['entry_price'] * existing['qty']
                        new_value = (buy.fill_price or buy.price) * (buy.qty or 0)
                        position_map[buy.symbol]['entry_price'] = (
                            (existing_value + new_value) / total_qty if total_qty > 0 else existing['entry_price']
                        )
                        position_map[buy.symbol]['qty'] = total_qty
                        # Keep most recent timestamp
                        if buy.timestamp > existing['timestamp']:
                            position_map[buy.symbol]['timestamp'] = buy.timestamp
                
                # Subtract all SELL orders
                for sell in sell_orders:
                    if sell.symbol in position_map:
                        position_map[sell.symbol]['qty'] -= (sell.qty or 0)
                        # If qty goes to zero or below, remove the position
                        if position_map[sell.symbol]['qty'] <= 0:
                            del position_map[sell.symbol]
                
                # Return only positions with positive qty
                open_positions = [
                    pos for pos in position_map.values()
                    if pos['qty'] > 0
                ]
                
                logger.debug(f"Found {len(open_positions)} open positions with strategy info")
                return open_positions
                
        except Exception as e:
            logger.exception(f"Failed to query open positions: {e}")
            return []
    
    def get_equity_curve(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[EquityCurve]:
        """
        Query equity curve data.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
            limit: Maximum number of results
        
        Returns:
            List of EquityCurve records (expired from session, access attributes immediately)
        """
        try:
            with self.session() as session:
                query = session.query(EquityCurve)
                
                if start_date:
                    query = query.filter(EquityCurve.timestamp >= start_date)
                
                if end_date:
                    query = query.filter(EquityCurve.timestamp <= end_date)
                
                query = query.order_by(EquityCurve.timestamp.asc()).limit(limit)
                
                # Expire objects to avoid DetachedInstanceError - access attributes in tests
                results = query.all()
                session.expunge_all()  # Detach objects from session
                return results
                
        except Exception as e:
            logger.exception(f"Failed to query equity curve: {e}")
            return []
    
    def log_backtest_result(
        self,
        strategy_name: str,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        total_return: float,
        sharpe_ratio: float,
        max_drawdown: float,
        win_rate: float,
        total_trades: int,
        avg_trade_return: float,
        passed: bool,
        min_sharpe: float,
        max_drawdown_threshold: float,
        min_win_rate: float,
        parameters: Optional[Dict[str, Any]] = None,
        risk_free_rate: float = 0.04,
        initial_cash: float = 10000.0,
        commission: float = 0.001,
        is_walk_forward: bool = False,
        walk_forward_period: Optional[int] = None
    ) -> str:
        """
        Log backtest result to database.
        
        Args:
            strategy_name: Strategy name
            symbol: Stock symbol
            start_date: Backtest start date
            end_date: Backtest end date
            total_return: Total return percentage
            sharpe_ratio: Sharpe ratio
            max_drawdown: Maximum drawdown percentage
            win_rate: Win rate percentage
            total_trades: Total number of trades
            avg_trade_return: Average trade return
            passed: Whether strategy passed validation
            min_sharpe: Minimum Sharpe threshold used
            max_drawdown_threshold: Maximum drawdown threshold used
            min_win_rate: Minimum win rate threshold used
            parameters: Strategy parameters (dict, will be JSON encoded)
            risk_free_rate: Risk-free rate used
            initial_cash: Initial capital
            commission: Commission rate
            is_walk_forward: Whether this is a walk-forward result
            walk_forward_period: Period number if walk-forward
        
        Returns:
            ID of the logged result
        """
        result_id = self.generate_correlation_id()
        
        try:
            with self.session() as session:
                result = BacktestResults(
                    id=result_id,
                    strategy_name=strategy_name,
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    total_return=total_return,
                    sharpe_ratio=sharpe_ratio,
                    max_drawdown=max_drawdown,
                    win_rate=win_rate,
                    total_trades=total_trades,
                    avg_trade_return=avg_trade_return,
                    passed=passed,
                    min_sharpe=min_sharpe,
                    max_drawdown_threshold=max_drawdown_threshold,
                    min_win_rate=min_win_rate,
                    parameters=json.dumps(parameters) if parameters else None,
                    risk_free_rate=risk_free_rate,
                    initial_cash=initial_cash,
                    commission=commission,
                    is_walk_forward=is_walk_forward,
                    walk_forward_period=walk_forward_period
                )
                
                session.add(result)
                logger.debug(f"Logged backtest result for {strategy_name} on {symbol}")
            
            return result_id
            
        except Exception as e:
            logger.exception(f"Failed to log backtest result: {e}")
            raise
    
    def get_backtest_results(
        self,
        strategy_name: Optional[str] = None,
        symbol: Optional[str] = None,
        passed_only: bool = False,
        limit: int = 100
    ) -> List[BacktestResults]:
        """
        Query backtest results.
        
        Args:
            strategy_name: Optional strategy filter
            symbol: Optional symbol filter
            passed_only: Only return passed results
            limit: Maximum number of results
        
        Returns:
            List of BacktestResults records
        """
        try:
            with self.session() as session:
                query = session.query(BacktestResults)
                
                if strategy_name:
                    query = query.filter(BacktestResults.strategy_name == strategy_name)
                
                if symbol:
                    query = query.filter(BacktestResults.symbol == symbol)
                
                if passed_only:
                    query = query.filter(BacktestResults.passed == True)
                
                query = query.order_by(BacktestResults.created_at.desc()).limit(limit)
                
                results = query.all()
                session.expunge_all()
                return results
                
        except Exception as e:
            logger.exception(f"Failed to query backtest results: {e}")
            return []
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check database health.
        
        Returns:
            Dictionary with health status
        """
        try:
            with self.session() as session:
                # Try a simple query using SQLAlchemy
                from sqlalchemy import text
                session.execute(text("SELECT 1"))
                return {"status": "healthy", "database": "accessible"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

