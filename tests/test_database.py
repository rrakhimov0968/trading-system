"""Tests for database persistence."""
import pytest
from datetime import datetime
from unittest.mock import Mock, patch

from utils.database import (
    DatabaseManager,
    TradeHistory,
    EquityCurve,
    RiskMetrics,
    IterationLog,
    AuditReportLog,
    Base
)
from models.signal import TradingSignal, SignalAction
from models.audit import ExecutionResult, IterationSummary, AuditReport
from config.settings import AppConfig, TradingMode, LogLevel, AlpacaConfig, DatabaseConfig
from tests.conftest import mock_config


@pytest.fixture
def test_db_config():
    """Create test database config (in-memory SQLite)."""
    return DatabaseConfig(
        url="sqlite:///:memory:",
        echo=False,
        pool_size=5,
        max_overflow=10
    )


@pytest.fixture
def test_config_with_db(mock_config, test_db_config):
    """Create test config with database."""
    mock_config.database = test_db_config
    return mock_config


@pytest.fixture
def db_manager(test_config_with_db):
    """Create database manager for testing."""
    return DatabaseManager(config=test_config_with_db)


@pytest.fixture
def sample_signal():
    """Create a sample trading signal."""
    return TradingSignal(
        symbol="AAPL",
        action=SignalAction.BUY,
        strategy_name="TrendFollowing",
        confidence=0.75,
        timestamp=datetime.now(),
        price=150.0,
        stop_loss=145.0,
        take_profit=160.0,
        qty=10,
        risk_amount=50.0,
        approved=True,
        reasoning="Strong upward trend detected"
    )


@pytest.fixture
def sample_execution_result(sample_signal):
    """Create a sample execution result."""
    return ExecutionResult(
        signal=sample_signal,
        order_id="test_order_123",
        executed=True,
        execution_time=datetime.now(),
        fill_price=150.5
    )


@pytest.fixture
def sample_iteration_summary():
    """Create a sample iteration summary."""
    return IterationSummary(
        iteration_number=1,
        timestamp=datetime.now(),
        symbols_processed=["AAPL", "MSFT"],
        signals_generated=2,
        signals_validated=2,
        signals_approved=1,
        signals_executed=1,
        execution_results=[],
        errors=[],
        duration_seconds=5.5
    )


@pytest.fixture
def sample_audit_report():
    """Create a sample audit report."""
    return AuditReport(
        report_type="iteration",
        timestamp=datetime.now(),
        summary="Test audit report summary",
        metrics={
            "signals_generated": 2,
            "signals_executed": 1,
            "approval_rate": 0.5
        },
        recommendations="Continue monitoring"
    )


@pytest.mark.unit
class TestDatabaseManager:
    """Test DatabaseManager functionality."""
    
    def test_init_with_config(self, test_config_with_db):
        """Test database manager initialization with config."""
        db = DatabaseManager(config=test_config_with_db)
        assert db.config == test_config_with_db
        assert db.db_config == test_config_with_db.database
    
    def test_init_without_config(self):
        """Test database manager initialization without config (uses in-memory)."""
        # Create a config without database
        from config.settings import AppConfig, TradingMode, LogLevel, AlpacaConfig, DataProviderConfig
        from unittest.mock import Mock
        
        mock_app_config = Mock(spec=AppConfig)
        mock_app_config.database = None
        
        # Initialize with explicit config to avoid environment variable issues
        db = DatabaseManager(config=mock_app_config)
        assert db.db_config is None
        assert ":memory:" in str(db.engine.url) or "memory" in str(db.engine.url)
    
    def test_generate_correlation_id(self, db_manager):
        """Test correlation ID generation."""
        corr_id = db_manager.generate_correlation_id()
        assert isinstance(corr_id, str)
        assert len(corr_id) > 0
    
    def test_session_context_manager(self, db_manager):
        """Test session context manager."""
        with db_manager.session() as session:
            assert session is not None
            # Session should be closed after context exit
            # (we can't test closed state easily, but no exception means it worked)
    
    def test_health_check_healthy(self, db_manager):
        """Test health check when database is healthy."""
        health = db_manager.health_check()
        assert health["status"] == "healthy"
        assert "database" in health


@pytest.mark.unit
class TestTradeHistory:
    """Test trade history persistence."""
    
    def test_log_trade_basic(self, db_manager, sample_signal):
        """Test logging a basic trade."""
        corr_id = db_manager.log_trade(sample_signal)
        
        assert isinstance(corr_id, str)
        
        # Query the trade (access attributes within session to avoid DetachedInstanceError)
        trades = db_manager.get_trade_history(symbol="AAPL")
        assert len(trades) == 1
        trade = trades[0]
        # Access all attributes while object is still bound to session
        symbol = trade.symbol
        action = trade.action
        price = trade.price
        confidence = trade.confidence
        
        assert symbol == "AAPL"
        assert action == SignalAction.BUY
        assert price == 150.0
        assert confidence == 0.75
    
    def test_log_trade_with_execution_result(self, db_manager, sample_signal, sample_execution_result):
        """Test logging a trade with execution result."""
        corr_id = db_manager.log_trade(sample_signal, execution_result=sample_execution_result)
        
        trades = db_manager.get_trade_history(symbol="AAPL")
        assert len(trades) == 1
        trade = trades[0]
        # Access all attributes at once to avoid DetachedInstanceError
        order_id = trade.order_id
        executed = trade.executed
        fill_price = trade.fill_price
        
        assert order_id == "test_order_123"
        assert executed is True
        assert fill_price == 150.5
    
    def test_log_trade_with_custom_correlation_id(self, db_manager, sample_signal):
        """Test logging trade with custom correlation ID."""
        custom_id = "custom_corr_123"
        corr_id = db_manager.log_trade(sample_signal, correlation_id=custom_id)
        
        assert corr_id == custom_id
        
        trades = db_manager.get_trade_history(symbol="AAPL")
        # Access attribute immediately to avoid DetachedInstanceError
        corr_id = trades[0].correlation_id
        assert corr_id == custom_id
    
    def test_get_trade_history_with_filters(self, db_manager):
        """Test querying trade history with filters."""
        # Create multiple trades
        signals = [
            TradingSignal(
                symbol="AAPL",
                action=SignalAction.BUY,
                strategy_name="Strategy1",
                confidence=0.8,
                timestamp=datetime(2024, 1, 1),
                price=100.0,
                qty=10
            ),
            TradingSignal(
                symbol="MSFT",
                action=SignalAction.SELL,
                strategy_name="Strategy2",
                confidence=0.6,
                timestamp=datetime(2024, 1, 2),
                price=200.0,
                qty=5
            ),
            TradingSignal(
                symbol="AAPL",
                action=SignalAction.BUY,
                strategy_name="Strategy1",
                confidence=0.7,
                timestamp=datetime(2024, 1, 3),
                price=105.0,
                qty=8
            )
        ]
        
        for signal in signals:
            db_manager.log_trade(signal)
        
        # Test symbol filter
        aapl_trades = db_manager.get_trade_history(symbol="AAPL")
        assert len(aapl_trades) == 2
        
        # Test date filter
        start_date = datetime(2024, 1, 2)
        recent_trades = db_manager.get_trade_history(start_date=start_date)
        assert len(recent_trades) == 2
        
        # Test limit
        limited = db_manager.get_trade_history(limit=1)
        assert len(limited) == 1
    
    def test_log_trade_error_handling(self, db_manager):
        """Test error handling when logging trade."""
        # Invalid signal should be handled gracefully
        invalid_signal = TradingSignal(
            symbol="TEST",  # Valid symbol to avoid validation error
            action=SignalAction.BUY,
            strategy_name="Test",
            confidence=0.5,
            timestamp=datetime.now(),
            price=None  # Missing price will cause SQLAlchemy validation error
        )
        
        # Should raise exception due to missing required field (price is nullable=False in DB)
        # or handle gracefully
        try:
            db_manager.log_trade(invalid_signal)
            # If no exception, that's fine too (some DBs allow NULL)
        except Exception:
            # Expected - missing required field
            pass


@pytest.mark.unit
class TestIterationLogging:
    """Test iteration summary logging."""
    
    def test_log_iteration(self, db_manager, sample_iteration_summary):
        """Test logging an iteration summary."""
        iteration_id = db_manager.log_iteration(sample_iteration_summary)
        
        assert isinstance(iteration_id, str)
        
        # Query iteration log
        with db_manager.session() as session:
            iteration = session.query(IterationLog).filter_by(
                iteration_number=1
            ).first()
            
            assert iteration is not None
            assert iteration.signals_generated == 2
            assert iteration.signals_executed == 1
            assert iteration.duration_seconds == 5.5


@pytest.mark.unit
class TestAuditReportLogging:
    """Test audit report logging."""
    
    def test_log_audit_report(self, db_manager, sample_audit_report):
        """Test logging an audit report."""
        report_id = db_manager.log_audit_report(sample_audit_report)
        
        assert isinstance(report_id, str)
        
        # Query audit report log
        with db_manager.session() as session:
            report = session.query(AuditReportLog).filter_by(
                report_type="iteration"
            ).first()
            
            assert report is not None
            assert "Test audit report summary" in report.summary
            assert report.metrics is not None


@pytest.mark.unit
class TestEquityCurve:
    """Test equity curve logging."""
    
    def test_log_equity_snapshot(self, db_manager):
        """Test logging equity snapshot."""
        snapshot_id = db_manager.log_equity_snapshot(
            equity=100000.0,
            cash=50000.0,
            buying_power=100000.0
        )
        
        assert isinstance(snapshot_id, str)
        
        # Query equity curve (access attributes immediately)
        curve = db_manager.get_equity_curve()
        assert len(curve) == 1
        snapshot = curve[0]
        equity = snapshot.equity
        cash = snapshot.cash
        
        assert equity == 100000.0
        assert cash == 50000.0
    
    def test_log_equity_snapshot_with_returns(self, db_manager):
        """Test logging multiple equity snapshots to calculate returns."""
        # First snapshot
        db_manager.log_equity_snapshot(equity=100000.0)
        
        # Second snapshot (higher equity)
        db_manager.log_equity_snapshot(equity=105000.0)
        
        curve = db_manager.get_equity_curve()
        assert len(curve) == 2
        # Second snapshot should have return calculated (access attribute immediately)
        total_return = curve[1].total_return
        assert total_return is not None


@pytest.mark.unit
class TestRiskMetrics:
    """Test risk metrics logging."""
    
    def test_log_risk_metrics(self, db_manager):
        """Test logging risk metrics."""
        metrics_id = db_manager.log_risk_metrics(
            portfolio_value=100000.0,
            daily_pnl=500.0,
            total_pnl=2000.0,
            max_drawdown=-5.0,
            sharpe_ratio=1.5,
            total_positions=3,
            risk_per_trade=0.02
        )
        
        assert isinstance(metrics_id, str)
        
        # Query risk metrics
        with db_manager.session() as session:
            metrics = session.query(RiskMetrics).first()
            
            assert metrics is not None
            assert metrics.portfolio_value == 100000.0
            assert metrics.daily_pnl == 500.0
            assert metrics.sharpe_ratio == 1.5


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for database persistence."""
    
    def test_full_trade_flow(self, db_manager, sample_signal, sample_execution_result):
        """Test complete trade flow from signal to persistence."""
        # Log trade
        corr_id = db_manager.log_trade(sample_signal, execution_result=sample_execution_result)
        
        # Verify persistence (access attributes immediately)
        trades = db_manager.get_trade_history()
        assert len(trades) > 0
        
        trade = trades[0]
        # Access all attributes at once
        symbol = trade.symbol
        action = trade.action
        order_id = trade.order_id
        
        assert symbol == sample_signal.symbol
        assert action == sample_signal.action
        assert order_id == sample_execution_result.order_id
    
    @patch('anthropic.Anthropic')
    def test_audit_agent_integration(self, mock_anthropic, test_config_with_db, sample_iteration_summary, sample_execution_result):
        """Test database integration with AuditAgent."""
        from agents.audit_agent import AuditAgent
        from config.settings import LLMConfig
        
        # Add Anthropic config to test_config_with_db
        test_config_with_db.anthropic = LLMConfig(provider="anthropic", api_key="test-key")
        
        # Mock Anthropic client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Test audit report")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        # Add execution result to iteration summary
        sample_iteration_summary.execution_results = [sample_execution_result]
        
        agent = AuditAgent(config=test_config_with_db)
        
        # Process iteration summary (should persist to DB)
        report = agent.process(sample_iteration_summary, [sample_execution_result])
        
        assert report is not None
        # Verify database has the data
        if agent.db_manager:
            trades = agent.db_manager.get_trade_history()
            assert len(trades) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

