"""Tests for AuditAgent."""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from agents.audit_agent import AuditAgent
from models.audit import IterationSummary, ExecutionResult, AuditReport
from models.signal import TradingSignal, SignalAction
from config.settings import AppConfig, AnthropicConfig
from utils.exceptions import AgentError


@pytest.fixture
def mock_config():
    """Create a mock config with Anthropic settings."""
    config = Mock(spec=AppConfig)
    config.anthropic = AnthropicConfig(
        api_key="test-key",
        model="claude-3-opus-20240229"
    )
    return config


@pytest.fixture
def sample_signal():
    """Create a sample trading signal."""
    return TradingSignal(
        symbol="AAPL",
        action=SignalAction.BUY,
        strategy_name="TrendFollowing",
        confidence=0.75,
        timestamp=datetime.now(),
        price=150.0
    )


@pytest.fixture
def sample_iteration_summary(sample_signal):
    """Create a sample iteration summary."""
    return IterationSummary(
        iteration_number=1,
        timestamp=datetime.now(),
        symbols_processed=["AAPL", "MSFT"],
        signals_generated=2,
        signals_validated=2,
        signals_approved=1,
        signals_executed=1,
        execution_results=[
            ExecutionResult(
                signal=sample_signal,
                executed=True,
                order_id="order-123",
                fill_price=150.0
            )
        ],
        errors=[],
        duration_seconds=5.5
    )


class TestAuditAgentInitialization:
    """Test AuditAgent initialization."""
    
    def test_init_without_anthropic_config(self):
        """Test that initialization fails without Anthropic config."""
        config = Mock(spec=AppConfig)
        config.anthropic = None
        
        with pytest.raises(AgentError) as exc_info:
            AuditAgent(config=config)
        
        assert "Anthropic configuration not found" in str(exc_info.value)
    
    @patch('agents.audit_agent.Anthropic')
    def test_init_success(self, mock_anthropic, mock_config):
        """Test successful initialization."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        agent = AuditAgent(config=mock_config)
        
        assert agent.claude_client == mock_client
        assert agent.model == "claude-3-opus-20240229"
        mock_anthropic.assert_called_once_with(api_key="test-key")
    
    @patch('agents.audit_agent.Anthropic')
    def test_init_with_default_model(self, mock_anthropic):
        """Test initialization with default model."""
        config = Mock(spec=AppConfig)
        config.anthropic = AnthropicConfig(api_key="test-key", model=None)
        
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        agent = AuditAgent(config=config)
        
        assert agent.model == "claude-3-opus-20240229"  # Default


class TestAuditAgentProcess:
    """Test AuditAgent.process method."""
    
    @patch('agents.audit_agent.Anthropic')
    def test_process_generates_report(self, mock_anthropic, mock_config, sample_iteration_summary):
        """Test that process generates an audit report."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Test narrative summary")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        agent = AuditAgent(config=mock_config)
        report = agent.process(sample_iteration_summary)
        
        assert isinstance(report, AuditReport)
        assert report.report_type == "iteration"
        assert report.summary == "Test narrative summary"
        assert "signals_generated" in report.metrics
        assert report.metrics["signals_generated"] == 2
    
    @patch('agents.audit_agent.Anthropic')
    def test_process_calculates_metrics(self, mock_anthropic, mock_config, sample_iteration_summary):
        """Test that process calculates correct metrics."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Summary")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        agent = AuditAgent(config=mock_config)
        report = agent.process(sample_iteration_summary)
        
        metrics = report.metrics
        assert metrics["signals_generated"] == 2
        assert metrics["signals_validated"] == 2
        assert metrics["signals_approved"] == 1
        assert metrics["signals_executed"] == 1
        assert metrics["approval_rate"] == 0.5  # 1/2
        assert metrics["execution_rate"] == 1.0  # 1/1
        assert metrics["duration_seconds"] == 5.5
    
    @patch('agents.audit_agent.Anthropic')
    def test_process_handles_llm_failure(self, mock_anthropic, mock_config, sample_iteration_summary):
        """Test that process handles LLM failures gracefully."""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("LLM error")
        mock_anthropic.return_value = mock_client
        
        agent = AuditAgent(config=mock_config)
        report = agent.process(sample_iteration_summary)
        
        assert isinstance(report, AuditReport)
        assert "failed" in report.summary.lower()
    
    @patch('agents.audit_agent.Anthropic')
    def test_process_with_execution_results(self, mock_anthropic, mock_config, sample_iteration_summary, sample_signal):
        """Test process with execution results."""
        execution_results = [
            ExecutionResult(signal=sample_signal, executed=True, order_id="order-1"),
            ExecutionResult(signal=sample_signal, executed=False, error="Failed")
        ]
        
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Summary with execution results")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        agent = AuditAgent(config=mock_config)
        report = agent.process(sample_iteration_summary, execution_results)
        
        assert report.metrics["execution_success_rate"] == 0.5
        assert report.metrics["execution_failures"] == 1


class TestAuditAgentReports:
    """Test daily and weekly report generation."""
    
    @patch('agents.audit_agent.Anthropic')
    def test_generate_daily_report(self, mock_anthropic, mock_config):
        """Test daily report generation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Daily summary")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        agent = AuditAgent(config=mock_config)
        
        # Add some iteration summaries
        summary1 = IterationSummary(
            iteration_number=1,
            timestamp=datetime.now(),
            symbols_processed=["AAPL"],
            signals_generated=1,
            signals_validated=1,
            signals_approved=1,
            signals_executed=1,
            errors=[],
            duration_seconds=5.0
        )
        agent._iteration_summaries.append(summary1)
        
        report = agent.generate_daily_report()
        
        assert report.report_type == "daily"
        assert "daily" in report.summary.lower() or "summary" in report.summary.lower()
        assert report.metrics["iterations"] == 1
    
    @patch('agents.audit_agent.Anthropic')
    def test_generate_daily_report_no_activity(self, mock_anthropic, mock_config):
        """Test daily report with no activity."""
        mock_client = Mock()
        mock_anthropic.return_value = mock_client
        
        agent = AuditAgent(config=mock_config)
        report = agent.generate_daily_report()
        
        assert report.report_type == "daily"
        assert "no trading activity" in report.summary.lower()
    
    @patch('agents.audit_agent.Anthropic')
    def test_generate_weekly_report(self, mock_anthropic, mock_config):
        """Test weekly report generation."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Weekly summary")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        agent = AuditAgent(config=mock_config)
        
        # Add summaries from last week
        for i in range(3):
            summary = IterationSummary(
                iteration_number=i+1,
                timestamp=datetime.now() - timedelta(days=i),
                symbols_processed=["AAPL"],
                signals_generated=1,
                signals_validated=1,
                signals_approved=1,
                signals_executed=1,
                errors=[],
                duration_seconds=5.0
            )
            agent._iteration_summaries.append(summary)
        
        report = agent.generate_weekly_report()
        
        assert report.report_type == "weekly"
        assert report.metrics["iterations"] == 3


class TestAuditAgentHealthCheck:
    """Test AuditAgent health check."""
    
    @patch('agents.audit_agent.Anthropic')
    def test_health_check_success(self, mock_anthropic, mock_config):
        """Test successful health check."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="OK")]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        agent = AuditAgent(config=mock_config)
        health = agent.health_check()
        
        assert health["status"] == "healthy"
        assert health["llm_accessible"] is True
        assert health["llm_provider"] == "anthropic"
    
    @patch('agents.audit_agent.Anthropic')
    def test_health_check_failure(self, mock_anthropic, mock_config):
        """Test health check with LLM failure."""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("Connection error")
        mock_anthropic.return_value = mock_client
        
        agent = AuditAgent(config=mock_config)
        health = agent.health_check()
        
        assert health["status"] == "unhealthy"
        assert health["llm_accessible"] is False

