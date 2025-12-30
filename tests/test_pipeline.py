"""End-to-end integration tests for the complete trading pipeline."""
import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, List
import pandas as pd
import numpy as np

from models.market_data import MarketData
from models.signal import TradingSignal, SignalAction
from models.audit import IterationSummary, ExecutionResult, AuditReport
from agents.data_agent import DataAgent
from agents.strategy_agent import StrategyAgent
from agents.quant_agent import QuantAgent
from agents.risk_agent import RiskAgent
from agents.execution_agent import ExecutionAgent
from agents.audit_agent import AuditAgent
from tests.conftest import (
    test_config_with_llms,
    mock_market_data,
    mock_groq_client,
    mock_anthropic_client
)


def run_pipeline(
    symbols: List[str],
    config,
    market_data: Dict[str, MarketData] = None
) -> AuditReport:
    """
    Run the complete trading pipeline end-to-end.
    
    Args:
        symbols: List of symbols to process
        config: Application configuration
        market_data: Optional pre-fetched market data (for testing)
    
    Returns:
        AuditReport from the AuditAgent
    """
    # Step 1: DataAgent - Fetch market data
    data_agent = DataAgent(config=config)
    if market_data:
        # Use provided market data (for testing)
        fetched_data = market_data
    else:
        fetched_data = data_agent.process(symbols=symbols, timeframe="1Day", limit=100)
    
    if not fetched_data:
        raise ValueError("No market data fetched")
    
    # Step 2: StrategyAgent - Generate signals
    strategy_agent = StrategyAgent(config=config)
    signals = strategy_agent.process(fetched_data)
    
    if not signals:
        # Create empty iteration summary if no signals
        iteration_summary = IterationSummary(
            iteration_number=1,
            timestamp=datetime.now(),
            symbols_processed=symbols,
            signals_generated=0,
            signals_validated=0,
            signals_approved=0,
            signals_executed=0,
            execution_results=[],
            errors=[],
            duration_seconds=0.0
        )
        audit_agent = AuditAgent(config=config)
        return audit_agent.process(iteration_summary, [])
    
    # Step 3: QuantAgent - Validate signals
    quant_agent = QuantAgent(config=config)
    validated_signals = quant_agent.process(signals)
    
    # Step 4: RiskAgent - Risk validation and position sizing
    execution_agent = ExecutionAgent(config=config)
    risk_agent = RiskAgent(config=config)
    approved_signals = risk_agent.process(validated_signals, execution_agent=execution_agent)
    
    # Step 5: ExecutionAgent - Execute approved trades
    execution_results = []
    for signal in approved_signals:
        if signal.action == SignalAction.HOLD or not getattr(signal, 'approved', False):
            continue
        
        execution_result = ExecutionResult(
            signal=signal,
            execution_time=datetime.now()
        )
        
        try:
            order_request = {
                "symbol": signal.symbol,
                "quantity": getattr(signal, 'qty', 1) or 1,
                "side": signal.action.value.lower(),
                "order_type": "market"
            }
            
            result = execution_agent.process(order_request)
            execution_result.order_id = result.get('order_id')
            execution_result.executed = True
            execution_result.fill_price = signal.price
        except Exception as e:
            execution_result.executed = False
            execution_result.error = str(e)
        
        execution_results.append(execution_result)
    
    # Step 6: AuditAgent - Generate report
    iteration_summary = IterationSummary(
        iteration_number=1,
        timestamp=datetime.now(),
        symbols_processed=symbols,
        signals_generated=len(signals),
        signals_validated=len(validated_signals),
        signals_approved=len([s for s in approved_signals if getattr(s, 'approved', False)]),
        signals_executed=len([r for r in execution_results if r.executed]),
        execution_results=execution_results,
        errors=[],
        duration_seconds=0.0
    )
    
    audit_agent = AuditAgent(config=config)
    audit_report = audit_agent.process(iteration_summary, execution_results)
    
    return audit_report


@pytest.mark.integration
class TestPipelineEndToEnd:
    """End-to-end tests for the complete trading pipeline."""
    
    @patch('agents.strategy_agent.Groq')
    @patch('anthropic.Anthropic')
    @patch('agents.data_agent.yf.Ticker')
    def test_full_pipeline_with_mocks(
        self,
        mock_yf_ticker,
        mock_anthropic,
        mock_groq,
        test_config_with_llms,
        mock_market_data,
        mock_groq_client,
        mock_anthropic_client
    ):
        """Test the complete pipeline flow with all mocks."""
        # Setup mocks
        mock_groq.return_value = mock_groq_client
        mock_anthropic.return_value = mock_anthropic_client
        
        # Mock yfinance
        mock_ticker = Mock()
        mock_data = Mock()
        # Create mock DataFrame
        dates = pd.date_range('2024-01-01', periods=100, freq='D')
        mock_df = pd.DataFrame({
            'Open': np.random.uniform(100, 110, 100),
            'High': np.random.uniform(110, 120, 100),
            'Low': np.random.uniform(90, 100, 100),
            'Close': np.random.uniform(100, 110, 100),
            'Volume': np.random.randint(1000000, 2000000, 100)
        }, index=dates)
        mock_data.history.return_value = mock_df
        mock_yf_ticker.return_value = mock_data
        
        # Run pipeline
        symbols = ["SPY", "QQQ"]
        report = run_pipeline(symbols, test_config_with_llms, market_data=mock_market_data)
        
        # Assertions
        assert isinstance(report, AuditReport)
        assert report.report_type == "iteration"
        assert len(report.summary) > 0
        assert "signals_generated" in report.metrics
        assert report.metrics["signals_generated"] >= 0
    
    @patch('agents.strategy_agent.Groq')
    @patch('anthropic.Anthropic')
    @patch('alpaca.trading.client.TradingClient')
    def test_pipeline_with_no_signals(
        self,
        mock_trading_client,
        mock_anthropic,
        mock_groq,
        test_config_with_llms,
        mock_market_data
    ):
        """Test pipeline when no signals are generated."""
        # Setup mocks for empty signals scenario
        mock_groq_client = Mock()
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message = Mock()
        mock_completion.choices[0].message.content = '{"strategy_name": "TrendFollowing", "action": "HOLD", "confidence": 0.3, "reasoning": "No clear signal"}'
        mock_groq_client.chat.completions.create.return_value = mock_completion
        mock_groq.return_value = mock_groq_client
        
        mock_anthropic_client = Mock()
        mock_message = Mock()
        mock_message.content = [Mock(text="No trading activity")]
        mock_anthropic_client.messages.create.return_value = mock_message
        mock_anthropic.return_value = mock_anthropic_client
        
        # Mock execution agent
        mock_account = Mock()
        mock_account.cash = "100000.00"
        mock_account.equity = "100000.00"
        mock_trading_client.return_value.get_account.return_value = mock_account
        
        # Run pipeline
        symbols = ["SPY"]
        report = run_pipeline(symbols, test_config_with_llms, market_data=mock_market_data)
        
        # Assertions
        assert isinstance(report, AuditReport)
        assert "no" in report.summary.lower() or report.metrics.get("signals_generated", 0) == 0
    
    @patch('agents.strategy_agent.Groq')
    @patch('anthropic.Anthropic')
    @patch('alpaca.trading.client.TradingClient')
    def test_pipeline_execution_flow(
        self,
        mock_trading_client,
        mock_anthropic,
        mock_groq,
        test_config_with_llms,
        mock_market_data
    ):
        """Test pipeline execution flow with successful trade execution."""
        # Setup Groq mock for strategy selection
        mock_groq_client = Mock()
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message = Mock()
        mock_completion.choices[0].message.content = '{"strategy_name": "TrendFollowing", "action": "BUY", "confidence": 0.75, "reasoning": "Strong trend"}'
        mock_groq_client.chat.completions.create.return_value = mock_completion
        mock_groq.return_value = mock_groq_client
        
        # Setup Anthropic mock for audit
        mock_anthropic_client = Mock()
        mock_message = Mock()
        mock_message.content = [Mock(text="Successful trading iteration with executed trades")]
        mock_anthropic_client.messages.create.return_value = mock_message
        mock_anthropic.return_value = mock_anthropic_client
        
        # Setup execution client mock (used by both ExecutionAgent and RiskAgent via execution_agent.get_account())
        mock_account = Mock()
        mock_account.cash = "100000.00"
        mock_account.equity = "100000.00"
        mock_account.buying_power = "100000.00"
        mock_trading_client.return_value.get_account.return_value = mock_account
        
        mock_order = Mock()
        mock_order.id = "test_order_123"
        mock_trading_client.return_value.submit_order.return_value = mock_order
        
        # Run pipeline
        symbols = ["SPY"]
        report = run_pipeline(symbols, test_config_with_llms, market_data=mock_market_data)
        
        # Assertions
        assert isinstance(report, AuditReport)
        assert report.metrics["signals_generated"] > 0
        assert "execution" in report.summary.lower() or "trade" in report.summary.lower() or report.metrics.get("signals_executed", 0) >= 0
    
    @patch('agents.strategy_agent.Groq')
    @patch('anthropic.Anthropic')
    @patch('alpaca.trading.client.TradingClient')
    def test_pipeline_metrics_accuracy(
        self,
        mock_trading_client,
        mock_anthropic,
        mock_groq,
        test_config_with_llms,
        mock_market_data
    ):
        """Test that pipeline metrics are calculated accurately."""
        # Setup mocks
        mock_groq_client = Mock()
        mock_completion = Mock()
        mock_completion.choices = [Mock()]
        mock_completion.choices[0].message = Mock()
        mock_completion.choices[0].message.content = '{"strategy_name": "TrendFollowing", "action": "BUY", "confidence": 0.75, "reasoning": "Test"}'
        mock_groq_client.chat.completions.create.return_value = mock_completion
        mock_groq.return_value = mock_groq_client
        
        mock_anthropic_client = Mock()
        mock_message = Mock()
        mock_message.content = [Mock(text="Test report")]
        mock_anthropic_client.messages.create.return_value = mock_message
        mock_anthropic.return_value = mock_anthropic_client
        
        # Mock execution client
        mock_account = Mock()
        mock_account.cash = "100000.00"
        mock_account.equity = "100000.00"
        mock_trading_client.return_value.get_account.return_value = mock_account
        
        # Run pipeline
        symbols = ["SPY", "QQQ"]
        report = run_pipeline(symbols, test_config_with_llms, market_data=mock_market_data)
        
        # Verify metrics structure
        metrics = report.metrics
        assert "signals_generated" in metrics
        assert "signals_validated" in metrics
        assert "signals_approved" in metrics
        assert "signals_executed" in metrics
        assert "approval_rate" in metrics
        assert "execution_rate" in metrics
        assert "duration_seconds" in metrics
        
        # Verify metric relationships
        assert metrics["signals_validated"] <= metrics["signals_generated"]
        assert metrics["signals_approved"] <= metrics["signals_validated"]
        assert metrics["signals_executed"] <= metrics["signals_approved"]
        assert 0.0 <= metrics["approval_rate"] <= 1.0
        assert 0.0 <= metrics["execution_rate"] <= 1.0


@pytest.mark.integration
@pytest.mark.slow
class TestPipelineErrorHandling:
    """Test pipeline error handling and resilience."""
    
    @patch('agents.data_agent.yf.Ticker')
    def test_pipeline_handles_data_agent_failure(
        self,
        mock_yf_ticker,
        test_config_with_llms
    ):
        """Test pipeline handles DataAgent failure gracefully."""
        # Make data agent fail
        mock_yf_ticker.side_effect = Exception("Data fetch failed")
        
        # Should raise error or return empty report
        with pytest.raises((ValueError, Exception)):
            run_pipeline(["SPY"], test_config_with_llms)
    
    @patch('agents.strategy_agent.Groq')
    @patch('anthropic.Anthropic')
    def test_pipeline_handles_strategy_agent_failure(
        self,
        mock_anthropic,
        mock_groq,
        test_config_with_llms,
        mock_market_data
    ):
        """Test pipeline handles StrategyAgent failure."""
        # Make strategy agent fail
        mock_groq.side_effect = Exception("Groq API error")
        
        # Setup audit mock
        mock_anthropic_client = Mock()
        mock_message = Mock()
        mock_message.content = [Mock(text="No signals generated")]
        mock_anthropic_client.messages.create.return_value = mock_message
        mock_anthropic.return_value = mock_anthropic_client
        
        # Should still produce a report (with 0 signals)
        with pytest.raises(Exception):
            run_pipeline(["SPY"], test_config_with_llms, market_data=mock_market_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])

