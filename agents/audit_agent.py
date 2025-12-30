"""Audit agent for generating reports and narratives."""
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json

from agents.base import BaseAgent
from models.signal import TradingSignal
from models.audit import IterationSummary, AuditReport, ExecutionResult
from utils.exceptions import AgentError


class AuditAgent(BaseAgent):
    """
    Agent responsible for generating audit reports and narratives.
    
    Role: LLM-first narrative generation for transparency and monetization
    - Generates summaries of trading activity
    - Explains what worked and what failed
    - Provides performance insights
    - Creates weekly/daily reports
    
    Flow: Input logs/results → LLM synthesizes/explains → Output reports
    Focus: Monetization enabler (credible, engaging reports)
    """
    
    def __init__(self, config=None):
        """
        Initialize the audit agent.
        
        Args:
            config: Application configuration. If None, loads from environment.
        """
        super().__init__(config)
        
        # Validate Claude configuration (required for AuditAgent)
        if not self.config.anthropic:
            raise AgentError(
                "Anthropic configuration not found. Set ANTHROPIC_API_KEY in environment.",
                correlation_id=self._correlation_id
            )
        
        # Initialize Claude client
        try:
            from anthropic import Anthropic
            self.claude_client = Anthropic(api_key=self.config.anthropic.api_key)
            self.model = self.config.anthropic.model or "claude-3-haiku-20240307"
            self.log_info(f"AuditAgent initialized with Claude ({self.model})")
        except Exception as e:
            error = self.handle_error(e, context={"provider": "anthropic"})
            raise AgentError(
                "Failed to initialize Claude client",
                correlation_id=self._correlation_id
            ) from error
        
        # Report storage (in production, persist to DB)
        self._iteration_summaries: List[IterationSummary] = []
        self._daily_reports: List[AuditReport] = []
        
        # Initialize database manager (optional)
        self.db_manager = None
        if self.config.database:
            try:
                from utils.database import DatabaseManager
                self.db_manager = DatabaseManager(config=self.config)
                self.log_info("Database persistence enabled")
            except Exception as e:
                self.log_warning(f"Failed to initialize database manager: {e}. Persistence disabled.")
                self.db_manager = None
    
    def process(
        self,
        iteration_summary: IterationSummary,
        execution_results: Optional[List[ExecutionResult]] = None
    ) -> AuditReport:
        """
        Process iteration summary and generate audit report.
        
        This is the main entry point for the agent.
        
        Args:
            iteration_summary: Summary of the iteration
            execution_results: Optional execution results
        
        Returns:
            AuditReport with narrative summary
        """
        self.generate_correlation_id()
        self.log_info(
            f"Generating audit report for iteration {iteration_summary.iteration_number}",
            iteration=iteration_summary.iteration_number
        )
        
        # Store iteration summary
        self._iteration_summaries.append(iteration_summary)
        
        # Generate narrative using Claude
        try:
            narrative = self._generate_narrative(iteration_summary, execution_results)
            
            # Calculate metrics
            metrics = self._calculate_metrics(iteration_summary, execution_results)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(iteration_summary, metrics)
            
            report = AuditReport(
                report_type="iteration",
                timestamp=datetime.now(),
                summary=narrative,
                metrics=metrics,
                recommendations=recommendations
            )
            
            # Persist to database if available
            if self.db_manager:
                try:
                    # Log iteration summary
                    self.db_manager.log_iteration(iteration_summary)
                    
                    # Log audit report
                    self.db_manager.log_audit_report(report)
                    
                    # Log trades from execution results
                    if execution_results:
                        for exec_result in execution_results:
                            if exec_result.signal:
                                self.db_manager.log_trade(
                                    signal=exec_result.signal,
                                    execution_result=exec_result,
                                    correlation_id=self._correlation_id
                                )
                    
                    self.log_debug("Data persisted to database")
                except Exception as e:
                    self.log_warning(f"Failed to persist to database: {e}")
                    # Continue even if persistence fails
            
            self.log_info(f"Audit report generated for iteration {iteration_summary.iteration_number}")
            return report
            
        except Exception as e:
            self.log_exception(f"Failed to generate audit report", e)
            # Return basic report on failure
            return AuditReport(
                report_type="iteration",
                timestamp=datetime.now(),
                summary=f"Audit report generation failed: {str(e)}",
                metrics={}
            )
    
    def _generate_narrative(
        self,
        iteration_summary: IterationSummary,
        execution_results: Optional[List[ExecutionResult]]
    ) -> str:
        """
        Generate narrative summary using Claude LLM.
        
        Args:
            iteration_summary: Iteration summary data
            execution_results: Execution results if available
        
        Returns:
            Narrative text
        """
        # Prepare data summary
        signals_summary = {
            "generated": iteration_summary.signals_generated,
            "validated": iteration_summary.signals_validated,
            "approved": iteration_summary.signals_approved,
            "executed": iteration_summary.signals_executed,
            "symbols": iteration_summary.symbols_processed
        }
        
        execution_summary = []
        if execution_results:
            for result in execution_results:
                execution_summary.append({
                    "symbol": result.signal.symbol,
                    "action": result.signal.action.value,
                    "executed": result.executed,
                    "order_id": result.order_id,
                    "error": result.error
                })
        
        prompt = f"""Generate a professional trading system audit report for iteration {iteration_summary.iteration_number}.

Trading Activity Summary:
- Symbols Processed: {', '.join(iteration_summary.symbols_processed)}
- Signals Generated: {iteration_summary.signals_generated}
- Signals Validated: {iteration_summary.signals_validated}
- Signals Approved: {iteration_summary.signals_approved}
- Signals Executed: {iteration_summary.signals_executed}
- Duration: {iteration_summary.duration_seconds:.2f} seconds

Execution Results:
{json.dumps(execution_summary, indent=2) if execution_summary else "No executions"}

Errors:
{chr(10).join(iteration_summary.errors) if iteration_summary.errors else "No errors"}

Generate a concise, professional narrative (2-3 paragraphs) that:
1. Summarizes the trading activity for this iteration
2. Highlights what worked well
3. Identifies any issues or failures
4. Provides context for the decisions made

Write in a clear, professional tone suitable for stakeholders or clients."""
        
        try:
            message = self.claude_client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            narrative = message.content[0].text if message.content else "Failed to generate narrative"
            self.log_debug("Narrative generated successfully")
            return narrative
            
        except Exception as e:
            self.log_exception("Failed to generate narrative with Claude", e)
            # Fallback to basic summary
            return (
                f"Iteration {iteration_summary.iteration_number} processed "
                f"{iteration_summary.signals_generated} signals across "
                f"{len(iteration_summary.symbols_processed)} symbols. "
                f"{iteration_summary.signals_executed} trades were executed successfully."
            )
    
    def _calculate_metrics(
        self,
        iteration_summary: IterationSummary,
        execution_results: Optional[List[ExecutionResult]]
    ) -> Dict[str, Any]:
        """
        Calculate performance metrics.
        
        Args:
            iteration_summary: Iteration summary
            execution_results: Execution results
        
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "iteration_number": iteration_summary.iteration_number,
            "signals_generated": iteration_summary.signals_generated,
            "signals_validated": iteration_summary.signals_validated,
            "signals_approved": iteration_summary.signals_approved,
            "signals_executed": iteration_summary.signals_executed,
            "approval_rate": (
                iteration_summary.signals_approved / iteration_summary.signals_generated
                if iteration_summary.signals_generated > 0 else 0.0
            ),
            "execution_rate": (
                iteration_summary.signals_executed / iteration_summary.signals_approved
                if iteration_summary.signals_approved > 0 else 0.0
            ),
            "duration_seconds": iteration_summary.duration_seconds,
            "error_count": len(iteration_summary.errors)
        }
        
        if execution_results:
            successful = sum(1 for r in execution_results if r.executed)
            failed = sum(1 for r in execution_results if not r.executed)
            metrics.update({
                "execution_success_rate": (
                    successful / len(execution_results) if execution_results else 0.0
                ),
                "execution_failures": failed
            })
        
        return metrics
    
    def _generate_recommendations(
        self,
        iteration_summary: IterationSummary,
        metrics: Dict[str, Any]
    ) -> Optional[str]:
        """
        Generate recommendations based on metrics.
        
        Args:
            iteration_summary: Iteration summary
            metrics: Calculated metrics
        
        Returns:
            Recommendations text or None
        """
        # Only generate recommendations if there are issues
        if metrics.get("error_count", 0) > 0 or metrics.get("approval_rate", 1.0) < 0.5:
            prompt = f"""Based on these trading system metrics, provide 2-3 actionable recommendations:

Metrics:
- Approval Rate: {metrics.get('approval_rate', 0):.2%}
- Execution Rate: {metrics.get('execution_rate', 0):.2%}
- Error Count: {metrics.get('error_count', 0)}
- Signals Generated: {metrics.get('signals_generated', 0)}
- Signals Approved: {metrics.get('signals_approved', 0)}

Provide brief, actionable recommendations (2-3 bullet points) to improve system performance."""
            
            try:
                message = self.claude_client.messages.create(
                    model=self.model,
                    max_tokens=300,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                recommendations = message.content[0].text if message.content else None
                return recommendations
            except Exception as e:
                self.log_debug(f"Failed to generate recommendations: {e}")
        
        return None
    
    def generate_daily_report(self) -> AuditReport:
        """
        Generate daily summary report from all iterations.
        
        Returns:
            Daily audit report
        """
        self.log_info("Generating daily audit report")
        
        # Get today's iterations
        today = datetime.now().date()
        today_summaries = [
            s for s in self._iteration_summaries
            if s.timestamp.date() == today
        ]
        
        if not today_summaries:
            return AuditReport(
                report_type="daily",
                timestamp=datetime.now(),
                summary="No trading activity today.",
                metrics={}
            )
        
        # Aggregate metrics
        total_signals = sum(s.signals_generated for s in today_summaries)
        total_executed = sum(s.signals_executed for s in today_summaries)
        total_errors = sum(len(s.errors) for s in today_summaries)
        
        prompt = f"""Generate a daily trading system summary report.

Daily Activity:
- Iterations: {len(today_summaries)}
- Total Signals Generated: {total_signals}
- Total Trades Executed: {total_executed}
- Total Errors: {total_errors}

Generate a professional daily summary (3-4 paragraphs) covering:
1. Overall trading activity and performance
2. Key successes and achievements
3. Challenges or issues encountered
4. System health and reliability

Write for stakeholders or clients."""
        
        try:
            message = self.claude_client.messages.create(
                model=self.model,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            summary = message.content[0].text if message.content else "Failed to generate daily summary"
            
            return AuditReport(
                report_type="daily",
                timestamp=datetime.now(),
                summary=summary,
                metrics={
                    "iterations": len(today_summaries),
                    "total_signals": total_signals,
                    "total_executed": total_executed,
                    "total_errors": total_errors
                }
            )
        except Exception as e:
            self.log_exception("Failed to generate daily report", e)
            return AuditReport(
                report_type="daily",
                timestamp=datetime.now(),
                summary=f"Daily report generation failed: {str(e)}",
                metrics={}
            )
    
    def generate_weekly_report(self) -> AuditReport:
        """
        Generate weekly summary report.
        
        Returns:
            Weekly audit report
        """
        self.log_info("Generating weekly audit report")
        
        # Get last 7 days of iterations
        week_ago = datetime.now() - timedelta(days=7)
        week_summaries = [
            s for s in self._iteration_summaries
            if s.timestamp >= week_ago
        ]
        
        if not week_summaries:
            return AuditReport(
                report_type="weekly",
                timestamp=datetime.now(),
                summary="No trading activity this week.",
                metrics={}
            )
        
        # Aggregate metrics
        total_signals = sum(s.signals_generated for s in week_summaries)
        total_executed = sum(s.signals_executed for s in week_summaries)
        avg_approval_rate = (
            sum(s.signals_approved / s.signals_generated if s.signals_generated > 0 else 0
                for s in week_summaries) / len(week_summaries)
        )
        
        prompt = f"""Generate a weekly trading system performance report.

Weekly Activity (Last 7 Days):
- Total Iterations: {len(week_summaries)}
- Total Signals Generated: {total_signals}
- Total Trades Executed: {total_executed}
- Average Approval Rate: {avg_approval_rate:.2%}

Generate a comprehensive weekly report (4-5 paragraphs) covering:
1. Weekly performance overview
2. Strategy effectiveness and signal quality
3. System reliability and uptime
4. Risk management effectiveness
5. Recommendations for next week

Write in a professional, engaging style suitable for clients or investors."""
        
        try:
            message = self.claude_client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            summary = message.content[0].text if message.content else "Failed to generate weekly summary"
            
            return AuditReport(
                report_type="weekly",
                timestamp=datetime.now(),
                summary=summary,
                metrics={
                    "iterations": len(week_summaries),
                    "total_signals": total_signals,
                    "total_executed": total_executed,
                    "avg_approval_rate": avg_approval_rate
                }
            )
        except Exception as e:
            self.log_exception("Failed to generate weekly report", e)
            return AuditReport(
                report_type="weekly",
                timestamp=datetime.now(),
                summary=f"Weekly report generation failed: {str(e)}",
                metrics={}
            )
    
    def health_check(self) -> Dict[str, Any]:
        """Check agent health by testing Claude connection."""
        health = super().health_check()
        
        try:
            # Simple test call to Claude
            test_message = self.claude_client.messages.create(
                model=self.model,
                max_tokens=10,
                messages=[{"role": "user", "content": "Say 'OK'"}]
            )
            
            health.update({
                "llm_provider": "anthropic",
                "model": self.model,
                "llm_accessible": True,
                "stored_iterations": len(self._iteration_summaries)
            })
        except Exception as e:
            health.update({
                "status": "unhealthy",
                "llm_provider": "anthropic",
                "llm_accessible": False,
                "error": str(e)
            })
        
        return health

