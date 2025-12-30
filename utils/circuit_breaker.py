"""Circuit breaker for trading system protection."""
from enum import Enum
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Circuit open, stop trading
    HALF_OPEN = "half_open"  # Testing if issue resolved


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    max_failures: int = 5  # Number of consecutive failures before opening
    timeout_seconds: int = 300  # Time to wait before half-open (5 minutes)
    half_open_max_attempts: int = 3  # Max attempts in half-open state
    data_quality_threshold: float = 0.8  # Minimum data quality score (0-1)
    equity_drop_threshold: float = 0.10  # 10% equity drop triggers circuit


@dataclass
class CircuitBreakerMetrics:
    """Metrics tracked by circuit breaker."""
    consecutive_failures: int = 0
    llm_failures: int = 0
    data_quality_failures: int = 0
    equity_drop_detected: bool = False
    last_failure_time: Optional[datetime] = None
    state: CircuitState = CircuitState.CLOSED
    opened_at: Optional[datetime] = None
    half_open_attempts: int = 0


class CircuitBreaker:
    """
    Circuit breaker to protect trading system from cascading failures.
    
    Monitors:
    1. Consecutive LLM failures
    2. Data quality degradation
    3. Account equity drops
    """
    
    def __init__(self, config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker.
        
        Args:
            config: Circuit breaker configuration
        """
        self.config = config or CircuitBreakerConfig()
        self.metrics = CircuitBreakerMetrics()
        
        # Track account equity for drop detection
        self._initial_equity: Optional[float] = None
        self._last_equity: Optional[float] = None
        
    def record_llm_failure(self) -> None:
        """Record an LLM failure."""
        self.metrics.llm_failures += 1
        self.metrics.consecutive_failures += 1
        self.metrics.last_failure_time = datetime.now()
        
        logger.warning(
            f"LLM failure recorded. Consecutive failures: {self.metrics.consecutive_failures}"
        )
        
        if self.metrics.consecutive_failures >= self.config.max_failures:
            self._open_circuit("LLM failures", self.metrics.consecutive_failures)
    
    def record_llm_success(self) -> None:
        """Record an LLM success (resets failure count)."""
        if self.metrics.consecutive_failures > 0:
            logger.info(
                f"LLM success - resetting failure count from {self.metrics.consecutive_failures}"
            )
        
        self.metrics.consecutive_failures = 0
        
        # If in half-open, increment attempts
        if self.metrics.state == CircuitState.HALF_OPEN:
            self.metrics.half_open_attempts += 1
            if self.metrics.half_open_attempts >= self.config.half_open_max_attempts:
                logger.info("Half-open attempts successful, closing circuit")
                self._close_circuit()
    
    def record_data_quality(self, quality_score: float) -> None:
        """
        Record data quality score.
        
        Args:
            quality_score: Data quality score (0.0 to 1.0)
        """
        if quality_score < self.config.data_quality_threshold:
            self.metrics.data_quality_failures += 1
            self.metrics.consecutive_failures += 1
            self.metrics.last_failure_time = datetime.now()
            
            logger.warning(
                f"Data quality below threshold: {quality_score:.2f} "
                f"(threshold: {self.config.data_quality_threshold:.2f})"
            )
            
            if self.metrics.consecutive_failures >= self.config.max_failures:
                self._open_circuit(
                    "Data quality",
                    f"quality_score={quality_score:.2f}"
                )
        else:
            # Good data quality - reset if we were counting failures
            if self.metrics.data_quality_failures > 0:
                logger.info("Data quality recovered")
                self.metrics.data_quality_failures = 0
                if self.metrics.consecutive_failures > 0:
                    self.metrics.consecutive_failures = max(0, self.metrics.consecutive_failures - 1)
    
    def check_equity_drop(self, current_equity: float) -> bool:
        """
        Check if equity has dropped significantly.
        
        Args:
            current_equity: Current account equity
        
        Returns:
            True if equity drop detected and circuit should open
        """
        if self._initial_equity is None:
            self._initial_equity = current_equity
            self._last_equity = current_equity
            return False
        
        self._last_equity = current_equity
        
        if self._initial_equity <= 0:
            return False
        
        drop_pct = (self._initial_equity - current_equity) / self._initial_equity
        
        if drop_pct >= self.config.equity_drop_threshold:
            if not self.metrics.equity_drop_detected:
                logger.error(
                    f"Equity drop detected: {drop_pct:.2%} "
                    f"(from ${self._initial_equity:.2f} to ${current_equity:.2f})"
                )
                self.metrics.equity_drop_detected = True
                self._open_circuit("Equity drop", f"{drop_pct:.2%}")
            return True
        
        # Reset if equity recovers
        if drop_pct < self.config.equity_drop_threshold * 0.5:
            if self.metrics.equity_drop_detected:
                logger.info("Equity recovered, resetting drop detection")
                self.metrics.equity_drop_detected = False
        
        return False
    
    def reset_equity_baseline(self, new_equity: float) -> None:
        """Reset equity baseline (e.g., after account funding)."""
        logger.info(f"Resetting equity baseline to ${new_equity:.2f}")
        self._initial_equity = new_equity
        self._last_equity = new_equity
        self.metrics.equity_drop_detected = False
    
    def is_open(self) -> bool:
        """Check if circuit is open (trading should stop)."""
        # Check if we should transition from OPEN to HALF_OPEN
        if self.metrics.state == CircuitState.OPEN:
            if self.metrics.opened_at:
                elapsed = (datetime.now() - self.metrics.opened_at).total_seconds()
                if elapsed >= self.config.timeout_seconds:
                    logger.info("Circuit timeout expired, transitioning to half-open")
                    self.metrics.state = CircuitState.HALF_OPEN
                    self.metrics.half_open_attempts = 0
                    return False  # Half-open allows testing
        
        return self.metrics.state == CircuitState.OPEN
    
    def is_closed(self) -> bool:
        """Check if circuit is closed (normal operation)."""
        return self.metrics.state == CircuitState.CLOSED
    
    def _open_circuit(self, reason: str, details: Any = None) -> None:
        """Open the circuit breaker."""
        if self.metrics.state == CircuitState.OPEN:
            return  # Already open
        
        logger.error(
            f"CIRCUIT BREAKER OPENED - Reason: {reason}, Details: {details}",
            extra={
                "circuit_state": "OPEN",
                "reason": reason,
                "details": str(details),
                "consecutive_failures": self.metrics.consecutive_failures
            }
        )
        
        self.metrics.state = CircuitState.OPEN
        self.metrics.opened_at = datetime.now()
    
    def _close_circuit(self) -> None:
        """Close the circuit breaker."""
        logger.info("Circuit breaker closed - resuming normal operation")
        self.metrics.state = CircuitState.CLOSED
        self.metrics.consecutive_failures = 0
        self.metrics.half_open_attempts = 0
        self.metrics.opened_at = None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current circuit breaker metrics."""
        return {
            "state": self.metrics.state.value,
            "consecutive_failures": self.metrics.consecutive_failures,
            "llm_failures": self.metrics.llm_failures,
            "data_quality_failures": self.metrics.data_quality_failures,
            "equity_drop_detected": self.metrics.equity_drop_detected,
            "last_failure_time": self.metrics.last_failure_time.isoformat() if self.metrics.last_failure_time else None,
            "opened_at": self.metrics.opened_at.isoformat() if self.metrics.opened_at else None,
            "initial_equity": self._initial_equity,
            "current_equity": self._last_equity,
            "equity_drop_pct": (
                (self._initial_equity - self._last_equity) / self._initial_equity * 100
                if self._initial_equity and self._initial_equity > 0 else None
            )
        }
    
    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        logger.info("Resetting circuit breaker")
        self.metrics = CircuitBreakerMetrics()
        self._initial_equity = None
        self._last_equity = None

