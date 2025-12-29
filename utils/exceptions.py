"""Custom exceptions for the trading system."""
from typing import Optional


class TradingSystemError(Exception):
    """Base exception for all trading system errors."""
    
    def __init__(
        self, 
        message: str, 
        correlation_id: Optional[str] = None,
        details: Optional[dict] = None
    ):
        super().__init__(message)
        self.message = message
        self.correlation_id = correlation_id
        self.details = details or {}


class AgentError(TradingSystemError):
    """Base exception for agent-related errors."""
    pass


class ConfigurationError(TradingSystemError):
    """Raised when configuration is invalid or missing."""
    pass


class ValidationError(TradingSystemError):
    """Raised when input validation fails."""
    pass


class APIError(TradingSystemError):
    """Raised when external API calls fail."""
    
    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        correlation_id: Optional[str] = None,
        details: Optional[dict] = None
    ):
        super().__init__(message, correlation_id, details)
        self.status_code = status_code


class ExecutionError(AgentError):
    """Raised when trade execution fails."""
    pass


class RiskCheckError(AgentError):
    """Raised when risk checks fail."""
    pass


class StrategyError(AgentError):
    """Raised when strategy evaluation fails."""
    pass


class QuantError(AgentError):
    """Raised when quantitative analysis fails."""
    pass

