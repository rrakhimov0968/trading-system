"""Base agent class with common functionality."""
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from utils.exceptions import AgentError
from config.settings import AppConfig, get_config


class BaseAgent(ABC):
    """Base class for all agents in the trading system."""
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize the base agent.
        
        Args:
            config: Application configuration. If None, loads from environment.
        """
        self.config = config or get_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._correlation_id: Optional[str] = None
    
    @property
    def correlation_id(self) -> Optional[str]:
        """Get the current correlation ID."""
        return self._correlation_id
    
    @correlation_id.setter
    def correlation_id(self, value: Optional[str]) -> None:
        """Set the correlation ID for request tracing."""
        self._correlation_id = value
    
    def generate_correlation_id(self) -> str:
        """Generate a new correlation ID."""
        self._correlation_id = str(uuid.uuid4())
        return self._correlation_id
    
    def _log(self, level: int, message: str, **kwargs: Any) -> None:
        """Internal logging method with correlation ID."""
        extra = {"correlation_id": self._correlation_id}
        extra.update(kwargs)
        self.logger.log(level, message, extra=extra)
    
    def log_debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message."""
        self._log(logging.DEBUG, message, **kwargs)
    
    def log_info(self, message: str, **kwargs: Any) -> None:
        """Log an info message."""
        self._log(logging.INFO, message, **kwargs)
    
    def log_warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message."""
        self._log(logging.WARNING, message, **kwargs)
    
    def log_error(self, message: str, **kwargs: Any) -> None:
        """Log an error message."""
        self._log(logging.ERROR, message, **kwargs)
    
    def log_exception(
        self, 
        message: str, 
        exc: Exception, 
        **kwargs: Any
    ) -> None:
        """Log an exception with traceback."""
        extra = {"correlation_id": self._correlation_id}
        extra.update(kwargs)
        self.logger.exception(message, extra=extra, exc_info=exc)
    
    def handle_error(
        self, 
        error: Exception, 
        context: Optional[Dict[str, Any]] = None
    ) -> AgentError:
        """
        Convert an exception to an AgentError with context.
        
        Args:
            error: The original exception
            context: Additional context information
            
        Returns:
            AgentError with correlation ID and details
        """
        context = context or {}
        agent_error = AgentError(
            message=str(error),
            correlation_id=self._correlation_id,
            details=context
        )
        self.log_exception(f"Error in {self.__class__.__name__}", error, **context)
        return agent_error
    
    @abstractmethod
    def process(self, *args: Any, **kwargs: Any) -> Any:
        """
        Main processing method - must be implemented by subclasses.
        
        This is the primary entry point for agent operations.
        """
        pass
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the agent.
        
        Returns:
            Dictionary with health status and details
        """
        return {
            "agent": self.__class__.__name__,
            "status": "healthy",
            "correlation_id": self._correlation_id
        }

