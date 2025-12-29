"""Retry utilities with exponential backoff."""
import time
import logging
from functools import wraps
from typing import Callable, TypeVar, Tuple, Optional, List, Any
from utils.exceptions import APIError, AgentError

logger = logging.getLogger(__name__)

T = TypeVar('T')


class RetryConfig:
    """Configuration for retry logic."""
    
    def __init__(
        self,
        max_attempts: int = 3,
        initial_delay: float = 1.0,
        max_delay: float = 60.0,
        exponential_base: float = 2.0,
        retryable_exceptions: Optional[Tuple[type, ...]] = None,
        jitter: bool = True
    ):
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.retryable_exceptions = retryable_exceptions or (
            APIError,
            ConnectionError,
            TimeoutError,
            OSError
        )
        self.jitter = jitter


def retry_with_backoff(
    config: Optional[RetryConfig] = None,
    correlation_id: Optional[str] = None
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        config: Retry configuration
        correlation_id: Optional correlation ID for logging
        
    Example:
        @retry_with_backoff()
        def api_call():
            ...
    """
    if config is None:
        config = RetryConfig()
    
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[Exception] = None
            
            for attempt in range(1, config.max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except config.retryable_exceptions as e:
                    last_exception = e
                    
                    if attempt == config.max_attempts:
                        logger.error(
                            f"Max retry attempts ({config.max_attempts}) reached for {func.__name__}",
                            extra={"correlation_id": correlation_id, "attempt": attempt}
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        config.initial_delay * (config.exponential_base ** (attempt - 1)),
                        config.max_delay
                    )
                    
                    # Add jitter to prevent thundering herd
                    if config.jitter:
                        import random
                        delay = delay * (0.5 + random.random() * 0.5)
                    
                    logger.warning(
                        f"Retrying {func.__name__} (attempt {attempt}/{config.max_attempts}) "
                        f"after {delay:.2f}s due to {type(e).__name__}",
                        extra={"correlation_id": correlation_id, "attempt": attempt}
                    )
                    time.sleep(delay)
                except Exception as e:
                    # Non-retryable exception
                    logger.error(
                        f"Non-retryable exception in {func.__name__}: {type(e).__name__}",
                        extra={"correlation_id": correlation_id}
                    )
                    raise
            
            # Should never reach here, but for type checking
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected retry loop exit")
        
        return wrapper
    return decorator

