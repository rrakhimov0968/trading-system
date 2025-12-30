"""Rate limiting utilities to prevent API rate limit violations."""
import time
import asyncio
from collections import deque
from typing import Optional
import logging
from threading import Lock

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Rate limiter using sliding window algorithm.
    
    Ensures we don't exceed a certain number of requests within a time window.
    This is thread-safe and async-safe.
    """
    
    def __init__(
        self, 
        max_requests: int = 200, 
        window_seconds: int = 60,
        name: str = "default"
    ):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in the window
            window_seconds: Time window in seconds (default: 60 = 1 minute)
            name: Name for logging purposes
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.name = name
        
        # Use deque to store request timestamps (sliding window)
        self._requests = deque()
        self._lock = Lock()
        
        # For async operations
        self._async_lock = asyncio.Lock()
        
        logger.info(
            f"RateLimiter '{name}' initialized: "
            f"{max_requests} requests per {window_seconds} seconds"
        )
    
    def _clean_old_requests(self) -> None:
        """Remove requests outside the time window."""
        current_time = time.time()
        cutoff_time = current_time - self.window_seconds
        
        # Remove requests older than the window
        while self._requests and self._requests[0] < cutoff_time:
            self._requests.popleft()
    
    def _can_make_request(self) -> bool:
        """Check if a request can be made now."""
        with self._lock:
            self._clean_old_requests()
            return len(self._requests) < self.max_requests
    
    def _wait_time(self) -> float:
        """Calculate how long to wait before next request can be made."""
        with self._lock:
            self._clean_old_requests()
            
            if len(self._requests) < self.max_requests:
                return 0.0
            
            # Oldest request time
            oldest_time = self._requests[0]
            wait_until = oldest_time + self.window_seconds
            wait_time = wait_until - time.time()
            
            return max(0.0, wait_time)
    
    def _record_request(self) -> None:
        """Record that a request was made."""
        with self._lock:
            current_time = time.time()
            self._requests.append(current_time)
            
            # Log if approaching limit
            count = len(self._requests)
            if count >= self.max_requests * 0.9:  # 90% threshold
                logger.warning(
                    f"RateLimiter '{self.name}': "
                    f"{count}/{self.max_requests} requests used in last {self.window_seconds}s"
                )
    
    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to make a request (synchronous).
        
        Args:
            blocking: If True, wait until request can be made
            timeout: Maximum time to wait (None = wait forever)
        
        Returns:
            True if request can be made, False if timeout expired
        """
        start_time = time.time()
        
        while True:
            if self._can_make_request():
                self._record_request()
                return True
            
            if not blocking:
                return False
            
            wait_time = self._wait_time()
            
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed + wait_time > timeout:
                    return False
                wait_time = min(wait_time, timeout - elapsed)
            
            if wait_time > 0:
                time.sleep(wait_time)
        
        return False
    
    async def acquire_async(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to make a request (asynchronous).
        
        Args:
            blocking: If True, wait until request can be made
            timeout: Maximum time to wait (None = wait forever)
        
        Returns:
            True if request can be made, False if timeout expired
        """
        start_time = time.time()
        
        async with self._async_lock:
            while True:
                # Check synchronously with lock
                if self._can_make_request():
                    self._record_request()
                    return True
                
                if not blocking:
                    return False
                
                wait_time = self._wait_time()
                
                if timeout is not None:
                    elapsed = time.time() - start_time
                    if elapsed + wait_time > timeout:
                        return False
                    wait_time = min(wait_time, timeout - elapsed)
                
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
        
        return False
    
    def get_stats(self) -> dict:
        """Get current rate limiter statistics."""
        with self._lock:
            self._clean_old_requests()
            count = len(self._requests)
            utilization = count / self.max_requests if self.max_requests > 0 else 0
            
            return {
                "name": self.name,
                "requests_in_window": count,
                "max_requests": self.max_requests,
                "window_seconds": self.window_seconds,
                "utilization_percent": utilization * 100,
                "remaining_requests": max(0, self.max_requests - count),
                "wait_time_seconds": self._wait_time()
            }


# Global rate limiters (singleton pattern)
_global_limiters = {}
_global_lock = Lock()


def get_rate_limiter(
    name: str = "alpaca",
    max_requests: int = 200,
    window_seconds: int = 60
) -> RateLimiter:
    """
    Get or create a global rate limiter instance.
    
    Args:
        name: Name of the rate limiter (e.g., 'alpaca', 'alpaca_data')
        max_requests: Maximum requests per window (default: 200 for Alpaca)
        window_seconds: Time window in seconds (default: 60 = 1 minute)
    
    Returns:
        RateLimiter instance
    """
    with _global_lock:
        if name not in _global_limiters:
            _global_limiters[name] = RateLimiter(
                max_requests=max_requests,
                window_seconds=window_seconds,
                name=name
            )
        return _global_limiters[name]

