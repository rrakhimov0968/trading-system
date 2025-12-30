"""Event bus for async pub/sub communication between agents."""
from typing import Callable, Dict, List, Any, Optional
import asyncio
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class EventBus:
    """
    Simple pub/sub event bus for async agent communication.
    
    Agents can subscribe to event types and publish events asynchronously.
    This enables decoupled, event-driven architecture.
    """
    
    def __init__(self):
        """Initialize the event bus."""
        self.subscribers: Dict[str, List[Callable]] = {}
        self._event_history: List[Dict[str, Any]] = []  # For debugging/testing
        self._max_history: int = 1000  # Limit history size
        self._lock = asyncio.Lock()
    
    def subscribe(self, event_type: str, callback: Callable) -> None:
        """
        Subscribe a callback to an event type.
        
        Args:
            event_type: The type of event to subscribe to (e.g., 'data_ready')
            callback: Async or sync callable to invoke when event is published
        """
        if event_type not in self.subscribers:
            self.subscribers[event_type] = []
        
        if callback not in self.subscribers[event_type]:
            self.subscribers[event_type].append(callback)
            logger.debug(
                f"Subscribed {callback.__name__ if hasattr(callback, '__name__') else str(callback)} "
                f"to event type '{event_type}'"
            )
    
    def unsubscribe(self, event_type: str, callback: Callable) -> None:
        """
        Unsubscribe a callback from an event type.
        
        Args:
            event_type: The type of event
            callback: The callback to remove
        """
        if event_type in self.subscribers:
            if callback in self.subscribers[event_type]:
                self.subscribers[event_type].remove(callback)
                logger.debug(
                    f"Unsubscribed {callback.__name__ if hasattr(callback, '__name__') else str(callback)} "
                    f"from event type '{event_type}'"
                )
    
    async def publish(self, event_type: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Publish an event to all subscribers.
        
        Args:
            event_type: The type of event (e.g., 'data_ready')
            data: The event data to pass to subscribers
            metadata: Optional metadata (timestamp, correlation_id, etc.)
        """
        async with self._lock:
            # Add to history
            event_record = {
                "event_type": event_type,
                "timestamp": datetime.now(),
                "data_type": type(data).__name__,
                "subscriber_count": len(self.subscribers.get(event_type, [])),
                "metadata": metadata or {}
            }
            self._event_history.append(event_record)
            
            # Limit history size
            if len(self._event_history) > self._max_history:
                self._event_history = self._event_history[-self._max_history:]
        
        if event_type not in self.subscribers:
            logger.debug(f"No subscribers for event type '{event_type}'")
            return
        
        logger.info(
            f"Publishing event '{event_type}' to {len(self.subscribers[event_type])} subscribers",
            extra={"event_type": event_type, "subscriber_count": len(self.subscribers[event_type])}
        )
        
        # Invoke all subscribers concurrently
        tasks = []
        for callback in self.subscribers[event_type]:
            try:
                # Check if callback is async or sync
                if asyncio.iscoroutinefunction(callback):
                    task = asyncio.create_task(callback(data))
                else:
                    # Wrap sync callback in coroutine
                    task = asyncio.create_task(self._call_sync_callback(callback, data))
                tasks.append(task)
            except Exception as e:
                logger.error(
                    f"Error creating task for callback {callback}: {e}",
                    exc_info=True
                )
        
        # Wait for all subscribers to complete
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log any errors
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    callback = self.subscribers[event_type][i]
                    logger.error(
                        f"Subscriber {callback.__name__ if hasattr(callback, '__name__') else str(callback)} "
                        f"raised exception: {result}",
                        exc_info=result
                    )
    
    async def _call_sync_callback(self, callback: Callable, data: Any) -> Any:
        """Call a synchronous callback in an executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, callback, data)
    
    def get_subscribers(self, event_type: str) -> List[Callable]:
        """Get list of subscribers for an event type."""
        return self.subscribers.get(event_type, []).copy()
    
    def get_event_history(self, event_type: Optional[str] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get event history for debugging/testing.
        
        Args:
            event_type: Optional filter by event type
            limit: Optional limit on number of events to return
        
        Returns:
            List of event records
        """
        history = self._event_history
        if event_type:
            history = [e for e in history if e["event_type"] == event_type]
        
        if limit:
            history = history[-limit:]
        
        return history
    
    def clear_history(self) -> None:
        """Clear event history."""
        self._event_history = []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics."""
        return {
            "total_event_types": len(self.subscribers),
            "total_subscribers": sum(len(callbacks) for callbacks in self.subscribers.values()),
            "event_types": {
                event_type: len(callbacks)
                for event_type, callbacks in self.subscribers.items()
            },
            "total_events_published": len(self._event_history)
        }

