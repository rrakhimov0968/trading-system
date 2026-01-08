"""Order tracking to prevent duplicate orders."""
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OrderTracker:
    """
    Track recent orders to prevent duplicates.
    
    Prevents placing multiple orders for the same symbol within a cooldown period.
    """
    
    def __init__(self, cooldown_minutes: int = 1440, max_orders_per_day: int = 10):
        """
        Initialize order tracker.
        
        Args:
            cooldown_minutes: Minimum minutes between orders for same symbol (default: 1440 = 24 hours)
            max_orders_per_day: Maximum total orders per day across all symbols (default: 10)
        """
        self.cooldown_minutes = cooldown_minutes
        self.max_orders_per_day = max_orders_per_day
        self.last_order_time: Dict[str, datetime] = {}
        self.daily_order_count: Dict[str, int] = {}
        self.last_reset_date = datetime.now().date()
        logger.info(
            f"OrderTracker initialized: {cooldown_minutes} min cooldown, "
            f"max {max_orders_per_day} orders/day"
        )
    
    def _reset_daily_count_if_needed(self) -> None:
        """Reset daily counter at midnight."""
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.daily_order_count.clear()
            self.last_reset_date = today
            logger.debug(f"Reset daily order count (new date: {today})")
    
    def can_place_order(self, symbol: str) -> Tuple[bool, Optional[str]]:
        """
        Check if order can be placed (cooldown + daily limit).
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Tuple of (can_place: bool, reason: Optional[str])
        """
        self._reset_daily_count_if_needed()
        
        # Check daily limit (across all symbols)
        total_today = sum(self.daily_order_count.values())
        if total_today >= self.max_orders_per_day:
            reason = (
                f"Daily order limit reached: {total_today}/{self.max_orders_per_day} orders today"
            )
            logger.warning(f"🚫 {reason}")
            return False, reason
        
        symbol_upper = symbol.upper()
        
        # Check cooldown for this specific symbol
        if symbol_upper not in self.last_order_time:
            return True, None
        
        last_time = self.last_order_time[symbol_upper]
        time_since = (datetime.now() - last_time).total_seconds() / 60
        
        if time_since < self.cooldown_minutes:
            reason = (
                f"Cooldown active for {symbol}: "
                f"{time_since:.0f} minutes since last order "
                f"(need {self.cooldown_minutes} minutes)"
            )
            return False, reason
        
        return True, None
    
    def record_order(self, symbol: str) -> None:
        """
        Record that an order was placed for this symbol.
        
        Args:
            symbol: Stock symbol
        """
        self._reset_daily_count_if_needed()
        
        symbol_upper = symbol.upper()
        self.last_order_time[symbol_upper] = datetime.now()
        
        # Increment daily count
        self.daily_order_count[symbol_upper] = self.daily_order_count.get(symbol_upper, 0) + 1
        total_today = sum(self.daily_order_count.values())
        
        logger.info(
            f"Recorded order for {symbol_upper} at {self.last_order_time[symbol_upper]}. "
            f"Daily count: {total_today}/{self.max_orders_per_day}"
        )
    
    def clear_tracking(self, symbol: Optional[str] = None) -> None:
        """
        Clear tracking for a symbol or all symbols.
        
        Args:
            symbol: Symbol to clear, or None to clear all
        """
        if symbol:
            symbol_upper = symbol.upper()
            if symbol_upper in self.last_order_time:
                del self.last_order_time[symbol_upper]
                logger.debug(f"Cleared tracking for {symbol_upper}")
        else:
            self.last_order_time.clear()
            logger.debug("Cleared all order tracking")

