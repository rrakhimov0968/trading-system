"""Market hours utilities for US stock exchanges."""
import pytz
from datetime import datetime, time, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def is_market_open(exchange: str = "NYSE", now: Optional[datetime] = None) -> bool:
    """
    Check if US stock market is open (NYSE/NASDAQ hours).
    
    Hours: 9:30 AM - 4:00 PM ET, Monday-Friday.
    Excludes US holidays (handled via weekday check only - full holiday
    calendar would require additional logic).
    
    Args:
        exchange: Exchange name (currently only NYSE/NASDAQ supported)
        now: Current datetime (UTC). If None, uses datetime.now(pytz.UTC)
    
    Returns:
        True if market is open, False otherwise
    """
    if now is None:
        now = datetime.now(pytz.UTC)
    
    # Convert to Eastern Time
    et = pytz.timezone("US/Eastern")
    now_et = now.astimezone(et)
    
    # Check if weekday (Monday=0, Friday=4)
    if now_et.weekday() >= 5:  # Saturday (5) or Sunday (6)
        return False
    
    # Check time
    market_open = time(9, 30)   # 9:30 AM ET
    market_close = time(16, 0)  # 4:00 PM ET
    
    return market_open <= now_et.time() <= market_close


def get_next_market_open(now: Optional[datetime] = None) -> datetime:
    """
    Calculate the next market opening time (9:30 AM ET on next weekday).
    
    Args:
        now: Current datetime (UTC). If None, uses datetime.now(pytz.UTC)
    
    Returns:
        Next market open datetime (UTC)
    """
    if now is None:
        now = datetime.now(pytz.UTC)
    
    et = pytz.timezone("US/Eastern")
    now_et = now.astimezone(et)
    
    # Start with today at 9:30 AM ET
    next_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    
    # If today is a weekday and before 9:30 AM, market opens today
    if now_et.weekday() < 5 and now_et.time() < time(9, 30):
        return next_open.astimezone(pytz.UTC)
    
    # Otherwise, find next weekday
    days_to_add = 1
    while True:
        candidate = next_open + timedelta(days=days_to_add)
        if candidate.weekday() < 5:  # Monday-Friday
            return candidate.astimezone(pytz.UTC)
        days_to_add += 1


def get_next_market_close(now: Optional[datetime] = None) -> datetime:
    """
    Calculate the next market closing time (4:00 PM ET).
    
    Args:
        now: Current datetime (UTC). If None, uses datetime.now(pytz.UTC)
    
    Returns:
        Next market close datetime (UTC). If market is currently open,
        returns today's close. Otherwise, returns next weekday close.
    """
    if now is None:
        now = datetime.now(pytz.UTC)
    
    et = pytz.timezone("US/Eastern")
    now_et = now.astimezone(et)
    
    # Start with today at 4:00 PM ET
    next_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    
    # If market is open today (weekday and between 9:30-4:00), close is today
    if is_market_open(now=now):
        return next_close.astimezone(pytz.UTC)
    
    # Otherwise, find next weekday close
    # If today is a weekday but after 4 PM, move to next weekday
    if now_et.weekday() < 5 and now_et.time() >= time(16, 0):
        days_to_add = 1
    elif now_et.weekday() < 5 and now_et.time() < time(9, 30):
        # Before market opens today
        days_to_add = 0
    else:
        # Weekend
        days_to_add = 1
    
    while True:
        candidate = next_close + timedelta(days=days_to_add)
        if candidate.weekday() < 5:  # Monday-Friday
            return candidate.astimezone(pytz.UTC)
        days_to_add += 1


def get_market_status_message(now: Optional[datetime] = None) -> str:
    """
    Get a human-readable message about market status.
    
    Args:
        now: Current datetime (UTC). If None, uses datetime.now(pytz.UTC)
    
    Returns:
        Status message string
    """
    if now is None:
        now = datetime.now(pytz.UTC)
    
    et = pytz.timezone("US/Eastern")
    now_et = now.astimezone(et)
    
    if is_market_open(now=now):
        next_close = get_next_market_close(now=now)
        next_close_et = next_close.astimezone(et)
        return f"Market is OPEN. Closes at {next_close_et.strftime('%I:%M %p %Z')} on {next_close_et.strftime('%A, %B %d')}"
    else:
        next_open = get_next_market_open(now=now)
        next_open_et = next_open.astimezone(et)
        return f"Market is CLOSED. Opens at {next_open_et.strftime('%I:%M %p %Z')} on {next_open_et.strftime('%A, %B %d')}"

