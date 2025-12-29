"""Enums for trading system models."""
from enum import Enum


class OrderSide(str, Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"
    
    @classmethod
    def from_string(cls, value: str) -> "OrderSide":
        """Convert string to OrderSide."""
        try:
            return cls(value.lower())
        except ValueError:
            raise ValueError(f"Invalid order side: {value}. Must be 'buy' or 'sell'")


class OrderType(str, Enum):
    """Order type."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    """Time in force for orders."""
    DAY = "day"
    GTC = "gtc"  # Good till cancelled
    OPG = "opg"  # Opening
    IOC = "ioc"  # Immediate or cancel
    FOK = "fok"  # Fill or kill


class OrderStatus(str, Enum):
    """Order status."""
    NEW = "new"
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class RiskCheckResult(str, Enum):
    """Risk check result."""
    APPROVED = "approved"
    REJECTED = "rejected"
    PENDING_REVIEW = "pending_review"

