"""
Market Regime - Immutable dataclass for regime state.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class MarketRegime:
    """
    Immutable market regime state.
    
    Attributes:
        allowed: Whether trading is allowed in current regime
        risk_scalar: Position sizing scalar (0.0-1.0, can be >1.0 for strong bull)
        reason: Explanation of regime status
    """
    allowed: bool
    risk_scalar: float
    reason: str
    
    def __post_init__(self):
        """Validate regime values."""
        if not 0.0 <= self.risk_scalar <= 1.25:
            raise ValueError(f"risk_scalar must be between 0.0 and 1.25, got {self.risk_scalar}")
