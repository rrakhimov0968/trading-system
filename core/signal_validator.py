"""
Signal Validator - Validates trading signals are still valid at execution time.

Protects against:
- Overnight gaps
- Pre-market shocks
- Macro events
- Volatility regime flips
"""
from typing import Optional, Dict
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of signal validation"""
    valid: bool
    reason: str
    scan_price: Optional[float] = None
    current_price: Optional[float] = None
    gap_pct: Optional[float] = None
    age_hours: Optional[float] = None


class SignalValidator:
    """
    Validates trading signals are still fresh and actionable.
    
    Critical for scanner-driven strategies where signals are generated
    at EOD but executed the next day.
    """
    
    def __init__(
        self,
        max_gap_pct: float = 0.02,        # 2% max price gap
        max_signal_age_hours: float = 24,  # 24 hours max age
        enable_regime_check: bool = True
    ):
        """
        Initialize validator.
        
        Args:
            max_gap_pct: Maximum allowed price gap (default 2%)
            max_signal_age_hours: Maximum signal age in hours
            enable_regime_check: Check for volatility regime changes
        """
        self.max_gap_pct = max_gap_pct
        self.max_signal_age_hours = max_signal_age_hours
        self.enable_regime_check = enable_regime_check
    
    def validate_price_freshness(
        self,
        symbol: str,
        scan_price: float,
        current_price: float,
        scan_timestamp: Optional[datetime] = None
    ) -> ValidationResult:
        """
        Validate that price hasn't gapped significantly since scan.
        
        Args:
            symbol: Stock symbol
            scan_price: Price when signal was generated
            current_price: Current market price
            scan_timestamp: When signal was generated (optional)
        
        Returns:
            ValidationResult with decision and reasoning
        """
        # Calculate gap
        gap = abs(current_price - scan_price) / scan_price
        gap_direction = "DOWN" if current_price < scan_price else "UP"
        
        # Check age if provided
        age_hours = None
        if scan_timestamp:
            age = datetime.now() - scan_timestamp
            age_hours = age.total_seconds() / 3600
            
            if age_hours > self.max_signal_age_hours:
                return ValidationResult(
                    valid=False,
                    reason=f"Signal too old: {age_hours:.1f} hours (max {self.max_signal_age_hours})",
                    scan_price=scan_price,
                    current_price=current_price,
                    gap_pct=gap,
                    age_hours=age_hours
                )
        
        # Check gap
        if gap > self.max_gap_pct:
            reason = (
                f"Price gap too large: {gap:.1%} {gap_direction} "
                f"(${scan_price:.2f} → ${current_price:.2f}, "
                f"max allowed: {self.max_gap_pct:.1%})"
            )
            
            logger.warning(f"🚫 {symbol}: {reason}")
            
            return ValidationResult(
                valid=False,
                reason=reason,
                scan_price=scan_price,
                current_price=current_price,
                gap_pct=gap,
                age_hours=age_hours
            )
        
        # Signal is fresh
        age_str = f"{age_hours:.1f}h" if age_hours else "N/A"
        logger.info(
            f"✅ {symbol}: Price freshness OK "
            f"(gap: {gap:.2%}, age: {age_str})"
        )
        
        return ValidationResult(
            valid=True,
            reason="Signal fresh",
            scan_price=scan_price,
            current_price=current_price,
            gap_pct=gap,
            age_hours=age_hours
        )
    
    def validate_from_scanner_data(
        self,
        symbol: str,
        current_price: float,
        scanner_data: Dict
    ) -> ValidationResult:
        """
        Validate signal using scanner metadata.
        
        Args:
            symbol: Stock symbol
            current_price: Current market price
            scanner_data: Scanner output with scan_prices and timestamp
        
        Returns:
            ValidationResult
        """
        scan_price = scanner_data.get('scan_prices', {}).get(symbol)
        
        if not scan_price:
            logger.warning(f"⚠️  {symbol}: No scan price in scanner data")
            return ValidationResult(
                valid=False,
                reason="No scan price available",
                current_price=current_price
            )
        
        # Parse timestamp
        scan_timestamp = None
        if 'scan_timestamp' in scanner_data:
            try:
                scan_timestamp = datetime.fromisoformat(
                    scanner_data['scan_timestamp']
                )
            except Exception as e:
                logger.warning(f"⚠️  Failed to parse timestamp: {e}")
        
        return self.validate_price_freshness(
            symbol,
            scan_price,
            current_price,
            scan_timestamp
        )
    
    def check_volatility_regime_flip(
        self,
        symbol: str,
        historical_data
    ) -> ValidationResult:
        """
        Check if volatility regime has flipped since signal generation.
        
        This catches cases where:
        - Market was calm when scanned, now turbulent
        - VIX spiked overnight
        - Sector-specific volatility event
        
        Args:
            symbol: Stock symbol
            historical_data: Recent price bars
        
        Returns:
            ValidationResult
        """
        if not self.enable_regime_check:
            return ValidationResult(valid=True, reason="Regime check disabled")
        
        try:
            import numpy as np
            
            if not historical_data or not historical_data.bars:
                return ValidationResult(valid=True, reason="No data for regime check")
            
            # Calculate recent volatility (last 5 days)
            if len(historical_data.bars) < 5:
                return ValidationResult(valid=True, reason="Insufficient data")
            
            closes = [bar.close for bar in historical_data.bars[-5:]]
            if len(closes) < 2:
                return ValidationResult(valid=True, reason="Insufficient closes")
            
            returns = np.diff(closes) / closes[:-1]
            recent_vol = np.std(returns) * np.sqrt(252)  # Annualized
            
            # Calculate longer-term volatility (last 20 days)
            if len(historical_data.bars) >= 20:
                closes_long = [bar.close for bar in historical_data.bars[-20:]]
                returns_long = np.diff(closes_long) / closes_long[:-1]
                normal_vol = np.std(returns_long) * np.sqrt(252)
                
                # Check if volatility spiked
                vol_ratio = recent_vol / normal_vol if normal_vol > 0 else 1.0
                
                if vol_ratio > 2.0:  # Volatility doubled
                    reason = (
                        f"Volatility regime flip detected: "
                        f"Recent vol {recent_vol:.1%} vs "
                        f"Normal vol {normal_vol:.1%} "
                        f"(ratio: {vol_ratio:.1f}x)"
                    )
                    logger.warning(f"⚠️  {symbol}: {reason}")
                    
                    return ValidationResult(
                        valid=False,
                        reason=reason
                    )
            
            return ValidationResult(valid=True, reason="Volatility regime stable")
            
        except Exception as e:
            logger.warning(f"⚠️  Regime check failed for {symbol}: {e}")
            # Don't block on check failure
            return ValidationResult(valid=True, reason=f"Check failed: {e}")
