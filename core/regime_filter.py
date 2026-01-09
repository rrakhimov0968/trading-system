"""
Market Regime Filter - System-level protection against unfavorable market conditions.

Prevents trading in bear markets and scales position sizes based on regime strength.
This is a system-level check that applies to ALL strategies.
"""
import logging
from typing import Tuple, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class RegimeFilter:
    """
    Filters trading activity based on market regime.
    
    Regime Check:
    - Bull Market: SPY > SMA200 → Favorable (100% sizing)
    - Bear Market: SPY < SMA200 → Unfavorable (0% or scaled sizing)
    
    Modes:
    - Strict: Blocks all trading in bear markets
    - Adaptive: Scales position sizes based on regime strength
    """
    
    def __init__(
        self,
        strict_mode: bool = True,
        benchmark_symbol: str = "SPY",
        sma_period: int = 200
    ):
        """
        Initialize regime filter.
        
        Args:
            strict_mode: If True, block all trades in bear markets.
                        If False, scale positions by regime strength (0-1).
            benchmark_symbol: Symbol to use for regime check (default: SPY)
            sma_period: SMA period for trend check (default: 200)
        """
        self.strict_mode = strict_mode
        self.benchmark_symbol = benchmark_symbol
        self.sma_period = sma_period
        self.last_regime_check = None
        self.last_regime_strength = None
    
    def check_regime(
        self,
        benchmark_data,
        current_price: Optional[float] = None
    ) -> Tuple[bool, float, str]:
        """
        Check current market regime.
        
        Args:
            benchmark_data: MarketData object for benchmark (SPY)
            current_price: Optional current price (if not provided, uses latest bar)
        
        Returns:
            Tuple of (can_trade: bool, regime_strength: float, reason: str)
            - can_trade: True if trading allowed in current regime
            - regime_strength: 0.0-1.0 scaling factor (1.0 = full bull, 0.0 = full bear)
            - reason: Explanation of regime status
        """
        try:
            if not benchmark_data or not benchmark_data.bars or len(benchmark_data.bars) < self.sma_period:
                logger.warning(
                    f"⚠️ Insufficient data for regime check ({len(benchmark_data.bars) if benchmark_data and benchmark_data.bars else 0} bars, "
                    f"need {self.sma_period}). Allowing trading."
                )
                return True, 1.0, "Insufficient data - allowing trading"
            
            # Convert to DataFrame
            df = benchmark_data.to_dataframe()
            if df.empty or len(df) < self.sma_period:
                logger.warning(f"⚠️ Insufficient DataFrame data for regime check. Allowing trading.")
                return True, 1.0, "Insufficient data - allowing trading"
            
            # Get current price
            if current_price is None:
                current_price = df['close'].iloc[-1]
            
            # Calculate SMA
            sma = df['close'].rolling(window=self.sma_period).mean()
            current_sma = sma.iloc[-1]
            
            if pd.isna(current_sma):
                logger.warning(f"⚠️ SMA{self.sma_period} is NaN. Allowing trading.")
                return True, 1.0, "SMA calculation failed - allowing trading"
            
            # Calculate regime strength (0.0 = bear, 1.0 = bull)
            regime_strength = current_price / current_sma
            # Clamp to reasonable range (0.5 to 1.5, then normalize to 0-1)
            regime_strength = max(0.5, min(1.5, regime_strength))
            regime_strength = (regime_strength - 0.5) / 1.0  # Normalize 0.5-1.5 → 0.0-1.0
            
            # Determine if we can trade
            is_bull_market = current_price > current_sma
            
            # Store for logging
            self.last_regime_check = is_bull_market
            self.last_regime_strength = regime_strength
            
            if self.strict_mode:
                # Strict mode: Block all trading in bear markets
                if is_bull_market:
                    reason = f"Bull market: {self.benchmark_symbol} ${current_price:.2f} > SMA{self.sma_period} ${current_sma:.2f}"
                    logger.info(f"✅ {reason}")
                    return True, 1.0, reason
                else:
                    reason = f"Bear market: {self.benchmark_symbol} ${current_price:.2f} < SMA{self.sma_period} ${current_sma:.2f} - BLOCKING ALL TRADES"
                    logger.warning(f"🚫 {reason}")
                    return False, 0.0, reason
            else:
                # Adaptive mode: Scale positions by regime strength
                if is_bull_market:
                    reason = f"Bull market: {self.benchmark_symbol} ${current_price:.2f} > SMA{self.sma_period} ${current_sma:.2f} (strength: {regime_strength:.1%})"
                    logger.info(f"✅ {reason}")
                    return True, 1.0, reason
                else:
                    reason = (
                        f"Bear market: {self.benchmark_symbol} ${current_price:.2f} < SMA{self.sma_period} ${current_sma:.2f} "
                        f"- SCALING POSITIONS TO {regime_strength:.1%}"
                    )
                    logger.warning(f"⚠️ {reason}")
                    return True, regime_strength, reason
                    
        except Exception as e:
            logger.error(f"Error checking regime: {e}", exc_info=True)
            # On error, allow trading (fail-safe)
            return True, 1.0, f"Regime check error - allowing trading: {e}"
    
    def apply_regime_scaling(
        self,
        position_value: float,
        regime_strength: float
    ) -> float:
        """
        Apply regime scaling to position value.
        
        Args:
            position_value: Original position value
            regime_strength: Regime strength factor (0.0-1.0)
        
        Returns:
            Scaled position value
        """
        if self.strict_mode:
            # In strict mode, regime_strength is either 0.0 (blocked) or 1.0 (allowed)
            # This method shouldn't be called if regime_strength is 0.0
            return position_value * regime_strength
        else:
            # Adaptive mode: Scale by regime strength
            return position_value * regime_strength
