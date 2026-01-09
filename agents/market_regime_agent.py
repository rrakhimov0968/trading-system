"""
Market Regime Agent - Determines market regime and risk scaling.

Pure code, no LLM. Evaluates SPY vs SMA200 to determine if we're in
a bull market (favorable) or bear market (unfavorable).
"""
import logging
from typing import Dict, Optional
import pandas as pd

from agents.base import BaseAgent
from agents.data_agent import DataAgent
from core.market_regime import MarketRegime
from models.market_data import MarketData

logger = logging.getLogger(__name__)


class MarketRegimeAgent(BaseAgent):
    """
    Evaluates market regime and returns risk scaling factor.
    
    Regime Logic:
    - Bull Market (SPY > SMA200): risk_scalar = 1.0 (full sizing)
    - Bear Market (SPY < SMA200): risk_scalar = 0.0 (soft gate) or scaled
    - Strong Bull (SPY >> SMA200): risk_scalar can be >1.0 (up to 1.25)
    
    This is a SOFT gate - never completely blocks trading, just scales position sizes.
    This allows strategies to work but reduces risk in unfavorable conditions.
    """
    
    def __init__(self, config=None, data_agent: Optional[DataAgent] = None):
        """
        Initialize market regime agent.
        
        Args:
            config: AppConfig (optional, loads from env if not provided)
            data_agent: DataAgent instance (required for fetching SPY data)
        """
        super().__init__(config)
        self.data_agent = data_agent
        self.benchmark_symbol = getattr(self.config, 'regime_benchmark', 'SPY')
        self.sma_period = getattr(self.config, 'regime_sma_period', 200)
        self.strict_mode = getattr(self.config, 'strict_regime', False)  # Default: soft scalar
    
    def process(self, market_data: Optional[Dict[str, MarketData]] = None) -> MarketRegime:
        """
        Evaluate current market regime.
        
        Args:
            market_data: Optional dict of MarketData (if None, fetches SPY)
        
        Returns:
            MarketRegime with allowed flag, risk_scalar, and reason
        """
        try:
            # Get SPY data
            if market_data and self.benchmark_symbol in market_data:
                spy_data = market_data[self.benchmark_symbol]
            elif self.data_agent:
                # Fetch SPY data if not provided
                spy_data = self.data_agent.fetch(
                    symbol=self.benchmark_symbol,
                    timeframe="1Day",
                    limit=252  # 1 year of data for SMA200
                )
            else:
                # No data available - allow trading with full sizing
                logger.warning(f"Cannot evaluate regime: no SPY data available")
                return MarketRegime(
                    allowed=True,
                    risk_scalar=1.0,
                    reason="No_SPY_data_available"
                )
            
            # Validate data
            if not spy_data or not spy_data.bars or len(spy_data.bars) < self.sma_period:
                logger.warning(
                    f"Insufficient SPY data for regime check: "
                    f"{len(spy_data.bars) if spy_data and spy_data.bars else 0} bars, "
                    f"need {self.sma_period}"
                )
                return MarketRegime(
                    allowed=True,
                    risk_scalar=1.0,
                    reason="Insufficient_SPY_data"
                )
            
            # Convert to DataFrame
            df = spy_data.to_dataframe()
            if df.empty or len(df) < self.sma_period:
                logger.warning("Insufficient DataFrame data for regime check")
                return MarketRegime(
                    allowed=True,
                    risk_scalar=1.0,
                    reason="Insufficient_dataframe_data"
                )
            
            # Calculate current price and SMA200
            current_price = df['close'].iloc[-1]
            sma = df['close'].rolling(window=self.sma_period).mean()
            current_sma = sma.iloc[-1]
            
            if pd.isna(current_sma):
                logger.warning("SMA calculation returned NaN")
                return MarketRegime(
                    allowed=True,
                    risk_scalar=1.0,
                    reason="SMA_calculation_failed"
                )
            
            # Determine regime
            is_bull_market = current_price > current_sma
            strength_ratio = current_price / current_sma
            
            if self.strict_mode:
                # Strict mode: Hard gate (block trading in bear markets)
                if is_bull_market:
                    return MarketRegime(
                        allowed=True,
                        risk_scalar=1.0,
                        reason=f"Bull_market_{self.benchmark_symbol}_{current_price:.2f}_above_SMA{self.sma_period}_{current_sma:.2f}"
                    )
                else:
                    return MarketRegime(
                        allowed=False,
                        risk_scalar=0.0,
                        reason=f"Bear_market_{self.benchmark_symbol}_{current_price:.2f}_below_SMA{self.sma_period}_{current_sma:.2f}"
                    )
            else:
                # Soft mode (default): Scale positions, never fully block
                if is_bull_market:
                    # Bull market: Full sizing or slightly more for strong bull
                    # Clamp strength_ratio to reasonable range (0.8-1.25) for scalar
                    risk_scalar = min(max(strength_ratio, 0.8), 1.25)
                    return MarketRegime(
                        allowed=True,
                        risk_scalar=risk_scalar,
                        reason=f"Bull_market_{self.benchmark_symbol}_{current_price:.2f}_above_SMA{self.sma_period}_{current_sma:.2f}_scalar_{risk_scalar:.2f}"
                    )
                else:
                    # Bear market: Scale down positions (but don't block)
                    # Strength ratio will be < 1.0, scale it to 0.0-0.5 range
                    # Example: SPY at 0.95 * SMA200 → scalar = 0.25
                    #          SPY at 0.90 * SMA200 → scalar = 0.10
                    #          SPY at 0.85 * SMA200 → scalar = 0.00
                    risk_scalar = max(0.0, (strength_ratio - 0.85) / 0.15)  # Maps 0.85-1.0 → 0.0-1.0
                    risk_scalar = min(risk_scalar, 0.5)  # Cap at 0.5 in bear markets
                    
                    return MarketRegime(
                        allowed=True,
                        risk_scalar=risk_scalar,
                        reason=f"Bear_market_{self.benchmark_symbol}_{current_price:.2f}_below_SMA{self.sma_period}_{current_sma:.2f}_scaled_to_{risk_scalar:.2f}"
                    )
                    
        except Exception as e:
            logger.error(f"Error evaluating market regime: {e}", exc_info=True)
            # Fail-safe: Allow trading with full sizing
            return MarketRegime(
                allowed=True,
                risk_scalar=1.0,
                reason=f"Regime_evaluation_error_{str(e)[:50]}"
            )
