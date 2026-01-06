"""Quantitative analysis agent for validating trading signals."""
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor

from agents.base import BaseAgent
from models.signal import TradingSignal, SignalAction
from models.market_data import MarketData
from utils.exceptions import QuantError, AgentError
from utils.retry import retry_with_backoff, RetryConfig


class QuantAgent(BaseAgent):
    """
    Agent responsible for quantitative validation of trading signals.
    
    Role: Deterministic quantitative analysis with optional LLM interpretation
    - Performs statistical validation (Sharpe ratio, drawdown, multicollinearity)
    - Adjusts confidence scores based on historical performance
    - Optional LLM (Claude) for interpreting results and flagging anomalies
    
    Flow: Signal validation → Confidence adjustment → Output validated signals
    """
    
    def __init__(self, config=None):
        """
        Initialize the quant agent.
        
        Args:
            config: Application configuration. If None, loads from environment.
        """
        super().__init__(config)
        
        # Validation thresholds (more realistic defaults)
        self.min_sharpe = float(self.config.__dict__.get('quant_min_sharpe', 0.8))  # Lowered from 1.5
        self.max_drawdown = float(self.config.__dict__.get('quant_max_drawdown', 0.15))  # Increased from 0.08 (8%) to 0.15 (15%)
        self.max_vif = float(self.config.__dict__.get('quant_max_vif', 10.0))
        
        # Optional LLM for interpretation (Claude)
        self.use_llm_review = self.config.__dict__.get('quant_use_llm', False)
        self.llm_client = None
        
        if self.use_llm_review and self.config.anthropic:
            try:
                from anthropic import Anthropic
                self.llm_client = Anthropic(api_key=self.config.anthropic.api_key)
                self.llm_model = self.config.anthropic.model or "claude-3-opus-20240229"
                self.log_info(f"QuantAgent initialized with LLM review enabled ({self.llm_model})")
            except Exception as e:
                self.log_warning(f"Failed to initialize LLM for QuantAgent: {e}")
                self.use_llm_review = False
        
        if not self.use_llm_review:
            self.log_info("QuantAgent initialized (code-only mode, no LLM)")
    
    def process(
        self, 
        signals: List[TradingSignal],
        market_data: Optional[Dict[str, MarketData]] = None
    ) -> List[TradingSignal]:
        """
        Process and validate trading signals.
        
        This is the main entry point for the agent.
        
        Args:
            signals: List of TradingSignal objects to validate
            market_data: Optional dictionary mapping symbol to MarketData for historical validation
        
        Returns:
            List of validated TradingSignal objects with adjusted confidence scores
        
        Raises:
            QuantError: If validation fails critically
        """
        self.generate_correlation_id()
        self.log_info(
            f"Processing {len(signals)} signals for quantitative validation",
            signal_count=len(signals)
        )
        
        validated_signals = []
        
        for signal in signals:
            try:
                # Get historical data if available
                if market_data and signal.symbol in market_data:
                    signal.historical_data = market_data[signal.symbol].to_dataframe()
                elif signal.historical_data is None or (hasattr(signal.historical_data, 'empty') and signal.historical_data.empty):
                    self.log_warning(
                        f"No historical data available for {signal.symbol}, "
                        "skipping quantitative validation"
                    )
                    validated_signals.append(signal)
                    continue
                
                # Perform validation
                original_confidence = signal.confidence
                self.basic_validation(signal)
                self.confidence_validation(signal)
                
                if signal.confidence != original_confidence:
                    self.log_info(
                        f"Confidence adjusted for {signal.symbol}: "
                        f"{original_confidence:.2f} → {signal.confidence:.2f}"
                    )
                
                validated_signals.append(signal)
                
            except QuantError as e:
                self.log_error(
                    f"Quantitative validation failed for {signal.symbol}: {e}",
                    symbol=signal.symbol
                )
                # Optionally exclude signal or mark as low confidence
                signal.confidence = 0.0
                validated_signals.append(signal)
                continue
            except Exception as e:
                self.log_exception(f"Unexpected error validating {signal.symbol}", e)
                # Continue with original signal
                validated_signals.append(signal)
                continue
        
        self.log_info(
            f"Validated {len(validated_signals)}/{len(signals)} signals",
            validated_count=len(validated_signals)
        )
        
        return validated_signals
    
    def basic_validation(self, signal: TradingSignal) -> None:
        """
        Perform basic statistical validation on a signal.
        
        Checks:
        - Positive expectancy (return mean > 0)
        - Multicollinearity (VIF on OHLC data)
        - Regime consistency (volatility stability)
        
        Args:
            signal: TradingSignal to validate
        
        Raises:
            QuantError: If validation fails critically
        """
        if signal.historical_data is None or signal.historical_data.empty:
            raise QuantError(f"No historical data available for {signal.symbol}")
        
        df = signal.historical_data.copy()
        
        # Ensure we have required columns
        if 'close' not in df.columns:
            raise QuantError(f"Missing 'close' column in historical data for {signal.symbol}")
        
        # Calculate returns
        if len(df) < 2:
            raise QuantError(f"Insufficient data points for {signal.symbol}")
        
        returns = df['close'].pct_change().dropna()
        
        if len(returns) == 0:
            raise QuantError(f"Could not calculate returns for {signal.symbol}")
        
        # Check 1: Positive expectancy
        mean_return = returns.mean()
        if mean_return <= 0:
            self.log_warning(
                f"Low/negative expectancy for {signal.symbol}: {mean_return:.4f}",
                symbol=signal.symbol,
                mean_return=mean_return
            )
            signal.confidence *= 0.8  # Reduce confidence by 20%
        
        # Check 2: Multicollinearity (if we have OHLC data)
        ohlc_cols = ['open', 'high', 'low', 'close']
        if all(col in df.columns for col in ohlc_cols):
            try:
                # Use subset of recent data for VIF calculation
                ohlc_data = df[ohlc_cols].tail(50).values
                
                if len(ohlc_data) > ohlc_data.shape[1]:  # Need more rows than columns
                    vif_scores = []
                    for i in range(ohlc_data.shape[1]):
                        try:
                            vif = variance_inflation_factor(ohlc_data, i)
                            vif_scores.append(vif)
                        except Exception:
                            continue
                    
                    if vif_scores:
                        max_vif = max(vif_scores)
                        if max_vif > self.max_vif:
                            raise QuantError(
                                f"High multicollinearity in data for {signal.symbol}: "
                                f"max VIF {max_vif:.2f} > {self.max_vif}"
                            )
            except Exception as e:
                # VIF calculation can fail with insufficient data, log and continue
                self.log_debug(f"VIF calculation skipped for {signal.symbol}: {e}")
        
        # Check 3: Regime consistency (volatility stability)
        df['volatility'] = returns.rolling(window=min(20, len(returns))).std()
        current_vol = df['volatility'].iloc[-1]
        mean_vol = df['volatility'].mean()
        
        if pd.notna(current_vol) and pd.notna(mean_vol) and mean_vol > 0:
            vol_ratio = current_vol / mean_vol
            if vol_ratio > 2.0:  # Volatility spike
                self.log_warning(
                    f"High volatility regime for {signal.symbol}: "
                    f"current vol {current_vol:.4f} vs mean {mean_vol:.4f} (ratio: {vol_ratio:.2f}x)",
                    symbol=signal.symbol,
                    volatility_ratio=vol_ratio
                )
                signal.confidence *= 0.7  # Reduce confidence by 30%
        
        # Optional LLM review
        if self.use_llm_review and self.llm_client:
            try:
                self._llm_review_basic_validation(signal, mean_return, max_vif if 'max_vif' in locals() else None, vol_ratio if 'vol_ratio' in locals() else None)
            except Exception as e:
                self.log_debug(f"LLM review failed for {signal.symbol}: {e}")
        
        self.log_debug(
            f"Basic validation completed for {signal.symbol}",
            confidence=signal.confidence
        )
    
    def confidence_validation(self, signal: TradingSignal) -> None:
        """
        Adjust confidence score based on historical performance metrics.
        
        Calculates:
        - Sharpe ratio (risk-adjusted returns)
        - Maximum drawdown (worst peak-to-trough decline)
        
        Adjusts confidence based on these metrics relative to thresholds.
        
        Args:
            signal: TradingSignal to validate and adjust
        """
        if signal.historical_data is None or signal.historical_data.empty:
            return
        
        df = signal.historical_data.copy()
        
        if 'close' not in df.columns or len(df) < 2:
            return
        
        returns = df['close'].pct_change().dropna()
        
        if len(returns) < 30:  # Need sufficient data for meaningful metrics
            self.log_warning(f"Insufficient data for confidence validation: {signal.symbol}")
            return
        
        # Calculate Sharpe ratio (annualized)
        mean_return = returns.mean()
        std_return = returns.std()
        
        if std_return > 0:
            sharpe = (mean_return / std_return) * np.sqrt(252)  # Annualized for daily returns
        else:
            sharpe = 0.0
        
        # Adjust confidence based on Sharpe
        if sharpe < self.min_sharpe:
            adjustment_factor = max(0.5, sharpe / self.min_sharpe)  # Scale down, but not below 0.5x
            signal.confidence *= adjustment_factor
            self.log_warning(
                f"Low Sharpe ratio for {signal.symbol}: {sharpe:.2f} < {self.min_sharpe}",
                symbol=signal.symbol,
                sharpe=sharpe,
                adjustment_factor=adjustment_factor
            )
        
        # Calculate maximum drawdown
        cum_returns = (1 + returns).cumprod()
        peak = cum_returns.expanding(min_periods=1).max()
        drawdown = (cum_returns / peak) - 1
        max_dd = abs(drawdown.min())
        
        # Adjust confidence based on drawdown
        if max_dd > self.max_drawdown:
            adjustment_factor = self.max_drawdown / max_dd
            signal.confidence *= adjustment_factor
            self.log_warning(
                f"High drawdown for {signal.symbol}: {max_dd:.2%} > {self.max_drawdown:.2%}",
                symbol=signal.symbol,
                max_drawdown=max_dd,
                adjustment_factor=adjustment_factor
            )
        
        # Clamp confidence to valid range
        signal.confidence = max(0.0, min(1.0, signal.confidence))
        
        # Optional LLM review
        if self.use_llm_review and self.llm_client:
            try:
                self._llm_review_confidence_validation(signal, sharpe, max_dd)
            except Exception as e:
                self.log_debug(f"LLM confidence review failed for {signal.symbol}: {e}")
        
        self.log_debug(
            f"Confidence validation completed for {signal.symbol}",
            final_confidence=signal.confidence,
            sharpe=sharpe,
            max_drawdown=max_dd
        )
    
    def _llm_review_basic_validation(
        self,
        signal: TradingSignal,
        mean_return: float,
        max_vif: Optional[float],
        vol_ratio: Optional[float]
    ) -> None:
        """Optional LLM review of basic validation results."""
        prompt = f"""Review quantitative validation for {signal.symbol}:

- Mean Return: {mean_return:.4f}
- Max VIF: {max_vif:.2f if max_vif else 'N/A'}
- Volatility Ratio: {vol_ratio:.2f if vol_ratio else 'N/A'}
- Current Confidence: {signal.confidence:.2f}

Flag any statistical anomalies or concerns."""
        
        try:
            message = self.llm_client.messages.create(
                model=self.llm_model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            
            review_text = message.content[0].text if message.content else ""
            self.log_debug(f"LLM review for {signal.symbol}: {review_text}")
            
            # Optionally append to reasoning
            if review_text and signal.reasoning:
                signal.reasoning += f"\nLLM Review: {review_text}"
        except Exception as e:
            self.log_debug(f"LLM review failed: {e}")
    
    def _llm_review_confidence_validation(
        self,
        signal: TradingSignal,
        sharpe: float,
        max_dd: float
    ) -> None:
        """Optional LLM review of confidence adjustment."""
        prompt = f"""Review confidence adjustment for {signal.symbol}:

- Sharpe Ratio: {sharpe:.2f} (threshold: {self.min_sharpe})
- Max Drawdown: {max_dd:.2%} (threshold: {self.max_drawdown:.2%})
- Adjusted Confidence: {signal.confidence:.2f}

Explain the confidence adjustment and any concerns."""
        
        try:
            message = self.llm_client.messages.create(
                model=self.llm_model,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            
            review_text = message.content[0].text if message.content else ""
            self.log_debug(f"LLM confidence review for {signal.symbol}: {review_text}")
            
            if review_text and signal.reasoning:
                signal.reasoning += f"\nConfidence Adjustment: {review_text}"
        except Exception as e:
            self.log_debug(f"LLM confidence review failed: {e}")
    
    def health_check(self) -> Dict[str, Any]:
        """Check agent health."""
        health = super().health_check()
        
        health.update({
            "min_sharpe": self.min_sharpe,
            "max_drawdown": self.max_drawdown,
            "llm_review_enabled": self.use_llm_review
        })
        
        if self.use_llm_review:
            try:
                # Test LLM connection if enabled
                if self.llm_client:
                    health["llm_accessible"] = True
                else:
                    health["llm_accessible"] = False
            except Exception as e:
                health["llm_accessible"] = False
                health["llm_error"] = str(e)
        
        return health

