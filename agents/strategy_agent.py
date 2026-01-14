"""Strategy agent for analyzing market data and selecting trading strategies."""
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from groq import Groq
try:
    from groq import RateLimitError as GroqRateLimitError
except ImportError:
    # Fallback if RateLimitError is not directly importable
    GroqRateLimitError = None

from agents.base import BaseAgent
from models.market_data import MarketData
from models.signal import TradingSignal, SignalAction
from models.llm_schemas import LLMStrategySelection
from utils.exceptions import AgentError, StrategyError
from utils.retry import retry_with_backoff, RetryConfig
from pydantic import ValidationError
from core.strategies import STRATEGY_REGISTRY, BaseStrategy


# Predefined strategy templates - Strategy Agent can only select from these
# Map to concrete strategy classes
AVAILABLE_STRATEGIES = list(STRATEGY_REGISTRY.keys())


class StrategyAgent(BaseAgent):
    """
    Agent responsible for analyzing market data and selecting trading strategies.
    
    Role: Context interpreter, not alpha generator
    - Uses LLM (Groq) to interpret market context (volatility, trends, regime)
    - Selects from predefined strategy templates (no invention)
    - Generates trading signals with confidence scores
    
    Flow: Data ingestion → LLM-restricted selection → Output signals
    """
    
    def __init__(self, config=None):
        """
        Initialize the strategy agent.
        
        Args:
            config: Application configuration. If None, loads from environment.
        """
        super().__init__(config)
        
        # Validate Groq configuration
        if not self.config.groq:
            raise AgentError(
                "Groq configuration not found. Set GROQ_API_KEY in environment.",
                correlation_id=self._correlation_id
            )
        
        # Initialize Groq client
        try:
            self.groq_client = Groq(api_key=self.config.groq.api_key)
            self.model = self.config.groq.model or "llama-3.3-70b-versatile"
            self.temperature = self.config.groq.temperature
            self.log_info(f"StrategyAgent initialized with Groq ({self.model})")
        except Exception as e:
            error = self.handle_error(e, context={"provider": "groq"})
            raise StrategyError(
                "Failed to initialize Groq client",
                correlation_id=self._correlation_id
            ) from error
    
    def process(self, market_data: Dict[str, MarketData]) -> List[TradingSignal]:
        """
        Process market data and generate trading signals.
        
        This is the main entry point for the agent.
        
        Args:
            market_data: Dictionary mapping symbol to MarketData
        
        Returns:
            List of TradingSignal objects
        
        Raises:
            StrategyError: If signal generation fails
        """
        self.generate_correlation_id()
        self.log_info(
            f"Processing market data for {len(market_data)} symbols",
            symbols=list(market_data.keys())
        )
        
        signals = []
        
        for symbol, data in market_data.items():
            try:
                # Skip if no data
                if not data.bars or len(data.bars) < 2:
                    self.log_warning(f"Insufficient data for {symbol}, skipping")
                    continue
                
                # Calculate market context metrics
                context = self._calculate_market_context(symbol, data)
                
                # Use LLM to interpret context and select strategy
                strategy_selection = self._select_strategy_with_llm(symbol, context)
                
                # Generate signal based on strategy selection
                signal = self._generate_signal(symbol, data, strategy_selection, context)
                
                if signal:
                    signals.append(signal)
                    self.log_info(
                        f"Generated signal for {symbol}: {signal.action.value} "
                        f"using {signal.strategy_name} (confidence: {signal.confidence:.2f})"
                    )
            
            except Exception as e:
                self.log_exception(f"Failed to process signal for {symbol}", e)
                # Continue with other symbols instead of failing completely
                continue
        
        self.log_info(f"Generated {len(signals)} signals from {len(market_data)} symbols")
        return signals
    
    def _calculate_market_context(self, symbol: str, data: MarketData) -> Dict[str, Any]:
        """
        Calculate market context metrics from MarketData.
        
        Args:
            symbol: Stock symbol
            data: MarketData object
        
        Returns:
            Dictionary with market context metrics
        """
        df = data.to_dataframe()
        if df.empty or len(df) < 2:
            return {}
        
        closes = df['close'].values
        
        # Calculate basic metrics
        current_price = closes[-1]
        price_change = closes[-1] - closes[-2] if len(closes) >= 2 else 0
        price_change_pct = (price_change / closes[-2] * 100) if len(closes) >= 2 and closes[-2] != 0 else 0
        
        # Calculate volatility (rolling standard deviation)
        if len(closes) >= 20:
            volatility = float(np.std(closes[-20:]) / np.mean(closes[-20:]) * 100)
            volatility_short = float(np.std(closes[-5:]) / np.mean(closes[-5:]) * 100)
        elif len(closes) >= 5:
            volatility = float(np.std(closes) / np.mean(closes) * 100)
            volatility_short = float(np.std(closes[-5:]) / np.mean(closes[-5:]) * 100)
        else:
            volatility = 0.0
            volatility_short = 0.0
        
        # Calculate moving averages
        if len(closes) >= 20:
            ma_20 = float(np.mean(closes[-20:]))
            ma_50 = float(np.mean(closes[-min(50, len(closes)):])) if len(closes) >= 10 else ma_20
            ma_trend = "UP" if current_price > ma_20 else "DOWN"
        else:
            ma_20 = float(np.mean(closes))
            ma_50 = ma_20
            ma_trend = "NEUTRAL"
        
        # Calculate volume metrics
        volumes = df['volume'].values
        avg_volume = float(np.mean(volumes[-20:])) if len(volumes) >= 20 else float(np.mean(volumes))
        current_volume = float(volumes[-1]) if len(volumes) > 0 else 0
        volume_ratio = (current_volume / avg_volume) if avg_volume > 0 else 1.0
        
        # Calculate price range
        high = float(df['high'].max())
        low = float(df['low'].min())
        price_range = high - low
        price_position = ((current_price - low) / price_range * 100) if price_range > 0 else 50.0
        
        # Determine market regime
        if volatility > 3.0:
            regime = "HIGH_VOLATILITY"
        elif volatility < 1.0:
            regime = "LOW_VOLATILITY"
        else:
            regime = "NORMAL_VOLATILITY"
        
        if abs(price_change_pct) > 2.0:
            regime += "_MOMENTUM"
        
        context = {
            "symbol": symbol,
            "current_price": current_price,
            "price_change_pct": price_change_pct,
            "volatility": volatility,
            "volatility_short": volatility_short,
            "ma_20": ma_20,
            "ma_50": ma_50,
            "ma_trend": ma_trend,
            "volume_ratio": volume_ratio,
            "price_position_in_range": price_position,
            "regime": regime,
            "bars_count": len(closes),
            "recent_trend": "BULLISH" if price_change_pct > 0 else "BEARISH" if price_change_pct < 0 else "NEUTRAL"
        }
        
        self.log_debug(f"Market context for {symbol}", context=context)
        return context
    
    def _deterministic_fallback(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Smart fallback strategy selection based on market conditions when LLM fails.
        
        Args:
            symbol: Stock symbol
            context: Market context metrics
        
        Returns:
            Dictionary with selected strategy and reasoning
        """
        volatility = context.get("volatility", 0.0) / 100.0  # Convert percentage to decimal
        trend_strength = abs(context.get("price_change_pct", 0.0)) / 100.0  # Use absolute price change as trend strength
        trend_direction = context.get("price_change_pct", 0.0)
        
        self.log_warning(f"Using deterministic fallback for {symbol} (volatility={volatility:.4f}, trend_strength={trend_strength:.4f})")
        
        # High volatility -> Use mean reversion with caution
        if volatility > 0.03:  # 3% volatility threshold
            return {
                "strategy_name": "MeanReversion",
                "action": "HOLD",
                "confidence": 0.4,
                "reasoning": f"High volatility ({volatility:.2%}) - cautious hold with mean reversion strategy"
            }
        
        # Strong trend -> Use trend following
        elif trend_strength > 0.07:  # 7% price change indicates strong trend
            if trend_direction > 0:
                action = "BUY"
            elif trend_direction < 0:
                action = "SELL"
            else:
                action = "HOLD"
            
            return {
                "strategy_name": "MovingAverageCrossover",
                "action": action,
                "confidence": 0.6,
                "reasoning": f"Strong trend detected (change={trend_direction:.2%}) - trend following fallback"
            }
        
        # Neutral market -> Use mean reversion with hold
        else:
            return {
                "strategy_name": "MeanReversion",
                "action": "HOLD",
                "confidence": 0.5,
                "reasoning": f"Neutral market conditions - mean reversion hold (volatility={volatility:.2%}, trend_change={trend_direction:.2%})"
            }
    
    def _select_strategy_with_llm(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to interpret market context and select appropriate strategy.
        
        Args:
            symbol: Stock symbol
            context: Market context metrics
        
        Returns:
            Dictionary with selected strategy and reasoning
        """
        # Prepare prompt for LLM with strategy descriptions
        strategy_descriptions = {
            "TrendFollowing": "Moving average crossover (MA50/MA200) - good for trending markets",
            "MomentumRotation": "6-month momentum - good for strong directional moves",
            "MeanReversion": "RSI-based mean reversion - good for ranging markets",
            "Breakout": "Donchian channel breakouts - good for volatility expansion",
            "VolatilityBreakout": "ATR-based breakouts - good for high volatility periods",
            "RelativeStrength": "Performance vs benchmark - good for outperformance",
            "SectorRotation": "Momentum ranking - good for sector trends",
            "DualMomentum": "Absolute + relative momentum - good for confirmation",
            "MovingAverageEnvelope": "MA envelope bands - good for mean reversion",
            "BollingerBands": "Bollinger Bands reversion - good for overbought/oversold"
        }
        strategies_list = "\n".join([
            f"- {name}: {desc}" 
            for name, desc in strategy_descriptions.items()
            if name in STRATEGY_REGISTRY
        ])
        
        prompt = f"""You are a trading strategy selector. Analyze the market context and select the MOST APPROPRIATE strategy from the predefined list below.

AVAILABLE STRATEGIES (you MUST select one of these):
{strategies_list}

MARKET CONTEXT for {symbol}:
- Current Price: ${context.get('current_price', 0):.2f}
- Price Change: {context.get('price_change_pct', 0):.2f}%
- Volatility: {context.get('volatility', 0):.2f}%
- Short-term Volatility: {context.get('volatility_short', 0):.2f}%
- Moving Average Trend: {context.get('ma_trend', 'NEUTRAL')}
- Volume Ratio: {context.get('volume_ratio', 1.0):.2f}x
- Price Position in Range: {context.get('price_position_in_range', 50):.1f}%
- Market Regime: {context.get('regime', 'NORMAL')}
- Recent Trend: {context.get('recent_trend', 'NEUTRAL')}

INSTRUCTIONS:
1. Analyze the market context above
2. Select ONE strategy from the predefined list that best fits this market condition
3. Determine the action (BUY, SELL, or HOLD)
4. Provide a confidence score (0.0 to 1.0)
5. Explain your reasoning briefly

Respond in this EXACT JSON format (no markdown, no extra text):
{{
    "strategy_name": "StrategyNameFromList",
    "action": "BUY|SELL|HOLD",
    "confidence": 0.75,
    "reasoning": "Brief explanation of why this strategy fits the market context"
}}

IMPORTANT:
- You MUST select a strategy from the predefined list. Do not invent new strategies.
- Action should be BUY, SELL, or HOLD based on the strategy and market conditions.
- Confidence should reflect your certainty in the selection (0.0 = uncertain, 1.0 = very confident).
"""
        
        try:
            self.log_debug(f"Calling Groq LLM for {symbol} strategy selection")
            
            completion = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a professional trading strategy selector. You analyze market context and select the best strategy from a predefined list. You always respond in valid JSON format."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            response_text = completion.choices[0].message.content
            self.log_debug(f"LLM response for {symbol}", response=response_text)
            
            # Parse and validate JSON response using Pydantic schema
            try:
                # Parse JSON first
                json_data = json.loads(response_text)
                
                # Validate against Pydantic schema (ensures strategy is from predefined list)
                validated_response = LLMStrategySelection(**json_data)
                
                # Convert Pydantic model to dict for compatibility
                result = {
                    "strategy_name": validated_response.strategy_name,
                    "action": validated_response.action,
                    "confidence": validated_response.confidence,
                    "reasoning": validated_response.reasoning
                }
                
                self.log_debug(f"Validated LLM response for {symbol}", result=result)
                return result
                
            except json.JSONDecodeError as e:
                self.log_error(f"Invalid JSON in LLM response for {symbol}: {e}")
                # Raise as ValueError to be caught by outer handler
                raise ValueError(f"Failed to parse JSON response: {e}") from e
            except ValidationError as e:
                # Log validation errors with details for debugging
                self.log_error(
                    f"LLM response validation failed for {symbol}. "
                    f"Strategy or action not in allowed list. "
                    f"Errors: {e.errors()}, Raw response: {response_text[:200]}"
                )
                # Re-raise to be caught by outer handler
                raise
            
        except (ValidationError, ValueError) as e:
            # Schema validation failed - LLM returned invalid strategy/action
            self.log_error(
                f"LLM validation failed for {symbol}: {e}. "
                f"Using safe default strategy."
            )
            # Fallback to safe default strategy
            return {
                "strategy_name": "MovingAverageCrossover",
                "action": "HOLD",
                "confidence": 0.3,
                "reasoning": f"LLM validation failed - strategy not in allowed list. Using safe default."
            }
        except Exception as e:
            # Check if this is a rate limit error
            is_rate_limit = False
            rate_limit_message = None
            
            # Check for Groq RateLimitError
            if GroqRateLimitError and isinstance(e, GroqRateLimitError):
                is_rate_limit = True
                rate_limit_message = str(e)
            elif hasattr(e, 'status_code') and e.status_code == 429:
                is_rate_limit = True
                rate_limit_message = "Rate limit exceeded (429)"
            elif 'rate limit' in str(e).lower() or '429' in str(e) or 'RateLimitError' in str(type(e).__name__):
                is_rate_limit = True
                rate_limit_message = str(e)
            
            if is_rate_limit:
                self.log_warning(
                    f"Groq API rate limit reached for {symbol}. "
                    f"Using deterministic fallback. Error: {rate_limit_message}"
                )
                # Don't retry on rate limit - immediately fallback
                return self._deterministic_fallback(symbol, context)
            
            # For other exceptions, log and fallback
            self.log_exception(f"LLM strategy selection failed for {symbol}", e)
            # Smart fallback based on actual market conditions
            return self._deterministic_fallback(symbol, context)
    
    def _generate_signal(
        self,
        symbol: str,
        data: MarketData,
        strategy_selection: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """
        Generate a trading signal from strategy selection and market data.
        
        This method uses the concrete strategy implementation to generate
        the actual signal action deterministically.
        
        Args:
            symbol: Stock symbol
            data: MarketData object
            strategy_selection: Strategy selection from LLM
            context: Market context metrics
        
        Returns:
            TradingSignal or None if signal should not be generated
        """
        try:
            strategy_name = strategy_selection.get("strategy_name", "TrendFollowing")
            
            # Get strategy class from registry
            strategy_class = STRATEGY_REGISTRY.get(strategy_name)
            if not strategy_class:
                self.log_warning(
                    f"Strategy '{strategy_name}' not found in registry, using TrendFollowing"
                )
                strategy_class = STRATEGY_REGISTRY.get("TrendFollowing")
            
            # Instantiate strategy with config
            strategy_config = self.config.__dict__.get("strategy_config", {})
            strategy: BaseStrategy = strategy_class(config=strategy_config)
            
            # Generate signal using concrete strategy implementation
            try:
                action = strategy.generate_signal(data)
            except ValueError as e:
                # Insufficient data for strategy
                self.log_warning(
                    f"Strategy {strategy_name} failed for {symbol} due to insufficient data: {e}"
                )
                return None
            
            # Get current price
            current_price = context.get("current_price", 0.0)
            if current_price == 0.0 and data.bars:
                current_price = data.bars[-1].close
            
            # Create trading signal with LLM confidence and reasoning
            signal = TradingSignal(
                symbol=symbol,
                action=action,
                strategy_name=strategy_name,
                confidence=float(strategy_selection.get("confidence", 0.5)),
                timestamp=datetime.now(),
                price=current_price,
                reasoning=strategy_selection.get("reasoning", "")
            )
            
            # Add stop loss and take profit if BUY or SELL
            if action in [SignalAction.BUY, SignalAction.SELL]:
                volatility = context.get("volatility", 2.0)
                # Simple stop loss at 2% below/above entry
                if action == SignalAction.BUY:
                    signal.stop_loss = current_price * 0.98
                    signal.take_profit = current_price * (1.0 + max(0.02, volatility / 100))
                else:  # SELL
                    signal.stop_loss = current_price * 1.02
                    signal.take_profit = current_price * (1.0 - max(0.02, volatility / 100))
            
            return signal
            
        except Exception as e:
            self.log_exception(f"Failed to generate signal for {symbol}", e)
            return None
    
    def health_check(self) -> Dict[str, Any]:
        """Check agent health by testing Groq connection."""
        health = super().health_check()
        
        try:
            # Simple test call to Groq
            test_completion = self.groq_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Say 'OK'"}],
                max_tokens=10
            )
            
            health.update({
                "llm_provider": "groq",
                "model": self.model,
                "llm_accessible": True
            })
        except Exception as e:
            health.update({
                "status": "unhealthy",
                "llm_provider": "groq",
                "llm_accessible": False,
                "error": str(e)
            })
        
        return health

