"""Strategy agent for analyzing market data and selecting trading strategies."""
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from groq import Groq

from agents.base import BaseAgent
from models.market_data import MarketData
from models.signal import TradingSignal, SignalAction
from utils.exceptions import AgentError, StrategyError
from utils.retry import retry_with_backoff, RetryConfig


# Predefined strategy templates - Strategy Agent can only select from these
AVAILABLE_STRATEGIES = [
    "MovingAverageCrossover",
    "MeanReversion",
    "Breakout",
    "Momentum",
    "TrendFollowing",
    "VolumeProfile",
    "RSI_OversoldOverbought",
    "BollingerBands",
    "SupportResistance",
    "ConsolidationBreakout"
]


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
            self.model = self.config.groq.model or "mixtral-8x7b-32768"
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
    
    @retry_with_backoff(config=RetryConfig(max_attempts=3, initial_delay=1.0))
    def _select_strategy_with_llm(self, symbol: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to interpret market context and select appropriate strategy.
        
        Args:
            symbol: Stock symbol
            context: Market context metrics
        
        Returns:
            Dictionary with selected strategy and reasoning
        """
        # Prepare prompt for LLM
        strategies_list = "\n".join([f"- {s}" for s in AVAILABLE_STRATEGIES])
        
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
            
            # Parse JSON response
            result = json.loads(response_text)
            
            # Validate strategy name is in available list
            strategy_name = result.get("strategy_name", "")
            if strategy_name not in AVAILABLE_STRATEGIES:
                self.log_warning(
                    f"LLM selected invalid strategy '{strategy_name}' for {symbol}, "
                    f"defaulting to 'MovingAverageCrossover'"
                )
                result["strategy_name"] = "MovingAverageCrossover"
            
            # Validate action
            action = result.get("action", "HOLD").upper()
            if action not in ["BUY", "SELL", "HOLD"]:
                self.log_warning(f"Invalid action '{action}' for {symbol}, defaulting to HOLD")
                result["action"] = "HOLD"
            
            # Validate confidence
            confidence = float(result.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
            result["confidence"] = confidence
            
            return result
            
        except Exception as e:
            self.log_exception(f"LLM strategy selection failed for {symbol}", e)
            # Fallback to default strategy
            return {
                "strategy_name": "MovingAverageCrossover",
                "action": "HOLD",
                "confidence": 0.3,
                "reasoning": f"LLM selection failed, using default strategy. Error: {str(e)}"
            }
    
    def _generate_signal(
        self,
        symbol: str,
        data: MarketData,
        strategy_selection: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Optional[TradingSignal]:
        """
        Generate a trading signal from strategy selection and market data.
        
        Args:
            symbol: Stock symbol
            data: MarketData object
            strategy_selection: Strategy selection from LLM
            context: Market context metrics
        
        Returns:
            TradingSignal or None if signal should not be generated
        """
        try:
            # Determine action
            action_str = strategy_selection.get("action", "HOLD").upper()
            action = SignalAction[action_str] if action_str in SignalAction.__members__ else SignalAction.HOLD
            
            # Get current price
            current_price = context.get("current_price", 0.0)
            if current_price == 0.0 and data.bars:
                current_price = data.bars[-1].close
            
            # Generate signal
            signal = TradingSignal(
                symbol=symbol,
                action=action,
                strategy_name=strategy_selection.get("strategy_name", "MovingAverageCrossover"),
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

