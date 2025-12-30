"""Risk management agent for enforcing risk rules and position sizing."""
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

from agents.base import BaseAgent
from models.signal import TradingSignal, SignalAction
from utils.exceptions import RiskCheckError, AgentError
from core.strategies.indicators import calculate_atr


class RiskAgent(BaseAgent):
    """
    Agent responsible for enforcing risk rules and calculating position sizes.
    
    Role: Hard-coded rule enforcement with optional LLM advisor
    - Enforces maximum risk per trade (default: 2%)
    - Enforces daily loss limits
    - Calculates position sizing based on confidence and risk
    - Optional LLM (OpenAI/Claude) for explaining rejections and recommending alternatives
    
    Flow: Risk checks → Position sizing → Approved signals
    Focus: Unbreakable rules (code enforces, never LLM)
    """
    
    def __init__(self, config=None):
        """
        Initialize the risk agent.
        
        Args:
            config: Application configuration. If None, loads from environment.
        """
        super().__init__(config)
        
        # Risk rules
        self.max_risk_per_trade = float(self.config.__dict__.get('risk_max_per_trade', 0.02))  # 2%
        self.max_daily_loss = float(self.config.__dict__.get('risk_max_daily_loss', 0.05))  # 5%
        self.min_confidence = float(self.config.__dict__.get('risk_min_confidence', 0.3))  # Minimum confidence
        self.max_qty = int(self.config.__dict__.get('risk_max_qty', 1000))  # Maximum shares per trade
        
        # Account balance (will be fetched from ExecutionAgent)
        self._account_balance: Optional[float] = None
        self._current_daily_loss: float = 0.0  # In production, track via DB
        
        # Optional LLM advisor (OpenAI or Claude)
        self.use_llm_advisor = self.config.__dict__.get('risk_use_llm_advisor', False)
        self.llm_client = None
        
        if self.use_llm_advisor:
            # Prefer OpenAI, fallback to Claude
            if self.config.openai:
                try:
                    from openai import OpenAI
                    self.llm_client = OpenAI(api_key=self.config.openai.api_key)
                    self.llm_provider = "openai"
                    self.llm_model = self.config.openai.model or "gpt-4o"
                    self.log_info(f"RiskAgent initialized with OpenAI LLM advisor ({self.llm_model})")
                except Exception as e:
                    self.log_warning(f"Failed to initialize OpenAI for RiskAgent: {e}")
                    self.use_llm_advisor = False
            
            if not self.llm_client and self.config.anthropic:
                try:
                    from anthropic import Anthropic
                    self.llm_client = Anthropic(api_key=self.config.anthropic.api_key)
                    self.llm_provider = "anthropic"
                    self.llm_model = self.config.anthropic.model or "claude-3-opus-20240229"
                    self.log_info(f"RiskAgent initialized with Claude LLM advisor ({self.llm_model})")
                except Exception as e:
                    self.log_warning(f"Failed to initialize Claude for RiskAgent: {e}")
                    self.use_llm_advisor = False
        
        if not self.use_llm_advisor:
            self.log_info("RiskAgent initialized (code-only mode, no LLM advisor)")
    
    def process(
        self,
        signals: List[TradingSignal],
        execution_agent: Optional[Any] = None
    ) -> List[TradingSignal]:
        """
        Process signals and enforce risk rules.
        
        This is the main entry point for the agent.
        
        Args:
            signals: List of validated TradingSignal objects from QuantAgent
            execution_agent: Optional ExecutionAgent to fetch account balance
        
        Returns:
            List of approved TradingSignal objects with position sizing
        
        Raises:
            RiskCheckError: If critical risk violations occur
        """
        self.generate_correlation_id()
        self.log_info(
            f"Processing {len(signals)} signals for risk validation",
            signal_count=len(signals)
        )
        
        # Fetch account balance if available
        if execution_agent and self._account_balance is None:
            try:
                account = execution_agent.get_account()
                self._account_balance = float(account.equity) if hasattr(account, 'equity') else float(account.cash)
                self.log_info(f"Account balance: ${self._account_balance:,.2f}")
            except Exception as e:
                self.log_warning(f"Failed to fetch account balance: {e}, using default")
                self._account_balance = float(self.config.__dict__.get('risk_default_account_balance', 10000.0))
        elif self._account_balance is None:
            self._account_balance = float(self.config.__dict__.get('risk_default_account_balance', 10000.0))
            self.log_info(f"Using default account balance: ${self._account_balance:,.2f}")
        
        approved_signals = []
        
        for signal in signals:
            try:
                # Skip HOLD signals
                if signal.action == SignalAction.HOLD:
                    self.log_debug(f"Skipping HOLD signal for {signal.symbol}")
                    approved_signals.append(signal)
                    continue
                
                # Enforce risk rules
                self.enforce_rules(signal)
                
                # Calculate position sizing
                self.calculate_position_sizing(signal)
                
                # Mark as approved
                signal.approved = True
                approved_signals.append(signal)
                
                self.log_info(
                    f"Signal approved for {signal.symbol}: qty={signal.qty}, "
                    f"risk_amount=${signal.risk_amount:.2f}"
                )
                
            except RiskCheckError as e:
                self.log_warning(
                    f"Risk violation for {signal.symbol}: {e.message}",
                    symbol=signal.symbol,
                    reason=str(e)
                )
                
                # Optional LLM advisor for rejection explanation
                if self.use_llm_advisor and self.llm_client:
                    try:
                        self._llm_advise_rejection(signal, str(e))
                    except Exception as llm_error:
                        self.log_debug(f"LLM advisor failed: {llm_error}")
                
                # Reject signal (don't add to approved list)
                signal.approved = False
                continue
                
            except Exception as e:
                self.log_exception(f"Unexpected error processing {signal.symbol}", e)
                # Reject on unexpected errors
                signal.approved = False
                continue
        
        self.log_info(
            f"Risk validation completed: {len(approved_signals)}/{len(signals)} signals approved",
            approved_count=len(approved_signals),
            rejected_count=len(signals) - len(approved_signals)
        )
        
        return approved_signals
    
    def enforce_rules(self, signal: TradingSignal) -> None:
        """
        Enforce hard-coded risk rules.
        
        Rules:
        1. Minimum confidence threshold
        2. Maximum risk per trade (2%)
        3. Daily loss limit
        
        Args:
            signal: TradingSignal to validate
        
        Raises:
            RiskCheckError: If rules are violated
        """
        # Rule 1: Minimum confidence
        if signal.confidence < self.min_confidence:
            raise RiskCheckError(
                f"Confidence {signal.confidence:.2f} below minimum {self.min_confidence}",
                correlation_id=self._correlation_id,
                details={"confidence": signal.confidence, "min_confidence": self.min_confidence}
            )
        
        # Rule 2: Maximum risk per trade (calculated after position sizing)
        # This check happens after calculate_position_sizing
        
        # Rule 3: Daily loss limit check (also after position sizing)
        # This check happens after calculate_position_sizing
        
        self.log_debug(f"Risk rules check passed for {signal.symbol}")
    
    def calculate_position_sizing(self, signal: TradingSignal) -> None:
        """
        Calculate position size based on risk and confidence.
        
        Position sizing:
        - Base risk: max_risk_per_trade * account_balance * confidence
        - Stop distance: ATR-based (1.5x ATR)
        - Quantity: risk_amount / stop_distance_per_share
        
        After calculation, validates against max_risk_per_trade and daily limits.
        
        Args:
            signal: TradingSignal to size
        
        Raises:
            RiskCheckError: If sizing violates risk rules
        """
        if signal.historical_data is None or signal.historical_data.empty:
            raise RiskCheckError(
                "No historical data for position sizing",
                correlation_id=self._correlation_id
            )
        
        df = signal.historical_data.copy()
        
        # Ensure we have required columns
        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            raise RiskCheckError(
                "Missing required columns (high, low, close) for position sizing",
                correlation_id=self._correlation_id
            )
        
        # Calculate ATR for stop distance
        if len(df) < 14:
            # Fallback: use simple high-low range
            atr = (df['high'] - df['low']).mean()
        else:
            try:
                atr = calculate_atr(df['high'], df['low'], df['close'], period=14).iloc[-1]
                if pd.isna(atr):
                    atr = (df['high'] - df['low']).mean()
            except Exception as e:
                self.log_warning(f"ATR calculation failed for {signal.symbol}, using fallback: {e}")
                atr = (df['high'] - df['low']).mean()
        
        # Stop distance per share (conservative: 1.5x ATR)
        stop_distance = atr * 1.5
        
        if stop_distance <= 0:
            raise RiskCheckError(
                f"Invalid stop distance calculated: {stop_distance}",
                correlation_id=self._correlation_id
            )
        
        # Base risk amount (scaled by confidence)
        # Higher confidence = larger position, but still capped at max_risk_per_trade
        risk_amount = self.max_risk_per_trade * self._account_balance * signal.confidence
        
        # Ensure risk doesn't exceed absolute maximum
        max_risk_absolute = self.max_risk_per_trade * self._account_balance
        risk_amount = min(risk_amount, max_risk_absolute)
        
        # Calculate quantity: risk_amount / stop_distance_per_share
        qty = risk_amount / stop_distance
        
        # Clamp to reasonable bounds
        qty = max(1, min(int(qty), self.max_qty))
        
        # Recalculate actual risk with integer qty
        actual_risk_amount = qty * stop_distance
        
        # Validate against max risk per trade (Rule 2)
        if actual_risk_amount > max_risk_absolute:
            raise RiskCheckError(
                f"Position sizing exceeds max risk: ${actual_risk_amount:.2f} > ${max_risk_absolute:.2f}",
                correlation_id=self._correlation_id,
                details={
                    "calculated_risk": actual_risk_amount,
                    "max_risk": max_risk_absolute,
                    "qty": qty,
                    "stop_distance": stop_distance
                }
            )
        
        # Validate against daily loss limit (Rule 3)
        max_daily_loss_absolute = self.max_daily_loss * self._account_balance
        potential_daily_loss = self._current_daily_loss + actual_risk_amount
        
        if potential_daily_loss > max_daily_loss_absolute:
            raise RiskCheckError(
                f"Would exceed daily loss limit: ${potential_daily_loss:.2f} > ${max_daily_loss_absolute:.2f} "
                f"(current daily loss: ${self._current_daily_loss:.2f})",
                correlation_id=self._correlation_id,
                details={
                    "potential_daily_loss": potential_daily_loss,
                    "max_daily_loss": max_daily_loss_absolute,
                    "current_daily_loss": self._current_daily_loss
                }
            )
        
        # Set signal properties
        signal.qty = qty
        signal.risk_amount = actual_risk_amount
        
        # Update stop loss if not set
        if signal.stop_loss is None and signal.price:
            if signal.action == SignalAction.BUY:
                signal.stop_loss = signal.price - stop_distance
            elif signal.action == SignalAction.SELL:
                signal.stop_loss = signal.price + stop_distance
        
        # Track daily loss (in production, persist to DB)
        self._current_daily_loss += actual_risk_amount
        
        # Optional LLM advisor for sizing rationale
        if self.use_llm_advisor and self.llm_client:
            try:
                self._llm_advise_sizing(signal, stop_distance, actual_risk_amount)
            except Exception as e:
                self.log_debug(f"LLM sizing advice failed: {e}")
        
        self.log_debug(
            f"Position sizing for {signal.symbol}: qty={qty}, "
            f"risk=${actual_risk_amount:.2f}, stop_distance=${stop_distance:.2f}"
        )
    
    def _llm_advise_rejection(self, signal: TradingSignal, reason: str) -> None:
        """
        Optional LLM advisor explains rejection and suggests alternatives.
        
        Args:
            signal: Rejected signal
            reason: Reason for rejection
        """
        prompt = f"""Trading signal rejected for {signal.symbol}:

Signal Details:
- Action: {signal.action.value}
- Confidence: {signal.confidence:.2f}
- Strategy: {signal.strategy_name}
- Price: ${signal.price:.2f if signal.price else 'N/A'}

Rejection Reason: {reason}

Risk Rules:
- Max Risk Per Trade: {self.max_risk_per_trade*100:.1f}%
- Max Daily Loss: {self.max_daily_loss*100:.1f}%
- Account Balance: ${self._account_balance:,.2f}

Provide:
1. Brief explanation of why this signal was rejected
2. What conditions would need to change for it to be approved
3. Any alternative approaches or adjustments

Keep response concise (2-3 sentences)."""
        
        try:
            if self.llm_provider == "openai":
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200
                )
                advice = response.choices[0].message.content
            else:  # anthropic
                message = self.llm_client.messages.create(
                    model=self.llm_model,
                    max_tokens=200,
                    messages=[{"role": "user", "content": prompt}]
                )
                advice = message.content[0].text if message.content else ""
            
            self.log_info(f"LLM advice for rejected {signal.symbol}: {advice}")
            
            # Optionally store advice in signal reasoning
            if signal.reasoning:
                signal.reasoning += f"\nRisk Rejection Advice: {advice}"
            else:
                signal.reasoning = f"Risk Rejection Advice: {advice}"
                
        except Exception as e:
            self.log_debug(f"LLM rejection advice failed: {e}")
    
    def _llm_advise_sizing(self, signal: TradingSignal, stop_distance: float, risk_amount: float) -> None:
        """
        Optional LLM advisor provides sizing rationale.
        
        Args:
            signal: Approved signal with sizing
            stop_distance: Calculated stop distance
            risk_amount: Risk amount for this trade
        """
        prompt = f"""Position sizing advice for {signal.symbol}:

Signal Details:
- Action: {signal.action.value}
- Confidence: {signal.confidence:.2f}
- Quantity: {signal.qty} shares
- Risk Amount: ${risk_amount:.2f}
- Stop Distance: ${stop_distance:.2f}
- Price: ${signal.price:.2f if signal.price else 'N/A'}

Risk Context:
- Account Balance: ${self._account_balance:,.2f}
- Risk as % of account: {(risk_amount/self._account_balance)*100:.2f}%
- Current Daily Loss: ${self._current_daily_loss:.2f}

Provide brief assessment of the position sizing (1-2 sentences)."""
        
        try:
            if self.llm_provider == "openai":
                response = self.llm_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=150
                )
                advice = response.choices[0].message.content
            else:  # anthropic
                message = self.llm_client.messages.create(
                    model=self.llm_model,
                    max_tokens=150,
                    messages=[{"role": "user", "content": prompt}]
                )
                advice = message.content[0].text if message.content else ""
            
            self.log_debug(f"LLM sizing advice for {signal.symbol}: {advice}")
            
            if signal.reasoning:
                signal.reasoning += f"\nPosition Sizing: {advice}"
            else:
                signal.reasoning = f"Position Sizing: {advice}"
                
        except Exception as e:
            self.log_debug(f"LLM sizing advice failed: {e}")
    
    def reset_daily_loss(self) -> None:
        """Reset daily loss tracking (call at start of trading day)."""
        self.log_info(f"Resetting daily loss: {self._current_daily_loss:.2f} → 0.0")
        self._current_daily_loss = 0.0
    
    def health_check(self) -> Dict[str, Any]:
        """Check agent health."""
        health = super().health_check()
        
        health.update({
            "max_risk_per_trade": self.max_risk_per_trade,
            "max_daily_loss": self.max_daily_loss,
            "min_confidence": self.min_confidence,
            "account_balance": self._account_balance,
            "current_daily_loss": self._current_daily_loss,
            "llm_advisor_enabled": self.use_llm_advisor
        })
        
        return health

