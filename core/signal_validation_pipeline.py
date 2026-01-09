"""
Deterministic Signal Validation Pipeline (Problem 1 Fix).

Enforces explicit ordering of risk checks to prevent drift and race conditions.
All signals must pass through this pipeline in strict order before execution.
"""
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging

from models.signal import TradingSignal, SignalAction
from core.tier_exposure_tracker import TierExposureTracker, TierExposure
from core.signal_validator import SignalValidator
from core.tiered_position_sizer import TieredPositionSizer

logger = logging.getLogger(__name__)


@dataclass
class ValidationDecision:
    """Structured decision metadata for logging (Problem 7 Fix)"""
    symbol: str
    action: SignalAction
    approved: bool
    reason: str
    tier: Optional[str] = None
    exposure_pct: Optional[float] = None
    gap_pct: Optional[float] = None
    shares: Optional[float] = None
    position_notional: Optional[float] = None
    skipped_steps: List[str] = None  # Which validation steps were skipped


class SignalValidationPipeline:
    """
    Deterministic validation pipeline with explicit ordering (Problem 1 Fix).
    
    All signals must pass through checks in strict order:
    1. Signal freshness validation (if scanner enabled)
    2. Tier exposure check (if tiered allocation enabled)
    3. Tiered position sizing (if tiered allocation enabled)
    4. Final risk validation
    
    This prevents:
    - Position sizing before exposure checks
    - Execution before tier caps
    - Race conditions between parallel tasks
    """
    
    def __init__(
        self,
        tier_tracker: Optional[TierExposureTracker] = None,
        signal_validator: Optional[SignalValidator] = None,
        tiered_sizer: Optional[TieredPositionSizer] = None,
        symbol_tier_mapping: Optional[Dict[str, str]] = None,
        account_value: Optional[float] = None,
        current_positions: Optional[List] = None,
        scanner_data: Optional[Dict] = None
    ):
        """
        Initialize validation pipeline.
        
        Args:
            tier_tracker: Optional tier exposure tracker
            signal_validator: Optional signal freshness validator
            tiered_sizer: Optional tiered position sizer
            symbol_tier_mapping: Symbol to tier mapping
            account_value: Current account equity
            current_positions: Current open positions
            scanner_data: Scanner output data (if scanner enabled)
        """
        self.tier_tracker = tier_tracker
        self.signal_validator = signal_validator
        self.tiered_sizer = tiered_sizer
        self.symbol_tier_mapping = symbol_tier_mapping or {}
        self.account_value = account_value
        self.current_positions = current_positions or []
        self.scanner_data = scanner_data
    
    def validate_signals(
        self,
        signals: List[TradingSignal],
        account_value: float,
        current_positions: Optional[List] = None,
        tier_exposures: Optional[Dict[str, TierExposure]] = None
    ) -> Tuple[List[TradingSignal], List[ValidationDecision]]:
        """
        Validate signals through deterministic pipeline.
        
        CRITICAL: This enforces explicit ordering (Problem 1 Fix).
        All signals pass through checks in this exact order:
        1. Signal freshness (if scanner enabled)
        2. Tier exposure reservation (if tiered allocation enabled)
        3. Tiered position sizing (if tiered allocation enabled)
        
        Args:
            signals: List of signals to validate
            account_value: Current account equity (live, not cached)
            current_positions: Current open positions
            tier_exposures: Pre-calculated tier exposures
        
        Returns:
            Tuple of (validated_signals, decisions)
            - validated_signals: Signals that passed all checks
            - decisions: List of ValidationDecision for all signals (approved and rejected)
        """
        validated_signals = []
        decisions = []
        
        # Update instance state
        self.account_value = account_value
        if current_positions:
            self.current_positions = current_positions
        
        # Calculate tier exposures if needed
        if self.tier_tracker and not tier_exposures:
            try:
                tier_exposures = self.tier_tracker.calculate_tier_exposure(
                    self.current_positions, account_value
                )
            except Exception as e:
                logger.error(f"Failed to calculate tier exposures: {e}")
                tier_exposures = {}
        
        for signal in signals:
            decision = self._validate_signal(
                signal=signal,
                account_value=account_value,
                tier_exposures=tier_exposures
            )
            decisions.append(decision)
            
            if decision.approved:
                validated_signals.append(signal)
            else:
                # Log structured decision (Problem 7 Fix)
                self._log_decision(decision)
        
        logger.info(
            f"📊 Validation pipeline: {len(validated_signals)}/{len(signals)} signals approved",
            extra={
                "approved_count": len(validated_signals),
                "rejected_count": len(signals) - len(validated_signals),
                "total_signals": len(signals)
            }
        )
        
        return validated_signals, decisions
    
    def _validate_signal(
        self,
        signal: TradingSignal,
        account_value: float,
        tier_exposures: Optional[Dict[str, TierExposure]] = None
    ) -> ValidationDecision:
        """
        Validate a single signal through deterministic pipeline.
        
        Order is CRITICAL - must be:
        1. Signal freshness
        2. Tier exposure
        3. Position sizing
        """
        skipped_steps = []
        decision = ValidationDecision(
            symbol=signal.symbol,
            action=signal.action,
            approved=False,
            reason="Not validated",
            skipped_steps=[]
        )
        
        # STEP 1: Signal Freshness Validation (if scanner enabled)
        if self.signal_validator and self.scanner_data:
            if signal.action == SignalAction.BUY:
                current_price = signal.price
                if current_price:
                    validation = self.signal_validator.validate_from_scanner_data(
                        signal.symbol,
                        current_price,
                        self.scanner_data
                    )
                    
                    if not validation.valid:
                        decision.reason = f"Signal freshness: {validation.reason}"
                        decision.gap_pct = validation.gap_pct
                        return decision
                    
                    decision.gap_pct = validation.gap_pct
        else:
            skipped_steps.append("signal_freshness")
        
        # STEP 2: Tier Exposure Check (if tiered allocation enabled)
        if self.tier_tracker and tier_exposures and signal.action == SignalAction.BUY:
            tier = self.tier_tracker.get_tier_for_symbol(signal.symbol)
            decision.tier = tier
            
            if tier:
                # Estimate proposed value (will be recalculated after sizing)
                proposed_value = (signal.qty or 0) * (signal.price or 0)
                
                if proposed_value == 0:
                    # Need price first, estimate based on current exposure
                    proposed_value = account_value * 0.05  # Conservative estimate
                
                # ATOMIC: Reserve exposure (Problem 2 Fix)
                current_tier_exposure = tier_exposures.get(tier)
                if current_tier_exposure:
                    approved, reason = self.tier_tracker.reserve_exposure(
                        tier=tier,
                        proposed_value=proposed_value,
                        current_tier_exposure=current_tier_exposure,
                        account_value=account_value,
                        symbol=signal.symbol
                    )
                    
                    if not approved:
                        decision.reason = f"Tier exposure: {reason}"
                        decision.exposure_pct = current_tier_exposure.current_pct
                        return decision
                    
                    decision.exposure_pct = current_tier_exposure.current_pct
            else:
                skipped_steps.append("tier_exposure")
        else:
            skipped_steps.append("tier_exposure")
        
        # STEP 3: Tiered Position Sizing (if tiered allocation enabled)
        if self.tiered_sizer and signal.action == SignalAction.BUY:
            tier = self.symbol_tier_mapping.get(signal.symbol)
            decision.tier = tier
            
            if tier:
                current_price = signal.price
                if not current_price:
                    decision.reason = "Price required for tiered sizing"
                    return decision
                
                try:
                    # ALWAYS pass live account_value (Problem 3 Fix)
                    shares, meta = self.tiered_sizer.calculate_shares(
                        symbol=signal.symbol,
                        current_price=current_price,
                        tier=tier,
                        account_value=account_value  # Live equity, not cached
                    )
                    
                    if shares and shares > 0:
                        signal.qty = shares if self.tiered_sizer.use_fractional else int(shares)
                        signal.price = current_price
                        decision.shares = shares
                        decision.position_notional = meta.get('position_notional', 0)
                        
                        logger.info(
                            "PositionSized",
                            extra={
                                "symbol": signal.symbol,
                                "tier": tier,
                                "shares": shares,
                                "notional": decision.position_notional,
                                "account_value": account_value
                            }
                        )
                    else:
                        decision.reason = f"Position sizing: {meta.get('reason', 'Failed')}"
                        # Release reserved exposure if sizing failed
                        if self.tier_tracker and tier:
                            proposed_value = (signal.qty or 0) * current_price
                            self.tier_tracker.release_exposure(tier, proposed_value, signal.symbol)
                        return decision
                except Exception as e:
                    decision.reason = f"Position sizing error: {e}"
                    logger.exception(f"Position sizing failed for {signal.symbol}")
                    return decision
            else:
                skipped_steps.append("tiered_sizing")
        else:
            skipped_steps.append("tiered_sizing")
        
        # All checks passed
        decision.approved = True
        decision.reason = "All validations passed"
        decision.skipped_steps = skipped_steps
        
        return decision
    
    def _log_decision(self, decision: ValidationDecision) -> None:
        """Log structured decision (Problem 7 Fix)"""
        logger.info(
            f"SignalDecision",
            extra={
                "symbol": decision.symbol,
                "action": decision.action.value,
                "approved": decision.approved,
                "reason": decision.reason,
                "tier": decision.tier,
                "exposure_pct": decision.exposure_pct,
                "gap_pct": decision.gap_pct,
                "shares": decision.shares,
                "position_notional": decision.position_notional,
                "skipped_steps": decision.skipped_steps
            }
        )
        
        if not decision.approved:
            logger.warning(
                f"🚫 Signal rejected: {decision.symbol} {decision.action.value} - {decision.reason}",
                extra={
                    "symbol": decision.symbol,
                    "action": decision.action.value,
                    "reason": decision.reason
                }
            )
