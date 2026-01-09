"""
Tier Exposure Tracker - Prevents tier allocation drift over time.

Critical protection against portfolio drift where winning positions
cause tier exposure to exceed configured caps.
"""
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
from datetime import datetime
import logging
import threading
import asyncio

logger = logging.getLogger(__name__)


@dataclass
class TierExposure:
    """Current exposure for a tier"""
    tier: str
    current_value: float
    current_pct: float
    cap_pct: float
    available_pct: float
    positions: List[str]  # List of symbols in this tier


class TierExposureTracker:
    """
    Tracks and enforces tier exposure caps across the portfolio.
    
    This is portfolio-aware - checks total tier exposure including
    existing positions that may have appreciated/depreciated.
    """
    
    TIER_CAPS = {
        'TIER1': 0.40,  # 40% max
        'TIER2': 0.30,  # 30% max
        'TIER3': 0.30,  # 30% max
    }
    
    # Safety buffer - reject if would exceed cap by this much
    SAFETY_BUFFER = 0.05  # 5% buffer
    
    def __init__(self, symbol_tier_mapping: Dict[str, str]):
        """
        Initialize tracker.
        
        Args:
            symbol_tier_mapping: Dict mapping symbols to tiers
                                 e.g. {'SPY': 'TIER1', 'AAPL': 'TIER3'}
        """
        self.symbol_tier_mapping = symbol_tier_mapping
        self.last_check = None
        
        # Atomic locking for race condition prevention (Problem 2 Fix)
        self._lock = threading.Lock()  # For sync operations
        self._async_lock = asyncio.Lock() if hasattr(asyncio, 'Lock') else None  # For async operations
        self._reserved_exposures: Dict[str, float] = {}  # Track reserved exposure by tier
        self._reserved_symbols: Dict[str, Set[str]] = {}  # Track which symbols have reserved exposure per tier (for debugging)
    
    def calculate_tier_exposure(
        self,
        positions,
        account_value: float
    ) -> Dict[str, TierExposure]:
        """
        Calculate current exposure for each tier.
        
        Args:
            positions: List of Position objects from broker
            account_value: Current account equity
        
        Returns:
            Dict of tier -> TierExposure
        """
        if account_value <= 0:
            raise ValueError(f"Invalid account value: {account_value}")
        
        tier_exposures = {}
        
        for tier, cap in self.TIER_CAPS.items():
            # Find all positions in this tier
            tier_value = 0.0
            tier_positions = []
            
            for pos in positions:
                symbol = pos.symbol
                if self.symbol_tier_mapping.get(symbol) == tier:
                    position_value = float(pos.qty) * float(pos.current_price)
                    tier_value += position_value
                    tier_positions.append(symbol)
            
            current_pct = (tier_value / account_value) if account_value > 0 else 0.0
            available_pct = cap - current_pct
            
            tier_exposures[tier] = TierExposure(
                tier=tier,
                current_value=tier_value,
                current_pct=current_pct,
                cap_pct=cap,
                available_pct=available_pct,
                positions=tier_positions
            )
            
            logger.debug(
                f"{tier}: ${tier_value:,.0f} ({current_pct:.1%}) / "
                f"{cap:.1%} cap, {len(tier_positions)} positions"
            )
        
        self.last_check = datetime.now()
        return tier_exposures
    
    def check_tier_capacity(
        self,
        tier: str,
        proposed_value: float,
        current_tier_exposure: TierExposure,
        account_value: float
    ) -> tuple[bool, str]:
        """
        Check if adding a position would exceed tier cap.
        
        DEPRECATED: Use reserve_exposure() for atomic operations.
        This method is kept for backward compatibility.
        
        Args:
            tier: Tier name (TIER1, TIER2, TIER3)
            proposed_value: Dollar value of proposed position
            proposed_value: Dollar value of proposed position
            current_tier_exposure: Current TierExposure for this tier
            account_value: Current account equity
        
        Returns:
            (approved: bool, reason: str)
        """
        if tier not in self.TIER_CAPS:
            return False, f"Unknown tier: {tier}"
        
        cap = self.TIER_CAPS[tier]
        current_pct = current_tier_exposure.current_pct
        proposed_pct = (proposed_value / account_value) if account_value > 0 else 0
        new_total_pct = current_pct + proposed_pct
        
        # Check against cap + buffer
        max_allowed = cap + self.SAFETY_BUFFER
        
        if new_total_pct > max_allowed:
            reason = (
                f"Tier {tier} capacity exceeded: "
                f"Current={current_pct:.1%}, "
                f"Proposed={proposed_pct:.1%}, "
                f"New Total={new_total_pct:.1%}, "
                f"Cap={cap:.1%} (+{self.SAFETY_BUFFER:.1%} buffer)"
            )
            logger.warning(f"🚫 {reason}")
            return False, reason
        
        # Warn if getting close to cap
        if new_total_pct > cap:
            logger.warning(
                f"⚠️  {tier} approaching cap: {new_total_pct:.1%} / {cap:.1%} "
                f"(within buffer)"
            )
        
        logger.info(
            f"✅ {tier} capacity OK: {new_total_pct:.1%} / {cap:.1%} "
            f"(available: {cap - new_total_pct:.1%})"
        )
        
        return True, "Approved"
    
    def reserve_exposure(
        self,
        tier: str,
        proposed_value: float,
        current_tier_exposure: TierExposure,
        account_value: float,
        symbol: str
    ) -> tuple[bool, str]:
        """
        ATOMIC: Reserve tier exposure before execution (Problem 2 Fix).
        
        This prevents race conditions where multiple signals pass checks
        before either order executes, causing tier cap violations.
        
        Args:
            tier: Tier name
            proposed_value: Dollar value of proposed position
            current_tier_exposure: Current TierExposure for this tier
            account_value: Current account equity
            symbol: Symbol for tracking
        
        Returns:
            (approved: bool, reason: str)
        """
        with self._lock:
            # Check capacity including already reserved exposure
            reserved_for_tier = self._reserved_exposures.get(tier, 0.0)
            
            cap = self.TIER_CAPS.get(tier)
            if not cap:
                return False, f"Unknown tier: {tier}"
            
            current_pct = current_tier_exposure.current_pct
            proposed_pct = (proposed_value / account_value) if account_value > 0 else 0
            reserved_pct = (reserved_for_tier / account_value) if account_value > 0 else 0
            new_total_pct = current_pct + reserved_pct + proposed_pct
            
            max_allowed = cap + self.SAFETY_BUFFER
            
            if new_total_pct > max_allowed:
                reason = (
                    f"Tier {tier} capacity exceeded (atomic check): "
                    f"Current={current_pct:.1%}, "
                    f"Reserved={reserved_pct:.1%}, "
                    f"Proposed={proposed_pct:.1%}, "
                    f"Total={new_total_pct:.1%}, "
                    f"Cap={cap:.1%} (+{self.SAFETY_BUFFER:.1%} buffer)"
                )
                logger.warning(f"🚫 {symbol}: {reason}")
                return False, reason
            
            # Reserve the exposure atomically
            self._reserved_exposures[tier] = reserved_for_tier + proposed_value
            
            # Track reserved symbols for debugging (optional hardening)
            if tier not in self._reserved_symbols:
                self._reserved_symbols[tier] = set()
            self._reserved_symbols[tier].add(symbol)
            
            logger.info(
                f"✅ {symbol} ({tier}): Exposure reserved atomically. "
                f"Total reserved for {tier}: ${self._reserved_exposures[tier]:,.0f}"
            )
            return True, "Reserved"
    
    def release_exposure(self, tier: str, value: float, symbol: str) -> None:
        """
        Release reserved exposure after execution (success or failure).
        
        Includes safeguard against double-release for forensic debugging.
        
        Args:
            tier: Tier name
            value: Dollar value to release
            symbol: Symbol for logging
        """
        with self._lock:
            # Safeguard against double-release (optional hardening)
            if tier not in self._reserved_symbols or symbol not in self._reserved_symbols[tier]:
                logger.error(
                    f"🚨 Double-release detected: {symbol} ({tier}) was not reserved",
                    extra={
                        "symbol": symbol,
                        "tier": tier,
                        "value": value,
                        "reserved_symbols": list(self._reserved_symbols.get(tier, set())),
                        "current_reserved": self._reserved_exposures.get(tier, 0.0)
                    }
                )
                # Don't throw - allow release to continue for resilience
                # This is just for forensic debugging
            
            current_reserved = self._reserved_exposures.get(tier, 0.0)
            new_reserved = max(0.0, current_reserved - value)
            self._reserved_exposures[tier] = new_reserved
            
            # Remove from reserved symbols tracking
            if tier in self._reserved_symbols:
                self._reserved_symbols[tier].discard(symbol)
            
            logger.debug(
                f"🔓 {symbol} ({tier}): Released ${value:,.0f}. "
                f"Remaining reserved for {tier}: ${self._reserved_exposures[tier]:,.0f}"
            )
    
    async def reserve_exposure_async(
        self,
        tier: str,
        proposed_value: float,
        current_tier_exposure: TierExposure,
        account_value: float,
        symbol: str
    ) -> tuple[bool, str]:
        """
        Async version of reserve_exposure for async orchestrator.
        
        Args:
            tier: Tier name
            proposed_value: Dollar value of proposed position
            current_tier_exposure: Current TierExposure for this tier
            account_value: Current account equity
            symbol: Symbol for tracking
        
        Returns:
            (approved: bool, reason: str)
        """
        if self._async_lock:
            async with self._async_lock:
                return self._reserve_exposure_internal(tier, proposed_value, current_tier_exposure, account_value, symbol)
        else:
            # Fallback to sync lock if async lock not available
            return self.reserve_exposure(tier, proposed_value, current_tier_exposure, account_value, symbol)
    
    def _reserve_exposure_internal(
        self,
        tier: str,
        proposed_value: float,
        current_tier_exposure: TierExposure,
        account_value: float,
        symbol: str
    ) -> tuple[bool, str]:
        """Internal method called within lock."""
        reserved_for_tier = self._reserved_exposures.get(tier, 0.0)
        
        cap = self.TIER_CAPS.get(tier)
        if not cap:
            return False, f"Unknown tier: {tier}"
        
        current_pct = current_tier_exposure.current_pct
        proposed_pct = (proposed_value / account_value) if account_value > 0 else 0
        reserved_pct = (reserved_for_tier / account_value) if account_value > 0 else 0
        new_total_pct = current_pct + reserved_pct + proposed_pct
        
        max_allowed = cap + self.SAFETY_BUFFER
        
        if new_total_pct > max_allowed:
            reason = (
                f"Tier {tier} capacity exceeded (atomic check): "
                f"Current={current_pct:.1%}, "
                f"Reserved={reserved_pct:.1%}, "
                f"Proposed={proposed_pct:.1%}, "
                f"Total={new_total_pct:.1%}, "
                f"Cap={cap:.1%}"
            )
            logger.warning(f"🚫 {symbol}: {reason}")
            return False, reason
        
        # Reserve atomically
        self._reserved_exposures[tier] = reserved_for_tier + proposed_value
        
        # Track reserved symbols for debugging (optional hardening)
        if tier not in self._reserved_symbols:
            self._reserved_symbols[tier] = set()
        self._reserved_symbols[tier].add(symbol)
        
        logger.info(
            f"✅ {symbol} ({tier}): Exposure reserved atomically. "
            f"Reserved: ${self._reserved_exposures[tier]:,.0f}"
        )
        return True, "Reserved"
    
    async def release_exposure_async(self, tier: str, value: float, symbol: str) -> None:
        """Async version of release_exposure."""
        if self._async_lock:
            async with self._async_lock:
                self._release_exposure_internal(tier, value, symbol)
        else:
            self.release_exposure(tier, value, symbol)
    
    def _release_exposure_internal(self, tier: str, value: float, symbol: str) -> None:
        """Internal method called within lock."""
        # Safeguard against double-release (optional hardening)
        if tier not in self._reserved_symbols or symbol not in self._reserved_symbols[tier]:
            logger.error(
                f"🚨 Double-release detected: {symbol} ({tier}) was not reserved",
                extra={
                    "symbol": symbol,
                    "tier": tier,
                    "value": value,
                    "reserved_symbols": list(self._reserved_symbols.get(tier, set())),
                    "current_reserved": self._reserved_exposures.get(tier, 0.0)
                }
            )
        
        current_reserved = self._reserved_exposures.get(tier, 0.0)
        new_reserved = max(0.0, current_reserved - value)
        self._reserved_exposures[tier] = new_reserved
        
        # Remove from reserved symbols tracking
        if tier in self._reserved_symbols:
            self._reserved_symbols[tier].discard(symbol)
        
        logger.debug(f"🔓 {symbol} ({tier}): Released ${value:,.0f}")
    
    def get_tier_for_symbol(self, symbol: str) -> Optional[str]:
        """Get tier assignment for a symbol"""
        return self.symbol_tier_mapping.get(symbol)
    
    def log_tier_status(self, tier_exposures: Dict[str, TierExposure]):
        """Log current tier exposure status"""
        logger.info("=" * 80)
        logger.info("📊 TIER EXPOSURE STATUS")
        logger.info("=" * 80)
        
        for tier in ['TIER1', 'TIER2', 'TIER3']:
            exposure = tier_exposures.get(tier)
            if not exposure:
                continue
            
            status = "⚠️  OVER CAP" if exposure.current_pct > exposure.cap_pct else "✅ OK"
            
            logger.info(
                f"{tier}: {exposure.current_pct:>6.1%} / {exposure.cap_pct:.1%} "
                f"({len(exposure.positions)} positions) {status}"
            )
            
            if exposure.positions:
                logger.info(f"  Positions: {', '.join(exposure.positions)}")
        
        logger.info("=" * 80)
