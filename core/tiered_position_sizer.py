"""
Tiered position sizing with safety fixes.

Addresses the "Expensive Stock Trap" where expensive stocks in Tier 3
cannot be purchased with small account sizes without fractional shares.

Key Safety Fixes:
1. Never mutate tier allocation config at runtime
2. Always enforce tier caps after sizing
3. Enforce minimum notional for fractional shares
4. Support dynamic tier compression for expensive stocks
"""
from typing import Dict, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TierConfig:
    """Immutable tier configuration."""
    name: str  # 'TIER1', 'TIER2', 'TIER3'
    allocation: float  # Percentage of account (0.40 = 40%)
    symbol_count: int  # Number of symbols in this tier
    max_position_pct: float = 0.20  # Max 20% per position within tier (safety cap)
    
    def __post_init__(self):
        """Validate tier config."""
        if not 0 <= self.allocation <= 1.0:
            raise ValueError(f"Tier allocation must be between 0 and 1, got {self.allocation}")
        if self.symbol_count < 1:
            raise ValueError(f"Tier symbol count must be >= 1, got {self.symbol_count}")


class TieredPositionSizer:
    """
    Position sizing with tiered allocation and safety fixes.
    
    Tiers:
    - TIER1: Index ETFs (40% of capital, 4 symbols)
    - TIER2: Sector ETFs (30% of capital, 5 symbols)
    - TIER3: Individual stocks (30% of capital, 20 symbols)
    
    Safety Features:
    - Fractional shares support (required for < $25k accounts)
    - Minimum notional enforcement ($10-25 minimum order)
    - Dynamic tier compression (reduce active symbols if prices too high)
    - Immutable config (never mutate tier allocation at runtime)
    - Tier cap enforcement (max 20% per position within tier)
    """
    
    # Default tier configurations (immutable)
    DEFAULT_TIER_CONFIGS = {
        'TIER1': TierConfig(
            name='TIER1',
            allocation=0.40,  # 40%
            symbol_count=4,   # SPY, QQQ, DIA, IWM
            max_position_pct=0.10  # Max 10% per symbol in Tier 1
        ),
        'TIER2': TierConfig(
            name='TIER2',
            allocation=0.30,  # 30%
            symbol_count=5,   # XLK, XLF, XLV, XLE, XLY
            max_position_pct=0.06  # Max 6% per symbol in Tier 2
        ),
        'TIER3': TierConfig(
            name='TIER3',
            allocation=0.30,  # 30%
            symbol_count=20,  # Individual stocks
            max_position_pct=0.015  # Max 1.5% per symbol in Tier 3
        )
    }
    
    def __init__(
        self,
        account_value: Optional[float] = None,  # Optional - can be passed per call
        tier_configs: Optional[Dict[str, TierConfig]] = None,
        use_fractional: bool = True,
        min_notional: float = 10.0,
        enable_compression: bool = True
    ):
        """
        Initialize tiered position sizer.
        
        Args:
            account_value: Optional initial account value (deprecated - pass per call instead)
                          Kept for backward compatibility but not used internally
            tier_configs: Optional custom tier configs (defaults to DEFAULT_TIER_CONFIGS)
            use_fractional: Enable fractional shares (required for < $25k accounts)
            min_notional: Minimum order notional in dollars ($10-25 recommended)
            enable_compression: Enable dynamic tier compression for expensive stocks
        """
        # NOTE: account_value is stored for backward compatibility but NOT used
        # Always pass account_value to calculate_shares() for live equity
        self._initial_account_value = account_value
        self.use_fractional = use_fractional
        self.min_notional = min_notional
        self.enable_compression = enable_compression
        
        # Use immutable tier configs (never mutate)
        self.tier_configs = tier_configs or self.DEFAULT_TIER_CONFIGS.copy()
        
        # Validate tier allocations sum to ~1.0 (allow small rounding errors)
        total_allocation = sum(cfg.allocation for cfg in self.tier_configs.values())
        if abs(total_allocation - 1.0) > 0.01:  # Allow 1% tolerance
            logger.warning(
                f"Tier allocations sum to {total_allocation:.2%}, expected ~100%. "
                f"This may cause over or under-allocation."
            )
        
        logger.info(
            f"TieredPositionSizer initialized: "
            f"account=${account_value:,.0f}, "
            f"fractional={'enabled' if use_fractional else 'disabled'}, "
            f"min_notional=${min_notional:.2f}"
        )
    
    def calculate_shares(
        self,
        symbol: str,
        current_price: float,
        tier: str,
        account_value: float  # REQUIRED: Always pass live equity (Problem 3 Fix)
    ) -> Tuple[Optional[float], Dict[str, any]]:
        """
        Calculate position size for a symbol based on tier.
        
        CRITICAL SAFETY FIXES:
        1. Never mutate tier config allocation
        2. Compute effective allocation (may be reduced if compression enabled)
        3. Enforce tier cap after sizing
        4. Enforce minimum notional for fractional shares
        5. Handle zero-share trap (expensive stocks)
        6. ALWAYS use live account_value (never cached) - Problem 3 Fix
        
        Args:
            symbol: Stock symbol
            current_price: Current price per share
            tier: Tier name ('TIER1', 'TIER2', or 'TIER3')
            account_value: REQUIRED - Current live account equity (NOT cached)
        
        Returns:
            Tuple of (shares: Optional[float], metadata: Dict)
            - shares: Number of shares (float if fractional, int if whole)
            - metadata: Dict with sizing details and warnings
        """
        if account_value <= 0:
            return None, {
                'skipped': True,
                'reason': f'Invalid account value: {account_value}',
                'error': 'Account value must be > 0'
            }
        if tier not in self.tier_configs:
            logger.error(f"Unknown tier: {tier}. Valid tiers: {list(self.tier_configs.keys())}")
            return None, {'error': f'Unknown tier: {tier}', 'skipped': True}
        
        config = self.tier_configs[tier]
        
        # SAFETY FIX 1: Never mutate config.allocation - compute effective allocation
        # SAFETY FIX 6: Always use live account_value, never cached
        tier_capital = account_value * config.allocation
        
        # SAFETY FIX 2: Dynamic tier compression for Tier 3 expensive stocks
        active_symbols = config.symbol_count
        if tier == 'TIER3' and self.enable_compression:
            # Check if stock is expensive relative to base allocation
            base_position_value = tier_capital / config.symbol_count
            if current_price > (base_position_value * 2):
                # Stock is too expensive, reduce active symbol count
                active_symbols = max(5, config.symbol_count // 2)
                logger.info(
                    f"Tier 3 compression: {symbol} @ ${current_price:.2f} is expensive. "
                    f"Reducing active symbols from {config.symbol_count} to {active_symbols}"
                )
        
        # Calculate base position value (equal weight within tier)
        base_position_value = tier_capital / active_symbols
        
        # SAFETY FIX 3: Enforce tier cap (max position % within tier)
        max_position_value_in_tier = account_value * config.max_position_pct
        position_value = min(base_position_value, max_position_value_in_tier)
        
        # Calculate shares
        if self.use_fractional:
            shares = position_value / current_price  # Float for fractional shares
        else:
            shares = int(position_value / current_price)  # Integer (rounds down)
        
        # SAFETY FIX 4: Check for zero-share trap (expensive stock trap)
        if shares == 0:
            if self.use_fractional:
                # With fractional shares enabled, this shouldn't happen unless price > position_value
                # Still check for minimum notional
                if position_value < self.min_notional:
                    logger.warning(
                        f"SKIPPING {symbol}: Position value ${position_value:.2f} < "
                        f"minimum notional ${self.min_notional:.2f}"
                    )
                    return None, {
                        'skipped': True,
                        'reason': f'Position value ${position_value:.2f} < min notional ${self.min_notional:.2f}',
                        'position_value': position_value,
                        'shares': 0
                    }
                # Position value is valid but shares would be < 1, which is fine for fractional
                # But check if it meets minimum notional
                if position_value >= self.min_notional:
                    shares = position_value / current_price  # Allow fractional
                else:
                    return None, {
                        'skipped': True,
                        'reason': f'Position value ${position_value:.2f} < min notional ${self.min_notional:.2f}',
                        'position_value': position_value,
                        'shares': 0
                    }
            else:
                # Without fractional shares, zero shares means we can't buy
                logger.warning(
                    f"SKIPPING {symbol}: Price ${current_price:.2f} > "
                    f"position value ${position_value:.2f} (no fractional shares)"
                )
                return None, {
                    'skipped': True,
                    'reason': f'Price ${current_price:.2f} > position value ${position_value:.2f} (fractional disabled)',
                    'position_value': position_value,
                    'shares': 0,
                    'suggestion': 'Enable fractional shares or increase account size'
                }
        
        # SAFETY FIX 5: Enforce minimum notional (even with fractional shares)
        position_notional = shares * current_price
        if position_notional < self.min_notional:
            logger.warning(
                f"SKIPPING {symbol}: Order notional ${position_notional:.2f} < "
                f"minimum ${self.min_notional:.2f}"
            )
            return None, {
                'skipped': True,
                'reason': f'Order notional ${position_notional:.2f} < min ${self.min_notional:.2f}',
                'position_value': position_value,
                'shares': shares,
                'notional': position_notional
            }
        
        # Build metadata
        metadata = {
            'tier': tier,
            'tier_allocation': config.allocation,
            'tier_capital': tier_capital,
            'base_position_value': base_position_value,
            'position_value': position_value,
            'position_notional': position_notional,
            'shares': shares,
            'current_price': current_price,
            'compression_applied': active_symbols < config.symbol_count if tier == 'TIER3' else False,
            'active_symbols': active_symbols,
            'fractional': self.use_fractional and shares < 1.0
        }
        
        logger.info(
            f"Position sizing for {symbol} ({tier}): "
            f"${position_value:,.2f} @ ${current_price:.2f} = {shares:.4f} shares "
            f"(notional: ${position_notional:,.2f})"
        )
        
        return shares, metadata
    
    def get_tier_for_symbol(self, symbol: str, symbol_tiers: Dict[str, str]) -> Optional[str]:
        """
        Get tier for a symbol.
        
        Args:
            symbol: Stock symbol
            symbol_tiers: Dict mapping symbol to tier
        
        Returns:
            Tier name or None
        """
        return symbol_tiers.get(symbol.upper())


# Example usage and validation
if __name__ == "__main__":
    # Test with small account (triggers expensive stock trap)
    print("=" * 80)
    print("TEST 1: Small Account ($10k) - Expensive Stock Trap")
    print("=" * 80)
    
    sizer = TieredPositionSizer(
        account_value=10000.0,
        use_fractional=True,  # Required for small accounts
        min_notional=10.0
    )
    
    # Test Tier 3 expensive stock (COST ~$900)
    shares, meta = sizer.calculate_shares('COST', 900.0, 'TIER3')
    if shares:
        print(f"✅ COST: {shares:.4f} shares, ${meta['position_notional']:.2f} notional")
    else:
        print(f"❌ COST: {meta.get('reason', 'Failed')}")
    
    # Test Tier 3 moderate stock (AAPL ~$150)
    shares, meta = sizer.calculate_shares('AAPL', 150.0, 'TIER3')
    if shares:
        print(f"✅ AAPL: {shares:.4f} shares, ${meta['position_notional']:.2f} notional")
    else:
        print(f"❌ AAPL: {meta.get('reason', 'Failed')}")
    
    # Test Tier 1 (SPY ~$450)
    shares, meta = sizer.calculate_shares('SPY', 450.0, 'TIER1')
    if shares:
        print(f"✅ SPY: {shares:.4f} shares, ${meta['position_notional']:.2f} notional")
    else:
        print(f"❌ SPY: {meta.get('reason', 'Failed')}")
    
    print("\n" + "=" * 80)
    print("TEST 2: Large Account ($100k) - No Fractional Needed")
    print("=" * 80)
    
    sizer2 = TieredPositionSizer(
        account_value=100000.0,
        use_fractional=False,  # Not needed for large accounts
        min_notional=25.0
    )
    
    shares, meta = sizer2.calculate_shares('COST', 900.0, 'TIER3')
    if shares:
        print(f"✅ COST: {int(shares)} shares, ${meta['position_notional']:.2f} notional")
    else:
        print(f"❌ COST: {meta.get('reason', 'Failed')}")
