"""
Orchestrator integration helpers for tiered allocation and safety checks.

Provides helper functions and initialization code that can be used
by both sync and async orchestrators.
"""
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import logging

from core.tier_exposure_tracker import TierExposureTracker
from core.signal_validator import SignalValidator
from core.tiered_position_sizer import TieredPositionSizer
from config.settings import AppConfig

logger = logging.getLogger(__name__)


def load_symbol_tier_mapping(config: AppConfig) -> Dict[str, str]:
    """
    Load symbol-to-tier mapping from config or defaults.
    
    Args:
        config: Application configuration
    
    Returns:
        Dict mapping symbols to tiers {'SPY': 'TIER1', ...}
    """
    # If config has explicit symbol_tiers, use it
    if hasattr(config, 'symbol_tiers') and config.symbol_tiers:
        return config.symbol_tiers
    
    # Otherwise, build from default tiers
    mapping = {}
    
    # Tier 1: Index ETFs
    tier1_symbols = ['SPY', 'QQQ', 'DIA', 'IWM']
    for symbol in tier1_symbols:
        mapping[symbol] = 'TIER1'
    
    # Tier 2: Sector ETFs
    tier2_symbols = ['XLK', 'XLF', 'XLV', 'XLE', 'XLY']
    for symbol in tier2_symbols:
        mapping[symbol] = 'TIER2'
    
    # Tier 3: Individual Stocks (top 20)
    tier3_symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA',
        'UNH', 'JPM', 'V', 'MA', 'LLY', 'WMT', 'HD', 'PG',
        'XOM', 'CVX', 'JNJ', 'BAC', 'COST'
    ]
    for symbol in tier3_symbols:
        mapping[symbol] = 'TIER3'
    
    # Also map any symbols from config.symbols that aren't in tiers yet
    if config.symbols:
        for symbol in config.symbols:
            if symbol not in mapping:
                # Default to TIER3 for unknown symbols
                mapping[symbol] = 'TIER3'
                logger.debug(f"Defaulting {symbol} to TIER3 (not in predefined tiers)")
    
    return mapping


def load_focus_symbols(config: AppConfig, scanner_file: Optional[Path] = None) -> List[str]:
    """
    Load symbols to monitor (scanner-driven or all configured symbols).
    
    Includes forced diversity to prevent scanner bias (Problem 5 Fix).
    
    Args:
        config: Application configuration
        scanner_file: Optional path to scanner output file
    
    Returns:
        List of symbols to monitor
    """
    baseline_symbols = config.baseline_symbols or ['SPY', 'QQQ']
    
    # Check if scanner is enabled and file exists
    if config.use_scanner:
        file_path = scanner_file or Path(config.scanner_file)
        
        if file_path.exists():
            try:
                with open(file_path) as f:
                    scanner_data = json.load(f)
                
                # Get scanner-selected symbols
                scanner_symbols = scanner_data.get('symbols', [])[:10]  # Top 10
                
                # PROBLEM 5 FIX: Force diversity
                focus_symbols = set(scanner_symbols)
                focus_symbols |= set(baseline_symbols)  # Always include baseline
                
                # Add random control symbol to prevent scanner bias
                all_symbols = config.symbols or []
                if all_symbols:
                    import random
                    # Select a random symbol from all available that's not in scanner picks
                    available_control = [s for s in all_symbols if s not in scanner_symbols]
                    if available_control:
                        control_symbol = random.choice(available_control)
                        focus_symbols.add(control_symbol)
                        logger.info(f"  Random control: {control_symbol} (prevents scanner bias)")
                
                focus_list = list(focus_symbols)
                
                logger.info(f"📊 Using scanner focus: {len(focus_list)} symbols")
                logger.info(f"  Scanner picks: {scanner_symbols}")
                logger.info(f"  Baseline: {baseline_symbols}")
                logger.info(f"  Forced diversity: {len(focus_list)} total (prevents concentration bias)")
                
                return focus_list
                
            except Exception as e:
                logger.warning(f"⚠️ Scanner data load failed: {e}, using all symbols")
    
    # Fallback: Use all configured symbols
    all_symbols = config.symbols or []
    if not all_symbols:
        # If no symbols configured, use defaults
        all_symbols = ['SPY', 'QQQ', 'AAPL', 'GOOGL', 'NVDA']
    
    logger.info(f"📊 Using all configured symbols: {len(all_symbols)} total")
    return all_symbols


def initialize_tier_tracker(config: AppConfig) -> Optional[TierExposureTracker]:
    """
    Initialize tier exposure tracker if tiered allocation is enabled.
    
    Args:
        config: Application configuration
    
    Returns:
        TierExposureTracker instance or None
    """
    if not config.enable_tiered_allocation:
        logger.info("Tiered allocation disabled, skipping tier tracker")
        return None
    
    try:
        symbol_mapping = load_symbol_tier_mapping(config)
        tracker = TierExposureTracker(symbol_mapping)
        logger.info("✅ Tier exposure tracker initialized")
        return tracker
    except Exception as e:
        logger.error(f"Failed to initialize tier tracker: {e}")
        return None


def initialize_signal_validator(config: AppConfig) -> Optional[SignalValidator]:
    """
    Initialize signal validator if scanner is enabled.
    
    Args:
        config: Application configuration
    
    Returns:
        SignalValidator instance or None
    """
    if not config.use_scanner:
        logger.info("Scanner disabled, skipping signal validator")
        return None
    
    try:
        validator = SignalValidator(
            max_gap_pct=config.max_gap_pct,
            max_signal_age_hours=config.max_signal_age_hours,
            enable_regime_check=True
        )
        logger.info("✅ Signal validator initialized")
        return validator
    except Exception as e:
        logger.error(f"Failed to initialize signal validator: {e}")
        return None


def initialize_tiered_sizer(config: AppConfig, account_value: Optional[float] = None) -> Optional[TieredPositionSizer]:
        """
        Initialize tiered position sizer if tiered allocation is enabled.
        
        Note: account_value is optional and stored for backward compatibility only.
        ALWAYS pass account_value to calculate_shares() for live equity (Problem 3 Fix).
        
        Args:
            config: Application configuration
            account_value: Optional initial account equity (deprecated - pass per call)
        
        Returns:
            TieredPositionSizer instance or None
        """
        if not config.enable_tiered_allocation:
            logger.info("Tiered allocation disabled, skipping tiered sizer")
            return None
        
        try:
            # account_value parameter is optional now - always pass to calculate_shares()
            sizer = TieredPositionSizer(
                account_value=account_value,  # Optional, for backward compatibility
                use_fractional=config.enable_fractional_shares,
                min_notional=config.min_order_notional,
                enable_compression=True
            )
            logger.info("✅ Tiered position sizer initialized (will use live account_value per call)")
            return sizer
        except Exception as e:
            logger.error(f"Failed to initialize tiered sizer: {e}")
            return None
