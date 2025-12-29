"""Deterministic strategy classes for signal generation.

This module contains pure-code strategy implementations that generate
trading signals based on technical analysis. Strategies are selected
by the StrategyAgent LLM and executed deterministically.
"""
from core.strategies.base_strategy import BaseStrategy
from core.strategies.trend_following import TrendFollowing
from core.strategies.momentum_rotation import MomentumRotation
from core.strategies.mean_reversion import MeanReversion
from core.strategies.breakout import Breakout
from core.strategies.volatility_breakout import VolatilityBreakout
from core.strategies.relative_strength import RelativeStrength
from core.strategies.sector_rotation import SectorRotation
from core.strategies.dual_momentum import DualMomentum
from core.strategies.ma_envelope import MovingAverageEnvelope
from core.strategies.bollinger_bands import BollingerBandsReversion

# Strategy registry - maps strategy names to classes
STRATEGY_REGISTRY = {
    "TrendFollowing": TrendFollowing,
    "MomentumRotation": MomentumRotation,
    "MeanReversion": MeanReversion,
    "Breakout": Breakout,
    "VolatilityBreakout": VolatilityBreakout,
    "RelativeStrength": RelativeStrength,
    "SectorRotation": SectorRotation,
    "DualMomentum": DualMomentum,
    "MovingAverageEnvelope": MovingAverageEnvelope,
    "BollingerBands": BollingerBandsReversion,
    # Legacy name mappings
    "MovingAverageCrossover": TrendFollowing,
    "Momentum": MomentumRotation,
    "RSI_OversoldOverbought": MeanReversion,
    "VolumeProfile": Breakout,
    "SupportResistance": Breakout,
    "ConsolidationBreakout": Breakout,
}

__all__ = [
    "BaseStrategy",
    "TrendFollowing",
    "MomentumRotation",
    "MeanReversion",
    "Breakout",
    "VolatilityBreakout",
    "RelativeStrength",
    "SectorRotation",
    "DualMomentum",
    "MovingAverageEnvelope",
    "BollingerBandsReversion",
    "STRATEGY_REGISTRY",
]

