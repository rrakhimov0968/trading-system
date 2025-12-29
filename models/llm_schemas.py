"""Pydantic schemas for LLM response validation."""
from pydantic import BaseModel, Field
from typing import Literal


class LLMStrategySelection(BaseModel):
    """
    Schema for validated LLM strategy selection response.
    
    This ensures the LLM can only return predefined strategies and valid actions.
    """
    strategy_name: Literal[
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
    ] = Field(
        description="Must match exactly one predefined strategy from the available list"
    )
    action: Literal["BUY", "SELL", "HOLD"] = Field(
        description="Trading action to take"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score between 0.0 and 1.0"
    )
    reasoning: str = Field(
        description="Brief explanation of why this strategy fits the market context"
    )

