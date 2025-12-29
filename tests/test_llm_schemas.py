"""Tests for LLM schema validation."""
import pytest
from pydantic import ValidationError

from models.llm_schemas import LLMStrategySelection


@pytest.mark.unit
class TestLLMStrategySelection:
    """Test LLMStrategySelection schema validation."""
    
    def test_valid_strategy_selection(self):
        """Test valid strategy selection."""
        selection = LLMStrategySelection(
            strategy_name="MovingAverageCrossover",
            action="BUY",
            confidence=0.75,
            reasoning="Strong trend detected"
        )
        
        assert selection.strategy_name == "MovingAverageCrossover"
        assert selection.action == "BUY"
        assert selection.confidence == 0.75
    
    def test_invalid_strategy_name(self):
        """Test that invalid strategy names are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            LLMStrategySelection(
                strategy_name="MyCustomStrategy",  # Not in allowed list
                action="BUY",
                confidence=0.75,
                reasoning="Test"
            )
        
        # Should have validation error for strategy_name
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("strategy_name",) for e in errors)
    
    def test_invalid_action(self):
        """Test that invalid actions are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            LLMStrategySelection(
                strategy_name="MovingAverageCrossover",
                action="INVALID",  # Not in allowed list
                confidence=0.75,
                reasoning="Test"
            )
        
        errors = exc_info.value.errors()
        assert any(e["loc"] == ("action",) for e in errors)
    
    def test_confidence_out_of_range(self):
        """Test that confidence is clamped to 0.0-1.0."""
        # Too high
        with pytest.raises(ValidationError):
            LLMStrategySelection(
                strategy_name="MovingAverageCrossover",
                action="BUY",
                confidence=1.5,  # > 1.0
                reasoning="Test"
            )
        
        # Too low
        with pytest.raises(ValidationError):
            LLMStrategySelection(
                strategy_name="MovingAverageCrossover",
                action="BUY",
                confidence=-0.1,  # < 0.0
                reasoning="Test"
            )
    
    def test_all_valid_strategies(self):
        """Test that all predefined strategies are accepted."""
        valid_strategies = [
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
        
        for strategy in valid_strategies:
            selection = LLMStrategySelection(
                strategy_name=strategy,
                action="BUY",
                confidence=0.75,
                reasoning="Test"
            )
            assert selection.strategy_name == strategy
    
    def test_all_valid_actions(self):
        """Test that all valid actions are accepted."""
        for action in ["BUY", "SELL", "HOLD"]:
            selection = LLMStrategySelection(
                strategy_name="MovingAverageCrossover",
                action=action,
                confidence=0.75,
                reasoning="Test"
            )
            assert selection.action == action

