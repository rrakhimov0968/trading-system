"""Tests for validation utilities."""
import pytest
from models.validation import (
    validate_symbol,
    validate_quantity,
    validate_order_side,
    validate_price
)
from models.enums import OrderSide
from utils.exceptions import ValidationError


@pytest.mark.unit
class TestSymbolValidation:
    """Test symbol validation."""
    
    def test_valid_symbol(self):
        """Test validation of valid symbols."""
        assert validate_symbol("AAPL") == "AAPL"
        assert validate_symbol("GOOGL") == "GOOGL"
        assert validate_symbol("  aapl  ") == "AAPL"  # Normalization
    
    def test_empty_symbol(self):
        """Test that empty symbols are rejected."""
        # Empty string raises "Symbol cannot be empty"
        with pytest.raises(ValidationError) as exc_info:
            validate_symbol("")
        assert "empty" in str(exc_info.value).lower() or "cannot" in str(exc_info.value).lower()
        
        # Whitespace only gets stripped and becomes empty, then fails format validation
        with pytest.raises(ValidationError) as exc_info:
            validate_symbol("   ")
        # After strip, becomes empty string, which then fails format check
        assert "empty" in str(exc_info.value).lower() or "invalid" in str(exc_info.value).lower() or "format" in str(exc_info.value).lower()
    
    def test_invalid_symbol_format(self):
        """Test that invalid symbol formats are rejected."""
        with pytest.raises(ValidationError, match="Invalid symbol format"):
            validate_symbol("AAPL1")  # Contains number
        with pytest.raises(ValidationError, match="Invalid symbol format"):
            validate_symbol("TOOLONG")  # Too long
        with pytest.raises(ValidationError, match="Invalid symbol format"):
            validate_symbol("aa-pl")  # Contains hyphen


@pytest.mark.unit
class TestQuantityValidation:
    """Test quantity validation."""
    
    def test_valid_quantity(self):
        """Test validation of valid quantities."""
        assert validate_quantity(1) == 1
        assert validate_quantity(100) == 100
        assert validate_quantity(1000000) == 1000000
    
    def test_invalid_quantity(self):
        """Test that invalid quantities are rejected."""
        with pytest.raises(ValidationError, match="Quantity must be positive"):
            validate_quantity(0)
        with pytest.raises(ValidationError, match="Quantity must be positive"):
            validate_quantity(-1)
        with pytest.raises(ValidationError, match="Quantity must be positive"):
            validate_quantity(-100)
    
    def test_quantity_too_large(self):
        """Test that quantities that are too large are rejected."""
        with pytest.raises(ValidationError, match="Quantity too large"):
            validate_quantity(1000001)


@pytest.mark.unit
class TestOrderSideValidation:
    """Test order side validation."""
    
    def test_valid_order_side(self):
        """Test validation of valid order sides."""
        assert validate_order_side("buy") == OrderSide.BUY
        assert validate_order_side("BUY") == OrderSide.BUY
        assert validate_order_side("Buy") == OrderSide.BUY
        assert validate_order_side("sell") == OrderSide.SELL
        assert validate_order_side("SELL") == OrderSide.SELL
    
    def test_invalid_order_side(self):
        """Test that invalid order sides are rejected."""
        with pytest.raises(ValidationError):
            validate_order_side("invalid")
        with pytest.raises(ValidationError):
            validate_order_side("purchase")
        with pytest.raises(ValidationError):
            validate_order_side("")


@pytest.mark.unit
class TestPriceValidation:
    """Test price validation."""
    
    def test_valid_price(self):
        """Test validation of valid prices."""
        assert validate_price(1.0) == 1.0
        assert validate_price(100.50) == 100.50
        assert validate_price(0.01) == 0.01
    
    def test_invalid_price(self):
        """Test that invalid prices are rejected."""
        with pytest.raises(ValidationError, match="Price must be positive"):
            validate_price(0.0)
        with pytest.raises(ValidationError, match="Price must be positive"):
            validate_price(-1.0)
    
    def test_zero_price_allowed(self):
        """Test that zero price can be allowed."""
        assert validate_price(0.0, allow_zero=True) == 0.0
    
    def test_price_too_large(self):
        """Test that prices that are too large are rejected."""
        with pytest.raises(ValidationError, match="Price too large"):
            validate_price(1000001.0)

