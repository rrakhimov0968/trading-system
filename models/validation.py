"""Validation utilities for trading models."""
import re
from typing import Optional
from utils.exceptions import ValidationError
from models.enums import OrderSide


def validate_symbol(symbol: str) -> str:
    """
    Validate and normalize a stock symbol.
    
    Args:
        symbol: Stock symbol to validate
        
    Returns:
        Normalized symbol (uppercase, stripped)
        
    Raises:
        ValidationError: If symbol is invalid
    """
    if not symbol:
        raise ValidationError("Symbol cannot be empty")
    
    symbol = symbol.strip().upper()
    
    # Basic validation: alphanumeric and common special chars, 1-5 chars
    if not re.match(r'^[A-Z]{1,5}$', symbol):
        raise ValidationError(
            f"Invalid symbol format: {symbol}. "
            "Symbols must be 1-5 uppercase letters"
        )
    
    return symbol


def validate_quantity(qty: int) -> int:
    """
    Validate order quantity.
    
    Args:
        qty: Quantity to validate
        
    Returns:
        Validated quantity
        
    Raises:
        ValidationError: If quantity is invalid
    """
    if qty <= 0:
        raise ValidationError(f"Quantity must be positive, got {qty}")
    
    if qty > 1000000:  # Reasonable upper bound
        raise ValidationError(f"Quantity too large: {qty}. Maximum is 1,000,000")
    
    return qty


def validate_order_side(side: str) -> OrderSide:
    """
    Validate and convert order side.
    
    Args:
        side: Order side string
        
    Returns:
        OrderSide enum
        
    Raises:
        ValidationError: If side is invalid
    """
    try:
        return OrderSide.from_string(side)
    except ValueError as e:
        raise ValidationError(str(e))


def validate_price(price: float, allow_zero: bool = False) -> float:
    """
    Validate price value.
    
    Args:
        price: Price to validate
        allow_zero: Whether to allow zero prices
        
    Returns:
        Validated price
        
    Raises:
        ValidationError: If price is invalid
    """
    if not allow_zero and price <= 0:
        raise ValidationError(f"Price must be positive, got {price}")
    
    if allow_zero and price < 0:
        raise ValidationError(f"Price cannot be negative, got {price}")
    
    if price > 1000000:  # Reasonable upper bound
        raise ValidationError(f"Price too large: {price}. Maximum is $1,000,000")
    
    return price

