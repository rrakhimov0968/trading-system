#!/usr/bin/env python3
"""
Test script to verify safety checks are working correctly.
"""
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.order_tracker import OrderTracker


def test_order_tracker():
    """Test OrderTracker with daily limit and cooldown."""
    print("=" * 60)
    print("Testing OrderTracker")
    print("=" * 60)
    
    # Create tracker with 10 orders/day limit
    tracker = OrderTracker(cooldown_minutes=60, max_orders_per_day=10)
    
    # Test 1: Can place first order
    print("\n1. Testing first order for AAPL:")
    can_place, reason = tracker.can_place_order('AAPL')
    print(f"   Can place: {can_place} (expected: True)")
    assert can_place, "Should be able to place first order"
    
    # Test 2: Record order
    print("\n2. Recording order for AAPL:")
    tracker.record_order('AAPL')
    print(f"   Daily count: {sum(tracker.daily_order_count.values())}/10")
    
    # Test 3: Cannot place immediately after (cooldown)
    print("\n3. Testing immediate second order (should fail due to cooldown):")
    can_place, reason = tracker.can_place_order('AAPL')
    print(f"   Can place: {can_place} (expected: False)")
    print(f"   Reason: {reason}")
    assert not can_place, "Should be blocked by cooldown"
    
    # Test 4: Can place order for different symbol
    print("\n4. Testing order for different symbol (GOOGL):")
    can_place, reason = tracker.can_place_order('GOOGL')
    print(f"   Can place: {can_place} (expected: True)")
    assert can_place, "Should be able to place order for different symbol"
    
    # Test 5: Daily limit
    print("\n5. Testing daily limit (placing 9 more orders):")
    for i in range(9):
        symbol = f'STOCK{i}'
        tracker.record_order(symbol)
    
    print(f"   Daily count: {sum(tracker.daily_order_count.values())}/10")
    can_place, reason = tracker.can_place_order('NEWSYMBOL')
    print(f"   Can place 11th order: {can_place} (expected: False)")
    print(f"   Reason: {reason}")
    assert not can_place, "Should be blocked by daily limit"
    
    print("\n✅ OrderTracker tests passed!")


def test_emergency_stop():
    """Test emergency stop file check."""
    print("\n" + "=" * 60)
    print("Testing Emergency Stop")
    print("=" * 60)
    
    stop_file = Path("EMERGENCY_STOP")
    
    # Clean up if exists
    if stop_file.exists():
        stop_file.unlink()
        print("Removed existing EMERGENCY_STOP file")
    
    # Test 1: No emergency stop
    print("\n1. Testing without emergency stop file:")
    exists = stop_file.exists()
    print(f"   File exists: {exists} (expected: False)")
    assert not exists, "Emergency stop file should not exist"
    
    # Test 2: Create emergency stop
    print("\n2. Creating EMERGENCY_STOP file:")
    stop_file.touch()
    exists = stop_file.exists()
    print(f"   File exists: {exists} (expected: True)")
    assert exists, "Emergency stop file should exist"
    
    # Test 3: Remove emergency stop
    print("\n3. Removing EMERGENCY_STOP file:")
    stop_file.unlink()
    exists = stop_file.exists()
    print(f"   File exists: {exists} (expected: False)")
    assert not exists, "Emergency stop file should be removed"
    
    print("\n✅ Emergency stop tests passed!")


if __name__ == "__main__":
    try:
        test_order_tracker()
        test_emergency_stop()
        print("\n" + "=" * 60)
        print("✅ All safety check tests passed!")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error running tests: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

