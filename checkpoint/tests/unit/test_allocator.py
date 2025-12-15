
import pytest
from src.runtime.allocator import AllocationTracker
from unittest.mock import Mock

class TestAllocationTracker:
    
    def test_initialization(self):
        mock_lib = Mock()
        tracker = AllocationTracker(mock_lib)
        assert tracker._lib == mock_lib
    
    def test_discover_allocations(self):
        mock_lib = Mock()
        tracker = AllocationTracker(mock_lib)
        tracker.discover_allocations()
        mock_lib.interceptor_discover_allocations.assert_called_once()
    
    def test_register_allocation_valid(self):
        mock_lib = Mock()
        tracker = AllocationTracker(mock_lib)
        tracker.register_allocation(0x1000, 1024 * 1024)
        mock_lib.interceptor_register_allocation.assert_called_once()
    
    def test_register_allocation_invalid_ptr(self):
        mock_lib = Mock()
        tracker = AllocationTracker(mock_lib)
        tracker.register_allocation(0, 1024)
        mock_lib.interceptor_register_allocation.assert_not_called()
    
    def test_register_allocation_invalid_size(self):
        mock_lib = Mock()
        tracker = AllocationTracker(mock_lib)
        tracker.register_allocation(0x1000, 0)
        mock_lib.interceptor_register_allocation.assert_not_called()
