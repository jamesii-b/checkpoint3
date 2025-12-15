
import ctypes

class AllocationTracker:
    
    def __init__(self, lib: ctypes.CDLL):
        self._lib = lib
    
    def discover_allocations(self) -> None:
        self._lib.interceptor_discover_allocations()
    
    def register_allocation(self, ptr: int, size: int) -> None:
        if ptr == 0 or size == 0:
            return
            
        self._lib.interceptor_register_allocation(ctypes.c_void_p(ptr), ctypes.c_size_t(size))
    
    def classify_buffers(self) -> None:
        self._lib.interceptor_classify_buffers()
    
    def print_statistics(self) -> None:
        self._lib.interceptor_print_stats()