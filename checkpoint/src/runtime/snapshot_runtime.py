
from pathlib import Path
from typing import Optional
import ctypes

from src.bindings.native_loader import NativeLibraryLoader
from src.runtime.allocator import AllocationTracker

class SnapshotRuntime:
    
    def __init__(self, lib_path: Optional[Path] = None):
        self._loader = NativeLibraryLoader(lib_path)
        self._lib: Optional[ctypes.CDLL] = None
        self._allocator: Optional[AllocationTracker] = None
        self._initialized = False
        self._cuda_ready = False
        
    def _ensure_cuda_loaded(self) -> None:
        if self._cuda_ready:
            return
            
        try:
            import torch
            if torch.cuda.is_available():
                _ = torch.cuda.current_device()
            self._cuda_ready = True
        except:
            self._cuda_ready = True
        
    def initialize(self) -> None:
        if self._initialized:
            return
        
        self._ensure_cuda_loaded()
        
        self._lib = self._loader.load()
        self._allocator = AllocationTracker(self._lib)
        self._lib.interceptor_init()
        self._initialized = True
    
    def shutdown(self) -> None:
        if self._lib and self._initialized:
            self._lib.interceptor_shutdown()
            self._initialized = False
    
    def tick_epoch(self) -> None:
        if self._lib:
            self._lib.interceptor_tick_epoch()
    
    def set_token_index(self, index: int) -> None:
        if self._lib:
            self._lib.interceptor_set_token_index(ctypes.c_uint64(index))
    
    def get_token_index(self) -> int:
        if self._lib:
            return self._lib.interceptor_get_token_index()
        return 0
    
    def set_rng_state(self, state: int) -> None:
        if self._lib:
            self._lib.interceptor_set_rng_state(ctypes.c_uint64(state))
    
    def get_rng_state(self) -> int:
        if self._lib:
            return self._lib.interceptor_get_rng_state()
        return 0
    
    def discover_allocations(self) -> None:
        if self._allocator:
            self._allocator.discover_allocations()
    
    def register_allocation(self, ptr: int, size: int) -> None:
        if self._allocator:
            self._allocator.register_allocation(ptr, size)
    
    def classify_buffers(self) -> None:
        if self._allocator:
            self._allocator.classify_buffers()
    
    def print_stats(self) -> None:
        if self._allocator:
            self._allocator.print_statistics()
    
    def snapshot(self, path: str) -> bool:
        if not self._lib:
            return False
        
        result = self._lib.interceptor_snapshot(path.encode('utf-8'))
        return result == 0
    
    def restore(self, path: str) -> bool:
        if not self._lib:
            return False
        
        result = self._lib.interceptor_restore(path.encode('utf-8'))
        return result == 0