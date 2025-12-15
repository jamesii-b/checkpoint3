import ctypes
from pathlib import Path
from typing import Optional

class NativeLibraryLoader:
    
    def __init__(self, lib_path: Optional[Path] = None):
        self._lib: Optional[ctypes.CDLL] = None
        self._lib_path = lib_path
        
    def _find_library(self) -> Path:
        if self._lib_path and self._lib_path.exists():
            return self._lib_path
            
        candidates = [
            Path(__file__).parent.parent.parent / "native" / "libcuda_interceptor.so",
            Path(__file__).parent.parent / "native" / "libcuda_interceptor.so",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        raise RuntimeError(f"Native library not found")
    
    def load(self) -> ctypes.CDLL:
        if self._lib is not None:
            return self._lib
            
        lib_path = self._find_library()
        
        try:
            self._lib = ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
        except OSError as e:
            raise RuntimeError(f"Failed to load native library: {e}")
        
        self._configure_signatures()
        return self._lib
    
    def _configure_signatures(self) -> None:
        if self._lib is None:
            return
        
        self._lib.interceptor_init.argtypes = []
        self._lib.interceptor_init.restype = None
        
        self._lib.interceptor_shutdown.argtypes = []
        self._lib.interceptor_shutdown.restype = None
        
        self._lib.interceptor_tick_epoch.argtypes = []
        self._lib.interceptor_tick_epoch.restype = None
        
        self._lib.interceptor_set_token_index.argtypes = [ctypes.c_uint64]
        self._lib.interceptor_set_token_index.restype = None
        
        self._lib.interceptor_get_token_index.argtypes = []
        self._lib.interceptor_get_token_index.restype = ctypes.c_uint64
        
        self._lib.interceptor_set_rng_state.argtypes = [ctypes.c_uint64]
        self._lib.interceptor_set_rng_state.restype = None
        
        self._lib.interceptor_get_rng_state.argtypes = []
        self._lib.interceptor_get_rng_state.restype = ctypes.c_uint64
        
        self._lib.interceptor_discover_allocations.argtypes = []
        self._lib.interceptor_discover_allocations.restype = None
        
        self._lib.interceptor_register_allocation.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self._lib.interceptor_register_allocation.restype = None
        
        self._lib.interceptor_snapshot.argtypes = [ctypes.c_char_p]
        self._lib.interceptor_snapshot.restype = ctypes.c_int
        
        self._lib.interceptor_restore.argtypes = [ctypes.c_char_p]
        self._lib.interceptor_restore.restype = ctypes.c_int
        
        self._lib.interceptor_classify_buffers.argtypes = []
        self._lib.interceptor_classify_buffers.restype = None
        
        self._lib.interceptor_print_stats.argtypes = []
        self._lib.interceptor_print_stats.restype = None
    
    @property
    def lib(self) -> ctypes.CDLL:
        if self._lib is None:
            raise RuntimeError("Library not loaded")
        return self._lib
