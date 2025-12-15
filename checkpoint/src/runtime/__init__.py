from typing import Optional
from src.runtime.snapshot_runtime import SnapshotRuntime

_runtime: Optional[SnapshotRuntime] = None

def get_runtime() -> SnapshotRuntime:
    global _runtime
    if _runtime is None:
        _runtime = SnapshotRuntime()
        _runtime.initialize()
    return _runtime

def shutdown_runtime() -> None:
    global _runtime
    if _runtime is not None:
        _runtime.shutdown()
        _runtime = None

__all__ = ['get_runtime', 'shutdown_runtime', 'SnapshotRuntime']
