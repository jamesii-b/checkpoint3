import os
import io
import json
import struct
import threading
import mmap
from pathlib import Path
from typing import Optional, Dict, Any, BinaryIO
from abc import ABC, abstractmethod

from src.distributed import DistributedConfig, SharedFSType


class SharedFS(ABC):
    @abstractmethod
    def open(self, path: str, mode: str) -> BinaryIO:
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        pass

    @abstractmethod
    def mkdir(self, path: str) -> None:
        pass

    @abstractmethod
    def remove(self, path: str) -> None:
        pass

    @abstractmethod
    def listdir(self, path: str) -> list:
        pass

    @abstractmethod
    def lock(self, path: str) -> Any:
        pass

    @abstractmethod
    def unlock(self, lock: Any) -> None:
        pass


class LocalSharedFS(SharedFS):
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._locks: Dict[str, threading.Lock] = {}
        self._lock_mutex = threading.Lock()

    def _resolve(self, path: str) -> Path:
        return self.base_path / path.lstrip("/")

    def open(self, path: str, mode: str) -> BinaryIO:
        full_path = self._resolve(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        return open(full_path, mode)

    def exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    def mkdir(self, path: str) -> None:
        self._resolve(path).mkdir(parents=True, exist_ok=True)

    def remove(self, path: str) -> None:
        full_path = self._resolve(path)
        if full_path.is_file():
            full_path.unlink()
        elif full_path.is_dir():
            import shutil
            shutil.rmtree(full_path)

    def listdir(self, path: str) -> list:
        full_path = self._resolve(path)
        if not full_path.exists():
            return []
        return [p.name for p in full_path.iterdir()]

    def lock(self, path: str) -> threading.Lock:
        with self._lock_mutex:
            if path not in self._locks:
                self._locks[path] = threading.Lock()
            lock = self._locks[path]
        lock.acquire()
        return lock

    def unlock(self, lock: Any) -> None:
        if lock is not None and hasattr(lock, 'release'):
            lock.release()


class MemorySharedFS(SharedFS):
    _storage: Dict[str, bytes] = {}
    _lock = threading.Lock()
    _file_locks: Dict[str, threading.Lock] = {}

    def __init__(self):
        pass

    def open(self, path: str, mode: str) -> BinaryIO:
        if "r" in mode:
            with self._lock:
                data = self._storage.get(path, b"")
            return io.BytesIO(data)
        else:
            return _MemoryFile(self, path)

    def exists(self, path: str) -> bool:
        with self._lock:
            if path in self._storage:
                return True
            for key in self._storage:
                if key.startswith(path + "/"):
                    return True
        return False

    def mkdir(self, path: str) -> None:
        pass

    def remove(self, path: str) -> None:
        with self._lock:
            keys_to_remove = [k for k in self._storage if k == path or k.startswith(path + "/")]
            for k in keys_to_remove:
                del self._storage[k]

    def listdir(self, path: str) -> list:
        with self._lock:
            prefix = path.rstrip("/") + "/"
            items = set()
            for key in self._storage:
                if key.startswith(prefix):
                    rest = key[len(prefix):]
                    items.add(rest.split("/")[0])
            return list(items)

    def lock(self, path: str) -> threading.Lock:
        with self._lock:
            if path not in self._file_locks:
                self._file_locks[path] = threading.Lock()
            lock = self._file_locks[path]
        lock.acquire()
        return lock

    def unlock(self, lock: Any) -> None:
        if lock is not None and hasattr(lock, 'release'):
            lock.release()

    def _write(self, path: str, data: bytes) -> None:
        with self._lock:
            self._storage[path] = data


class _MemoryFile(io.BytesIO):
    def __init__(self, fs: MemorySharedFS, path: str):
        super().__init__()
        self._fs = fs
        self._path = path

    def close(self) -> None:
        self._fs._write(self._path, self.getvalue())
        super().close()


class ConsolidatedCheckpoint:
    MAGIC = b"CKPT"
    VERSION = 1

    def __init__(self, fs: SharedFS):
        self.fs = fs

    def save(self, path: str, world_size: int, rank_data: Dict[int, bytes], metadata: Dict[str, Any]) -> None:
        lock = self.fs.lock(path)
        try:
            with self.fs.open(path, "wb") as f:
                f.write(self.MAGIC)
                f.write(struct.pack("<I", self.VERSION))
                f.write(struct.pack("<I", world_size))

                meta_bytes = json.dumps(metadata).encode("utf-8")
                f.write(struct.pack("<Q", len(meta_bytes)))
                f.write(meta_bytes)

                f.write(struct.pack("<I", len(rank_data)))
                for rank in sorted(rank_data.keys()):
                    data = rank_data[rank]
                    f.write(struct.pack("<I", rank))
                    f.write(struct.pack("<Q", len(data)))
                    f.write(data)
        finally:
            self.fs.unlock(lock)

    def save_rank(self, path: str, rank: int, world_size: int, data: bytes, metadata: Dict[str, Any]) -> None:
        lock = self.fs.lock(path)
        try:
            existing_data = {}
            existing_meta = metadata

            if self.fs.exists(path):
                with self.fs.open(path, "rb") as f:
                    content = f.read()
                if len(content) > 0:
                    existing_data, existing_meta, _ = self._parse(content)
                    existing_meta.update(metadata)

            existing_data[rank] = data

            with self.fs.open(path, "wb") as f:
                f.write(self.MAGIC)
                f.write(struct.pack("<I", self.VERSION))
                f.write(struct.pack("<I", world_size))

                meta_bytes = json.dumps(existing_meta).encode("utf-8")
                f.write(struct.pack("<Q", len(meta_bytes)))
                f.write(meta_bytes)

                f.write(struct.pack("<I", len(existing_data)))
                for r in sorted(existing_data.keys()):
                    d = existing_data[r]
                    f.write(struct.pack("<I", r))
                    f.write(struct.pack("<Q", len(d)))
                    f.write(d)
        finally:
            self.fs.unlock(lock)

    def load(self, path: str) -> tuple:
        with self.fs.open(path, "rb") as f:
            content = f.read()
        return self._parse(content)

    def load_rank(self, path: str, rank: int) -> tuple:
        rank_data, metadata, world_size = self.load(path)
        return rank_data.get(rank, b""), metadata, world_size

    def _parse(self, content: bytes) -> tuple:
        offset = 0

        magic = content[offset:offset + 4]
        offset += 4
        if magic != self.MAGIC:
            raise ValueError("Invalid checkpoint file")

        version = struct.unpack("<I", content[offset:offset + 4])[0]
        offset += 4

        world_size = struct.unpack("<I", content[offset:offset + 4])[0]
        offset += 4

        meta_len = struct.unpack("<Q", content[offset:offset + 8])[0]
        offset += 8
        meta_bytes = content[offset:offset + meta_len]
        offset += meta_len
        metadata = json.loads(meta_bytes.decode("utf-8"))

        num_ranks = struct.unpack("<I", content[offset:offset + 4])[0]
        offset += 4

        rank_data = {}
        for _ in range(num_ranks):
            rank = struct.unpack("<I", content[offset:offset + 4])[0]
            offset += 4
            data_len = struct.unpack("<Q", content[offset:offset + 8])[0]
            offset += 8
            rank_data[rank] = content[offset:offset + data_len]
            offset += data_len

        return rank_data, metadata, world_size


def create_shared_fs(config: DistributedConfig) -> SharedFS:
    if config.shared_fs_type == SharedFSType.MEMORY:
        return MemorySharedFS()
    else:
        return LocalSharedFS(config.shared_fs_path)
