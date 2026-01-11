import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.distributed.shared_fs import MemorySharedFS, LocalSharedFS, ConsolidatedCheckpoint
import tempfile
import shutil


def test_memory_fs_basic():
    fs = MemorySharedFS()
    
    with fs.open("/test/file.txt", "wb") as f:
        f.write(b"hello world")
    
    assert fs.exists("/test/file.txt")
    
    with fs.open("/test/file.txt", "rb") as f:
        data = f.read()
    
    assert data == b"hello world"


def test_memory_fs_listdir():
    fs = MemorySharedFS()
    
    with fs.open("/dir/file1.txt", "wb") as f:
        f.write(b"1")
    with fs.open("/dir/file2.txt", "wb") as f:
        f.write(b"2")
    with fs.open("/dir/subdir/file3.txt", "wb") as f:
        f.write(b"3")
    
    items = fs.listdir("/dir")
    assert "file1.txt" in items
    assert "file2.txt" in items
    assert "subdir" in items


def test_memory_fs_remove():
    fs = MemorySharedFS()
    
    with fs.open("/to_remove.txt", "wb") as f:
        f.write(b"delete me")
    
    assert fs.exists("/to_remove.txt")
    fs.remove("/to_remove.txt")
    assert not fs.exists("/to_remove.txt")


def test_local_fs_basic():
    tmpdir = tempfile.mkdtemp()
    try:
        fs = LocalSharedFS(Path(tmpdir))
        
        with fs.open("/test/file.txt", "wb") as f:
            f.write(b"local test")
        
        assert fs.exists("/test/file.txt")
        
        with fs.open("/test/file.txt", "rb") as f:
            data = f.read()
        
        assert data == b"local test"
    finally:
        shutil.rmtree(tmpdir)


def test_consolidated_checkpoint():
    fs = MemorySharedFS()
    checkpoint = ConsolidatedCheckpoint(fs)
    
    rank_data = {
        0: b"rank0 data here",
        1: b"rank1 data here",
    }
    metadata = {"epoch": 5, "step": 100, "loss": 0.5}
    
    checkpoint.save("/checkpoint.bin", world_size=2, rank_data=rank_data, metadata=metadata)
    
    loaded_data, loaded_meta, world_size = checkpoint.load("/checkpoint.bin")
    
    assert world_size == 2
    assert loaded_meta["epoch"] == 5
    assert loaded_meta["step"] == 100
    assert loaded_data[0] == b"rank0 data here"
    assert loaded_data[1] == b"rank1 data here"


def test_consolidated_checkpoint_incremental():
    fs = MemorySharedFS()
    checkpoint = ConsolidatedCheckpoint(fs)
    
    checkpoint.save_rank("/checkpoint.bin", rank=0, world_size=2, 
                        data=b"rank0 initial", metadata={"epoch": 1})
    
    checkpoint.save_rank("/checkpoint.bin", rank=1, world_size=2,
                        data=b"rank1 data", metadata={"epoch": 1})
    
    loaded_data, loaded_meta, world_size = checkpoint.load("/checkpoint.bin")
    
    assert world_size == 2
    assert 0 in loaded_data
    assert 1 in loaded_data
    assert loaded_data[0] == b"rank0 initial"
    assert loaded_data[1] == b"rank1 data"


def test_load_single_rank():
    fs = MemorySharedFS()
    checkpoint = ConsolidatedCheckpoint(fs)
    
    rank_data = {
        0: b"rank0",
        1: b"rank1",
        2: b"rank2",
    }
    checkpoint.save("/checkpoint.bin", world_size=3, rank_data=rank_data, metadata={})
    
    data, _, _ = checkpoint.load_rank("/checkpoint.bin", rank=1)
    assert data == b"rank1"
    
    data, _, _ = checkpoint.load_rank("/checkpoint.bin", rank=2)
    assert data == b"rank2"


if __name__ == "__main__":
    test_memory_fs_basic()
    print("test_memory_fs_basic passed")
    
    test_memory_fs_listdir()
    print("test_memory_fs_listdir passed")
    
    test_memory_fs_remove()
    print("test_memory_fs_remove passed")
    
    test_local_fs_basic()
    print("test_local_fs_basic passed")
    
    test_consolidated_checkpoint()
    print("test_consolidated_checkpoint passed")
    
    test_consolidated_checkpoint_incremental()
    print("test_consolidated_checkpoint_incremental passed")
    
    test_load_single_rank()
    print("test_load_single_rank passed")
    
    print("\nAll shared_fs tests passed!")
