import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.distributed import load_config, DistributedMode, SharedFSType


def _clear_env():
    keys = ["DISTRIBUTED_MODE", "WORLD_SIZE", "NODE_RANK", "LOCAL_RANK",
            "MASTER_ADDR", "MASTER_PORT", "BACKEND", "SHARED_FS_PATH",
            "SHARED_FS_TYPE", "DOCKER_NETWORK", "DOCKER_NODES", "PHYSICAL_NODES"]
    for k in keys:
        if k in os.environ:
            del os.environ[k]


def test_config_defaults():
    _clear_env()
    config = load_config()
    assert config.mode == DistributedMode.SINGLE
    assert config.world_size == 1
    assert config.node_rank == 0
    assert config.backend == "nccl"


def test_config_from_env():
    os.environ["DISTRIBUTED_MODE"] = "docker"
    os.environ["WORLD_SIZE"] = "4"
    os.environ["NODE_RANK"] = "2"
    os.environ["MASTER_ADDR"] = "master-node"
    os.environ["MASTER_PORT"] = "12345"
    os.environ["BACKEND"] = "gloo"
    
    config = load_config()
    
    assert config.mode == DistributedMode.DOCKER
    assert config.world_size == 4
    assert config.node_rank == 2
    assert config.master_addr == "master-node"
    assert config.master_port == 12345
    assert config.backend == "gloo"
    
    del os.environ["DISTRIBUTED_MODE"]
    del os.environ["WORLD_SIZE"]
    del os.environ["NODE_RANK"]
    del os.environ["MASTER_ADDR"]
    del os.environ["MASTER_PORT"]
    del os.environ["BACKEND"]


def test_config_from_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("DISTRIBUTED_MODE=physical\n")
        f.write("WORLD_SIZE=8\n")
        f.write("NODE_RANK=3\n")
        f.write("PHYSICAL_NODES=192.168.1.1,192.168.1.2,192.168.1.3\n")
        f.flush()
        
        for key in ["DISTRIBUTED_MODE", "WORLD_SIZE", "NODE_RANK", "PHYSICAL_NODES"]:
            if key in os.environ:
                del os.environ[key]
        
        config = load_config(Path(f.name))
        
        assert config.mode == DistributedMode.PHYSICAL
        assert config.world_size == 8
        assert config.node_rank == 3
        assert len(config.physical_nodes) == 3
        
        os.unlink(f.name)
    
    for key in ["DISTRIBUTED_MODE", "WORLD_SIZE", "NODE_RANK", "PHYSICAL_NODES"]:
        if key in os.environ:
            del os.environ[key]


def test_is_distributed():
    for key in ["DISTRIBUTED_MODE", "WORLD_SIZE", "NODE_RANK", "DOCKER_NODES", "PHYSICAL_NODES"]:
        if key in os.environ:
            del os.environ[key]
    
    os.environ["WORLD_SIZE"] = "1"
    os.environ["DISTRIBUTED_MODE"] = "single"
    config = load_config()
    assert not config.is_distributed
    
    os.environ["WORLD_SIZE"] = "2"
    config = load_config()
    assert config.is_distributed
    
    del os.environ["WORLD_SIZE"]
    del os.environ["DISTRIBUTED_MODE"]


def test_is_master():
    for key in ["NODE_RANK"]:
        if key in os.environ:
            del os.environ[key]
    
    os.environ["NODE_RANK"] = "0"
    config = load_config()
    assert config.is_master
    
    os.environ["NODE_RANK"] = "1"
    config = load_config()
    assert not config.is_master
    
    del os.environ["NODE_RANK"]


if __name__ == "__main__":
    test_config_defaults()
    print("test_config_defaults passed")
    
    test_config_from_env()
    print("test_config_from_env passed")
    
    test_config_from_file()
    print("test_config_from_file passed")
    
    test_is_distributed()
    print("test_is_distributed passed")
    
    test_is_master()
    print("test_is_master passed")
    
    print("\nAll config tests passed!")
