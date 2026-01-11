import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class DistributedMode(Enum):
    SINGLE = "single"
    DOCKER = "docker"
    PHYSICAL = "physical"


class SharedFSType(Enum):
    LOCAL = "local"
    NFS = "nfs"
    MEMORY = "memory"


@dataclass
class DistributedConfig:
    mode: DistributedMode
    world_size: int
    node_rank: int
    local_rank: int
    master_addr: str
    master_port: int
    backend: str
    shared_fs_path: Path
    shared_fs_type: SharedFSType
    docker_network: str
    docker_nodes: List[str]
    physical_nodes: List[str]

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_master(self) -> bool:
        return self.node_rank == 0


def load_config(env_path: Optional[Path] = None) -> DistributedConfig:
    if env_path and env_path.exists():
        _load_env_file(env_path)

    mode = DistributedMode(os.environ.get("DISTRIBUTED_MODE", "single"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    node_rank = int(os.environ.get("NODE_RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    master_addr = os.environ.get("MASTER_ADDR", "localhost")
    master_port = int(os.environ.get("MASTER_PORT", "29500"))
    backend = os.environ.get("BACKEND", "nccl")
    shared_fs_path = Path(os.environ.get("SHARED_FS_PATH", "/shared/snapshots"))
    shared_fs_type = SharedFSType(os.environ.get("SHARED_FS_TYPE", "local"))
    docker_network = os.environ.get("DOCKER_NETWORK", "checkpoint-net")
    docker_nodes = _parse_list(os.environ.get("DOCKER_NODES", ""))
    physical_nodes = _parse_list(os.environ.get("PHYSICAL_NODES", ""))

    if mode == DistributedMode.DOCKER and docker_nodes:
        world_size = max(world_size, len(docker_nodes))
    elif mode == DistributedMode.PHYSICAL and physical_nodes:
        world_size = max(world_size, len(physical_nodes))

    return DistributedConfig(
        mode=mode,
        world_size=world_size,
        node_rank=node_rank,
        local_rank=local_rank,
        master_addr=master_addr,
        master_port=master_port,
        backend=backend,
        shared_fs_path=shared_fs_path,
        shared_fs_type=shared_fs_type,
        docker_network=docker_network,
        docker_nodes=docker_nodes,
        physical_nodes=physical_nodes,
    )


def _load_env_file(path: Path) -> None:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def _parse_list(value: str) -> List[str]:
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]
