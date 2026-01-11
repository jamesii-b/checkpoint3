import os
import time
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import torch.distributed as dist

from src.distributed import DistributedConfig, DistributedMode, load_config
from src.distributed.shared_fs import SharedFS, ConsolidatedCheckpoint, create_shared_fs
from src.runtime.training_runtime import TrainingRuntime, get_training_runtime


class DistributedCoordinator:
    def __init__(self, config: Optional[DistributedConfig] = None):
        self.config = config or load_config()
        self.fs: Optional[SharedFS] = None
        self.checkpoint: Optional[ConsolidatedCheckpoint] = None
        self._initialized = False

    def initialize(self) -> None:
        if self._initialized:
            return

        self.fs = create_shared_fs(self.config)
        self.checkpoint = ConsolidatedCheckpoint(self.fs)

        if self.config.is_distributed:
            self._init_distributed()

        self._initialized = True

    def _init_distributed(self) -> None:
        if dist.is_initialized():
            return

        os.environ["MASTER_ADDR"] = self.config.master_addr
        os.environ["MASTER_PORT"] = str(self.config.master_port)
        os.environ["WORLD_SIZE"] = str(self.config.world_size)
        os.environ["RANK"] = str(self.config.node_rank)

        backend = self.config.backend
        if backend == "nccl" and not torch.cuda.is_available():
            backend = "gloo"

        dist.init_process_group(
            backend=backend,
            world_size=self.config.world_size,
            rank=self.config.node_rank,
        )

    def shutdown(self) -> None:
        if dist.is_initialized():
            dist.destroy_process_group()
        self._initialized = False

    def barrier(self) -> None:
        if self.config.is_distributed and dist.is_initialized():
            dist.barrier()

    def snapshot(self, path: str, runtime: Optional[TrainingRuntime] = None) -> bool:
        runtime = runtime or get_training_runtime()

        try:
            state = runtime.get_training_state()
            data = runtime.serialize_state(state)

            metadata = {
                "timestamp": time.time(),
                "epoch": state.epoch,
                "step": state.step,
                "loss": state.loss,
                "world_size": self.config.world_size,
            }

            self.barrier()

            self.checkpoint.save_rank(
                path=path,
                rank=self.config.node_rank,
                world_size=self.config.world_size,
                data=data,
                metadata=metadata,
            )

            self.barrier()

            return True
        except Exception as e:
            print(f"[COORDINATOR] Snapshot failed on rank {self.config.node_rank}: {e}")
            return False

    def restore(self, path: str, runtime: Optional[TrainingRuntime] = None) -> bool:
        runtime = runtime or get_training_runtime()

        try:
            self.barrier()

            data, metadata, saved_world_size = self.checkpoint.load_rank(
                path=path,
                rank=self.config.node_rank,
            )

            if not data:
                print(f"[COORDINATOR] No data found for rank {self.config.node_rank}")
                return False

            state = runtime.deserialize_state(data)
            runtime.restore_training_state(state)

            self.barrier()

            return True
        except Exception as e:
            print(f"[COORDINATOR] Restore failed on rank {self.config.node_rank}: {e}")
            return False

    def broadcast_tensor(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        if not self.config.is_distributed:
            return tensor

        if dist.is_initialized():
            dist.broadcast(tensor, src=src)

        return tensor

    def all_reduce(self, tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
        if not self.config.is_distributed:
            return tensor

        if dist.is_initialized():
            reduce_op = dist.ReduceOp.SUM if op == "sum" else dist.ReduceOp.AVG
            dist.all_reduce(tensor, op=reduce_op)

        return tensor

    @property
    def rank(self) -> int:
        return self.config.node_rank

    @property
    def world_size(self) -> int:
        return self.config.world_size

    @property
    def is_master(self) -> bool:
        return self.config.is_master


_coordinator: Optional[DistributedCoordinator] = None


def get_coordinator(config: Optional[DistributedConfig] = None) -> DistributedCoordinator:
    global _coordinator
    if _coordinator is None:
        _coordinator = DistributedCoordinator(config)
    return _coordinator
