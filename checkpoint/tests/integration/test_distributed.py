import os
import sys
import tempfile
import multiprocessing as mp
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.distributed import DistributedConfig, DistributedMode, SharedFSType
from src.distributed.coordinator import DistributedCoordinator
from src.distributed.shared_fs import LocalSharedFS, ConsolidatedCheckpoint
from src.runtime.training_runtime import TrainingRuntime


def make_config(rank, world_size, tmpdir):
    return DistributedConfig(
        mode=DistributedMode.DOCKER,
        node_rank=rank,
        world_size=world_size,
        local_rank=rank,
        master_addr='localhost',
        master_port=29500,
        backend='gloo',
        shared_fs_path=Path(tmpdir),
        shared_fs_type=SharedFSType.LOCAL,
        docker_network='test-net',
        docker_nodes=[],
        physical_nodes=[],
    )


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 5)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def run_worker(rank, world_size, tmpdir, test_func):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['NODE_RANK'] = str(rank)
    
    try:
        test_func(rank, world_size, tmpdir)
    except Exception as e:
        print(f"Worker {rank} failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def worker_snapshot_test(rank, world_size, tmpdir):
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    torch.manual_seed(42 + rank)
    for _ in range(5):
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        optimizer.step()
    
    config = make_config(rank, world_size, tmpdir)
    coordinator = DistributedCoordinator(config)
    coordinator.initialize()
    
    training_runtime = TrainingRuntime()
    training_runtime.register_model(model)
    training_runtime.register_optimizer(optimizer)
    training_runtime.set_epoch(1)
    training_runtime.set_step(5)
    training_runtime.set_loss(0.5)
    
    state = training_runtime.get_training_state()
    data = training_runtime.serialize_state(state)
    
    fs = LocalSharedFS(tmpdir)
    checkpoint = ConsolidatedCheckpoint(fs)
    checkpoint_path = f"test_checkpoint_rank{rank}.bin"
    
    checkpoint.save_rank(checkpoint_path, rank, 1, data, {
        'world_size': world_size,
        'epoch': 1,
        'step': 5,
    })
    
    coordinator.barrier()
    
    loaded, metadata, saved_world_size = checkpoint.load_rank(checkpoint_path, rank)
    assert loaded is not None
    assert len(loaded) > 0
    
    loaded_state = training_runtime.deserialize_state(loaded)
    assert loaded_state.epoch == 1
    assert loaded_state.step == 5
    
    coordinator.shutdown()


def worker_restore_test(rank, world_size, tmpdir):
    model = SimpleModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    torch.manual_seed(42 + rank)
    for _ in range(5):
        x = torch.randn(4, 10)
        y = torch.randn(4, 5)
        optimizer.zero_grad()
        loss = nn.MSELoss()(model(x), y)
        loss.backward()
        optimizer.step()
    
    original_fc1_weight = model.fc1.weight.data.clone()
    
    config = make_config(rank, world_size, tmpdir)
    coordinator = DistributedCoordinator(config)
    coordinator.initialize()
    
    training_runtime = TrainingRuntime()
    training_runtime.register_model(model)
    training_runtime.register_optimizer(optimizer)
    training_runtime.set_epoch(1)
    training_runtime.set_step(5)
    training_runtime.set_loss(0.5)
    
    state = training_runtime.get_training_state()
    data = training_runtime.serialize_state(state)
    
    fs = LocalSharedFS(tmpdir)
    checkpoint = ConsolidatedCheckpoint(fs)
    checkpoint_path = f"restore_test_rank{rank}.bin"
    
    checkpoint.save_rank(checkpoint_path, rank, 1, data, {
        'world_size': world_size,
        'epoch': 1,
        'step': 5,
    })
    
    coordinator.barrier()
    
    model2 = SimpleModel()
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    
    loaded, metadata, saved_world_size = checkpoint.load_rank(checkpoint_path, rank)
    loaded_state = training_runtime.deserialize_state(loaded)
    
    training_runtime2 = TrainingRuntime()
    training_runtime2.register_model(model2)
    training_runtime2.register_optimizer(optimizer2)
    training_runtime2.restore_training_state(loaded_state)
    
    assert torch.allclose(model2.fc1.weight.data, original_fc1_weight)
    
    coordinator.shutdown()


def worker_coordinator_barrier_test(rank, world_size, tmpdir):
    config = make_config(rank, world_size, tmpdir)
    coordinator = DistributedCoordinator(config)
    coordinator.initialize()
    
    result_file = Path(tmpdir) / f"rank_{rank}.txt"
    
    with open(result_file, 'w') as f:
        f.write(f"before_barrier_{rank}\n")
    
    coordinator.barrier()
    
    all_files = list(Path(tmpdir).glob("rank_*.txt"))
    assert len(all_files) == world_size
    
    coordinator.barrier()
    
    with open(result_file, 'a') as f:
        f.write(f"after_barrier_{rank}\n")
    
    coordinator.shutdown()


def test_distributed_snapshot():
    if not torch.distributed.is_available():
        print("SKIP: torch.distributed not available")
        return
    
    world_size = 2
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = mp.get_context('spawn')
        processes = []
        for rank in range(world_size):
            p = ctx.Process(target=run_worker, args=(rank, world_size, tmpdir, worker_snapshot_test))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join(timeout=60)
            assert p.exitcode == 0, f"Process exited with code {p.exitcode}"


def test_distributed_restore():
    if not torch.distributed.is_available():
        print("SKIP: torch.distributed not available")
        return
    
    world_size = 2
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = mp.get_context('spawn')
        processes = []
        for rank in range(world_size):
            p = ctx.Process(target=run_worker, args=(rank, world_size, tmpdir, worker_restore_test))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join(timeout=60)
            assert p.exitcode == 0, f"Process exited with code {p.exitcode}"


def test_coordinator_barrier():
    if not torch.distributed.is_available():
        print("SKIP: torch.distributed not available")
        return
    
    world_size = 2
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = mp.get_context('spawn')
        processes = []
        for rank in range(world_size):
            p = ctx.Process(target=run_worker, args=(rank, world_size, tmpdir, worker_coordinator_barrier_test))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join(timeout=60)
            assert p.exitcode == 0, f"Process exited with code {p.exitcode}"


if __name__ == '__main__':
    print("Running coordinator barrier test...")
    test_coordinator_barrier()
    print("PASSED: coordinator barrier test")
    
    print("Running distributed snapshot test...")
    test_distributed_snapshot()
    print("PASSED: distributed snapshot test")
    
    print("Running distributed restore test...")
    test_distributed_restore()
    print("PASSED: distributed restore test")
    
    print("\nAll distributed tests passed!")
