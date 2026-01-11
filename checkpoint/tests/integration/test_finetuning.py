import os
import sys
import tempfile
import multiprocessing as mp
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

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
        master_port=29501,
        backend='gloo',
        shared_fs_path=Path(tmpdir),
        shared_fs_type=SharedFSType.LOCAL,
        docker_network='test-net',
        docker_nodes=[],
        physical_nodes=[],
    )


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=1000, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.fc(x)


def create_dummy_data(num_samples=100, seq_len=16, vocab_size=1000):
    inputs = torch.randint(0, vocab_size, (num_samples, seq_len))
    targets = torch.randint(0, vocab_size, (num_samples, seq_len))
    return TensorDataset(inputs, targets)


def train_step(model, optimizer, batch, criterion):
    optimizer.zero_grad()
    inputs, targets = batch
    outputs = model(inputs)
    loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
    loss.backward()
    optimizer.step()
    return loss.item()


def run_worker(rank, world_size, tmpdir, results_queue):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29501'
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['NODE_RANK'] = str(rank)
    
    try:
        result = run_finetuning_test(rank, world_size, tmpdir)
        results_queue.put((rank, result))
    except Exception as e:
        import traceback
        traceback.print_exc()
        results_queue.put((rank, {'error': str(e)}))


def run_finetuning_test(rank, world_size, tmpdir):
    config = make_config(rank, world_size, tmpdir)
    coordinator = DistributedCoordinator(config)
    coordinator.initialize()
    
    torch.manual_seed(42)
    model = SimpleTransformer(vocab_size=1000, d_model=64, nhead=4, num_layers=2)
    model = DDP(model)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    dataset = create_dummy_data(100, 16, 1000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)
    
    training_runtime = TrainingRuntime()
    training_runtime.register_model(model.module)
    training_runtime.register_optimizer(optimizer)
    
    fs = LocalSharedFS(tmpdir)
    checkpoint = ConsolidatedCheckpoint(fs)
    checkpoint_path = "finetuning_checkpoint.bin"
    
    losses_before_checkpoint = []
    checkpoint_step = 5
    total_steps = 10
    
    step = 0
    for epoch in range(2):
        sampler.set_epoch(epoch)
        for batch in dataloader:
            loss = train_step(model, optimizer, batch, criterion)
            losses_before_checkpoint.append(loss)
            step += 1
            
            training_runtime.set_epoch(epoch)
            training_runtime.set_step(step)
            training_runtime.set_loss(loss)
            
            if step == checkpoint_step:
                state = training_runtime.get_training_state()
                data = training_runtime.serialize_state(state)
                
                checkpoint.save_rank(checkpoint_path, rank, world_size, data, {
                    'world_size': world_size,
                    'epoch': epoch,
                    'step': step,
                    'loss': loss,
                })
                
                coordinator.barrier()
                
                model_state_at_checkpoint = {k: v.clone() for k, v in model.module.state_dict().items()}
                optimizer_state_at_checkpoint = {
                    'step': optimizer.state[optimizer.param_groups[0]['params'][0]]['step'].clone() 
                    if optimizer.param_groups[0]['params'][0] in optimizer.state else None
                }
            
            if step >= total_steps:
                break
        if step >= total_steps:
            break
    
    coordinator.barrier()
    
    torch.manual_seed(42)
    model2 = SimpleTransformer(vocab_size=1000, d_model=64, nhead=4, num_layers=2)
    model2 = DDP(model2)
    optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.001)
    
    loaded, metadata, saved_world_size = checkpoint.load_rank(checkpoint_path, rank)
    
    training_runtime2 = TrainingRuntime()
    training_runtime2.register_model(model2.module)
    training_runtime2.register_optimizer(optimizer2)
    
    loaded_state = training_runtime2.deserialize_state(loaded)
    training_runtime2.restore_training_state(loaded_state)
    
    for name, param in model2.module.named_parameters():
        original_param = model_state_at_checkpoint[name]
        if not torch.allclose(param.data, original_param, atol=1e-6):
            coordinator.shutdown()
            return {
                'success': False,
                'error': f'Model parameter {name} mismatch after restore'
            }
    
    losses_after_restore = []
    step = checkpoint_step
    for epoch in range(2):
        if epoch < loaded_state.epoch:
            continue
        sampler.set_epoch(epoch)
        batch_idx = 0
        for batch in dataloader:
            if epoch == loaded_state.epoch and batch_idx < checkpoint_step:
                batch_idx += 1
                continue
            
            loss = train_step(model2, optimizer2, batch, criterion)
            losses_after_restore.append(loss)
            step += 1
            batch_idx += 1
            
            if step >= total_steps:
                break
        if step >= total_steps:
            break
    
    coordinator.barrier()
    coordinator.shutdown()
    
    return {
        'success': True,
        'rank': rank,
        'losses_before': losses_before_checkpoint[:checkpoint_step],
        'checkpoint_step': checkpoint_step,
        'restored_epoch': loaded_state.epoch,
        'restored_step': loaded_state.step,
    }


def test_distributed_finetuning():
    if not torch.distributed.is_available():
        print("SKIP: torch.distributed not available")
        return
    
    world_size = 2
    with tempfile.TemporaryDirectory() as tmpdir:
        ctx = mp.get_context('spawn')
        results_queue = ctx.Queue()
        
        processes = []
        for rank in range(world_size):
            p = ctx.Process(target=run_worker, args=(rank, world_size, tmpdir, results_queue))
            p.start()
            processes.append(p)
        
        for p in processes:
            p.join(timeout=120)
        
        results = {}
        while not results_queue.empty():
            rank, result = results_queue.get()
            results[rank] = result
        
        for rank, result in results.items():
            if 'error' in result:
                raise AssertionError(f"Rank {rank} failed: {result['error']}")
            assert result['success'], f"Rank {rank} test failed"
            print(f"Rank {rank}: checkpoint_step={result['checkpoint_step']}, "
                  f"restored_step={result['restored_step']}")
        
        print("Distributed fine-tuning test PASSED!")


if __name__ == '__main__':
    print("Running distributed fine-tuning end-to-end test...")
    test_distributed_finetuning()
    print("\nAll tests passed!")
