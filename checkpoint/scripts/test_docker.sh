#!/bin/bash
set -e

cd "$(dirname "$0")/.."

echo "=== Building Docker images ==="
docker compose -f docker/docker-compose.yml build

echo "=== Starting nodes ==="
docker compose -f docker/docker-compose.yml up -d

sleep 5

echo "=== Running unit tests on node0 ==="
docker exec checkpoint-node0 python3 -m pytest tests/unit/ -v

echo "=== Running distributed test ==="
docker exec checkpoint-node0 bash -c "cd /app && python3 -c \"
import os
import sys
sys.path.insert(0, '/app')

os.environ['DISTRIBUTED_MODE'] = 'docker'
os.environ['WORLD_SIZE'] = '1'
os.environ['NODE_RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'
os.environ['BACKEND'] = 'gloo'

from src.distributed import load_config
config = load_config()
print(f'Config loaded: mode={config.mode}, world_size={config.world_size}')

from src.distributed.shared_fs import LocalSharedFS, ConsolidatedCheckpoint
fs = LocalSharedFS('/shared/snapshots')
fs.mkdir('test')
print('SharedFS working')

from src.runtime.training_runtime import TrainingRuntime
import torch
import torch.nn as nn

model = nn.Linear(10, 5)
optimizer = torch.optim.Adam(model.parameters())

x = torch.randn(4, 10)
y = model(x)
loss = y.sum()
loss.backward()
optimizer.step()

runtime = TrainingRuntime()
runtime.register_model(model)
runtime.register_optimizer(optimizer)
runtime.set_epoch(1)
runtime.set_step(10)

state = runtime.get_training_state()
data = runtime.serialize_state(state)

checkpoint = ConsolidatedCheckpoint(fs)
checkpoint.save_rank('test/checkpoint.bin', 0, 1, data, {'epoch': 1, 'step': 10})

loaded, meta, ws = checkpoint.load_rank('test/checkpoint.bin', 0)
loaded_state = runtime.deserialize_state(loaded)

assert loaded_state.epoch == 1
assert loaded_state.step == 10
print('Checkpoint save/restore working')

print('All Docker tests passed!')
\""

echo "=== Stopping nodes ==="
docker compose -f docker/docker-compose.yml down

echo "=== All Docker tests completed successfully ==="
