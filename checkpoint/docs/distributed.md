

## Physical Multi-Node Setup

### Step 1: Setup NFS (Shared Filesystem)

**On Master Node (192.168.1.100):**

```bash
sudo apt update
sudo apt install nfs-kernel-server

sudo mkdir -p /shared/checkpoint4
sudo mkdir -p /shared/snapshots

sudo chown -R $USER:$USER /shared

echo "/shared 192.168.1.0/24(rw,sync,no_subtree_check,no_root_squash)" | sudo tee -a /etc/exports

sudo exportfs -ra
sudo systemctl restart nfs-kernel-server

sudo ufw allow from 192.168.1.0/24 to any port nfs
sudo ufw allow 29500
```

**On Worker Nodes (192.168.1.101, .102, .103):**

```bash
sudo apt update
sudo apt install nfs-common

sudo mkdir -p /shared

sudo mount 192.168.1.100:/shared /shared

echo "192.168.1.100:/shared /shared nfs defaults 0 0" | sudo tee -a /etc/fstab
```

**Verify on all nodes:**
```bash
ls /shared/
touch /shared/test_from_$(hostname)
ls /shared/
```

### Step 2: Setup Code and Environment

**On Master Node:**

```bash
cd /shared
git clone <your-repo-url> checkpoint4
cd checkpoint4

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install torch torchvision
```

All workers automatically have access via NFS.

### Step 3: Configure Each Node

Edit `/shared/checkpoint4/config/distributed.physical.env`:

```env
DISTRIBUTED_MODE=physical
WORLD_SIZE=4
MASTER_ADDR=192.168.1.100
MASTER_PORT=29500
BACKEND=nccl
SHARED_FS_PATH=/shared/snapshots
SHARED_FS_TYPE=local
PHYSICAL_NODES=192.168.1.100,192.168.1.101,192.168.1.102,192.168.1.103
```

### Step 4: Launch Training

**Manual Launch (run on each node):**

```bash
# Node 0 (192.168.1.100):
cd /shared/checkpoint4
source venv/bin/activate
NODE_RANK=0 python train.py

# Node 1 (192.168.1.101):
cd /shared/checkpoint4
source venv/bin/activate
NODE_RANK=1 python train.py

# Node 2, 3... same pattern
```

**Automated Launch (from master):**

```bash
cd /shared/checkpoint4
./scripts/launch_distributed.sh
```

---

## Using torchrun (Alternative)

PyTorch's built-in launcher handles rank assignment:

**On each node:**

```bash
torchrun \
  --nnodes=4 \
  --nproc_per_node=1 \
  --node_rank=<THIS_NODE_RANK> \
  --master_addr=192.168.1.100 \
  --master_port=29500 \
  train.py
```

---

## Checkpoint/Restore

### Taking a Checkpoint

```python
from src.distributed.coordinator import DistributedCoordinator
from src.runtime.training_runtime import TrainingRuntime

coordinator = DistributedCoordinator()
coordinator.initialize()

runtime = TrainingRuntime()
runtime.register_model(model)
runtime.register_optimizer(optimizer)
runtime.set_epoch(epoch)
runtime.set_step(step)

coordinator.snapshot("/shared/snapshots/checkpoint.bin", runtime)
```

### Restoring

```python
coordinator = DistributedCoordinator()
coordinator.initialize()

runtime = TrainingRuntime()
runtime.register_model(model)
runtime.register_optimizer(optimizer)

coordinator.restore("/shared/snapshots/checkpoint.bin", runtime)
resumed_epoch = runtime._epoch
resumed_step = runtime._step
```

---

## Network Requirements

| Port | Purpose |
|------|---------|
| 29500 | PyTorch distributed (MASTER_PORT) |
| 2049 | NFS |
| 111 | NFS portmapper |

**Firewall (on all nodes):**

```bash
sudo ufw allow from 192.168.1.0/24
```

---

## Troubleshooting

### "Connection refused" on init_process_group

- Check MASTER_ADDR is correct
- Check port 29500 is open: `nc -zv 192.168.1.100 29500`
- Start master node (rank 0) first

### "NCCL error"

- Use `BACKEND=gloo` for CPU-only or debugging
- For GPUs, ensure CUDA versions match across nodes
- Set `NCCL_DEBUG=INFO` for verbose output

### NFS mount issues

- Check NFS server is running: `sudo systemctl status nfs-kernel-server`
- Check exports: `showmount -e 192.168.1.100`
- Check firewall allows NFS traffic

### Checkpoints not visible on workers

- Verify NFS mount: `df -h | grep shared`
- Check file permissions
- Ensure SHARED_FS_PATH matches on all nodes

---

## Example: Distributed Fine-tuning

```python
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from src.distributed import load_config
from src.distributed.coordinator import DistributedCoordinator
from src.runtime.training_runtime import TrainingRuntime

config = load_config()
coordinator = DistributedCoordinator(config)
coordinator.initialize()

model = YourModel().cuda()
model = DDP(model)
optimizer = torch.optim.Adam(model.parameters())

runtime = TrainingRuntime()
runtime.register_model(model.module)
runtime.register_optimizer(optimizer)

if os.path.exists("/shared/snapshots/checkpoint.bin"):
    coordinator.restore("/shared/snapshots/checkpoint.bin", runtime)
    start_epoch = runtime._epoch
else:
    start_epoch = 0

for epoch in range(start_epoch, num_epochs):
    for step, batch in enumerate(dataloader):
        loss = train_step(model, batch)
        
        runtime.set_epoch(epoch)
        runtime.set_step(step)
        runtime.set_loss(loss.item())
        
        if step % checkpoint_interval == 0:
            coordinator.snapshot("/shared/snapshots/checkpoint.bin", runtime)

coordinator.shutdown()
```

---

## Quick Reference

| Nodes | Config |
|-------|--------|
| Single machine | `DISTRIBUTED_MODE=single` |
| Docker containers | `DISTRIBUTED_MODE=docker` |
| Physical machines | `DISTRIBUTED_MODE=physical` |

| Backend | Use case |
|---------|----------|
| `nccl` | Multi-GPU (fastest) |
| `gloo` | CPU or debugging |
