# GPU Snapshot

CUDA memory snapshot and restore.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
make
```

## Usage

```python
from src.runtime import get_runtime

runtime = get_runtime()

runtime.register_allocation(ptr, size)
runtime.set_token_index(idx)
runtime.tick_epoch()

runtime.snapshot("checkpoint.bin")
runtime.restore("checkpoint.bin")
```

## Test

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
python tests/integration/test_gpt2_inference.py
RESTORE_MODE=1 python tests/integration/test_gpt2_inference.py
```

