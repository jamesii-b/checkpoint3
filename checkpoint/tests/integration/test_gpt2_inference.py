import os
import sys
import time
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from src.runtime import get_runtime

SNAPSHOT_PATH = Path(__file__).parent.parent.parent / "snapshots" / "checkpoint.bin"
RESTORE_MODE = os.environ.get('RESTORE_MODE', '0') == '1'
MAX_TOKENS = 50
SNAPSHOT_AT_TOKEN = 15

def setup_model():
    print("[TEST] Loading GPT-2 model...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    model.cuda()
    print("[TEST] Model loaded")
    return model, tokenizer

def register_model_allocations(model, runtime):
    for param in model.named_parameters():
        if param[1].is_cuda:
            runtime.register_allocation(param[1].data_ptr(), 
                                       param[1].numel() * param[1].element_size())

def fast_forward_to_token(model, tokenizer, prompt, target_idx, runtime):
    input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
    generated = input_ids
    
    for i in range(target_idx):
        runtime.set_token_index(i)
        runtime.tick_epoch()
        with torch.no_grad():
            outputs = model(generated)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated = torch.cat([generated, next_token], dim=-1)
    
    return generated

def generate_tokens(model, tokenizer, prompt, start_idx=0, max_tokens=MAX_TOKENS):
    runtime = get_runtime()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').cuda()
    generated = input_ids
    
    print(f"[TEST] Starting generation from token {start_idx}")
    
    if start_idx > 0:
        generated = fast_forward_to_token(model, tokenizer, prompt, start_idx, runtime)
    else:
        register_model_allocations(model, runtime)
    
    for i in range(start_idx, max_tokens):
        runtime.set_token_index(i)
        runtime.tick_epoch()
        
        with torch.no_grad():
            outputs = model(generated)
            next_token_logits = outputs.logits[:, -1, :]
            torch.manual_seed(42 + i)
            next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated = torch.cat([generated, next_token], dim=-1)
        
        token_text = tokenizer.decode(next_token[0])
        print(f"[TOKEN {i:2d}] {token_text}")
        
        if i == SNAPSHOT_AT_TOKEN and not RESTORE_MODE:
            create_snapshot(runtime)
    
    print(f"\n[TEST] Generated text:\n{tokenizer.decode(generated[0])}\n")

def create_snapshot(runtime):
    print(f"\n[TEST] Creating snapshot at token {SNAPSHOT_AT_TOKEN}...")
    runtime.discover_allocations()
    runtime.classify_buffers()
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    if runtime.snapshot(str(SNAPSHOT_PATH)):
        print(f"[TEST] Snapshot saved: {SNAPSHOT_PATH.stat().st_size / (1024 * 1024):.1f} MB")
        print("[TEST] Simulating process termination...")
        time.sleep(1)
        os._exit(0)
    else:
        sys.exit(1)

def restore_snapshot(runtime, model):
    print("[TEST] === RESTORE MODE ===")
    if not SNAPSHOT_PATH.exists():
        print(f"[TEST] ERROR: No snapshot found")
        return 0
    
    register_model_allocations(model, runtime)
    
    if not runtime.restore(str(SNAPSHOT_PATH)):
        print("[TEST] ERROR: Restore failed")
        return 0
    
    restored_token_idx = runtime.get_token_index()
    print(f"[TEST] Restored at token {restored_token_idx}, resuming from {restored_token_idx + 1}\n")
    return restored_token_idx + 1

def main():
    runtime = get_runtime()
    model, tokenizer = setup_model()
    prompt = "In a shocking turn of events, scientists"
    start_idx = 0
    
    if RESTORE_MODE:
        start_idx = restore_snapshot(runtime, model)
    
    generate_tokens(model, tokenizer, prompt, start_idx=start_idx)
    print("[TEST] Test completed")

if __name__ == '__main__':
    main()
