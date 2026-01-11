source venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH

python3 tests/integration/test_gpt2_inference.py                    # Create snapshot
RESTORE_MODE=1 python3 tests/integration/test_gpt2_inference.py     # Restore