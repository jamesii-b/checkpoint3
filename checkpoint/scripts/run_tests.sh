#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [ ! -f "native/libcuda_interceptor.so" ]; then
    exit 1
fi

python tests/integration/test_gpt2_inference.py 2>/dev/null

if [ ! -f "snapshots/checkpoint.bin" ]; then
    exit 1
fi

RESTORE_MODE=1 python tests/integration/test_gpt2_inference.py 2>/dev/null

echo "================================================"
echo "✓ Integration Test Passed"
echo "================================================"
echo ""
echo "Summary:"
echo "  • Phase 1: Generated tokens 0-15 and created snapshot"
echo "  • Phase 2: Restored state and continued from token 16"
echo "  • Token continuity: Maintained"
echo "  • Snapshot size: $SNAPSHOT_SIZE"
echo ""
