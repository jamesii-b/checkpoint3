#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

if [ ! -f "native/libcuda_interceptor.so" ]; then
    exit 1
fi

python tests/integration/test_gpt2_inference.py 2>&1 | head -30
RESTORE_MODE=1 python tests/integration/test_gpt2_inference.py 2>&1 | head -30

echo ""
echo "Key metrics to check:"
echo "  • Buffers copied D->H (Device to Host)"
echo "  • Buffers copied H->D (Host to Device)"
echo "  • Token continuity across phases"
echo "  • Snapshot file size (~475MB expected)"
echo ""
