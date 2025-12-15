#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT/src/native"
make clean
make

if [ -f "libcuda_interceptor.so" ]; then
    make install
else
    exit 1
fi

