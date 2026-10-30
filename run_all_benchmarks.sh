#!/usr/bin/env bash
# BIG-LL Automated Verification and Full Benchmark Suite
# Target Architecture: Apple Silicon M-Series SoC (Firestorm/Icestorm + Metal UMA)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

CACHE_FLAGS="--global-cache-dir /tmp/zig-global-cache --cache-dir /tmp/zig-cache"

echo "[BIG-LL] System Verification and Benchmark Pipeline"
echo "[BIG-LL] Hardware Environment: $(uname -s) $(uname -m)"

echo ""
echo "=== Step 1/5: Full Unit Test Suite (83 Tests) ==="
zig build test --summary all ${CACHE_FLAGS}

echo ""
echo "=== Step 2/5: Zero-Heap-Allocation Invariant Audit ==="
zig build bench-mem ${CACHE_FLAGS}

echo ""
echo "=== Step 3/5: CPU Throughput & Island Topology Scaling ==="
zig build bench ${CACHE_FLAGS}

echo ""
echo "=== Step 4/5: Apple Silicon Dedicated Metal GPU Batch Evaluation ==="
zig build bench-metal ${CACHE_FLAGS}

echo ""
echo "=== Step 5/7: Synthetic Industrial Domain Suite (6 Domains x 3 Tiers) ==="
zig build bench-domains ${CACHE_FLAGS}

echo ""
echo "=== Step 6/7: Scaled Multi-Batch L2 Cache Tiling Benchmark (L=256..8192) ==="
zig build bench-scaled ${CACHE_FLAGS}

echo ""
echo "=== Step 7/7: Real-World Industrial Dataset Suite (CVRPLIB, IEEE, S&P 500, etc.) ==="
zig build bench-real ${CACHE_FLAGS}

echo ""
echo "[BIG-LL] Benchmark Pipeline Completed Successfully."
