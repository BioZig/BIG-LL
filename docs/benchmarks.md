# Component Documentation: Performance Benchmarking (`benchmarks/`)

## 1. Specification

The `bench_throughput.zig` benchmark quantifies execution throughput, generational latency, and memory behavior of the BIG-LL genetic engine under ReleaseFast optimization.

### Benchmark Setup

* **Platform:** Apple Silicon (macOS ARM64, Darwin kernel).
* **Compiler:** Zig 0.16.0-dev (`-OReleaseFast`).
* **Timer:** High-precision monotonic timer via `std.c.mach_absolute_time()` with `mach_timebase_info` scaling.
* **Test Workload:**
  * Population Size: $N = 256$ individuals.
  * Chromosome Dimension: $L = 256$ bits (4 $\times$ 64-bit words).
  * Generations: 50 generational epochs.
  * Total Evaluations: $12,800$ bitwise evaluations.
  * Linkage Learning: Full in-register transposition, adaptive statistical thresholding, and sparse graph recombination active on every generation.

### Benchmark Results

| Metric | Measured Value | Notes |
| :--- | :--- | :--- |
| **Throughput** | **1,753,384 evals / sec** | > 1.75 Million evaluations per second |
| **Generational Latency** | **146.00 us / generation** | Complete evolutionary cycle including linkage & selection |
| **Initialization Time** | **0.09 ms** | Setup of dual-slab arena and L1 dedup table |
| **L1-DEDUP Hit Ratio** | Tracked online | Filters redundant duplicate evaluations |
| **Heap Allocations in Loop** | **0 bytes** | Zero dynamic heap allocations during generation loop |

### Hardware Profiling & Efficiency

* **Memory Residency:** Entire $L=256$ population fits within $256 \times 64\text{ bytes} = 16\text{ KB}$ (48B payload padded to 64B cacheline alignment), well within L1 data cache ($64\text{ KB}$ - $128\text{ KB}$).
* **L2 Cache Tiling:** Linkage matrix operations operate in $512\text{ KB}$ dense tiles, avoiding main DRAM bus stalls.
* **SIMD Utilization:** Single-cycle 64-bit bitwise AND (`&`) and hardware population count (`@popCount`) operations saturate execution units.
