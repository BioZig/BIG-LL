# BIG-LL

Bitsliced genetic algorithm and combinatorial optimization engine written in Zig with Apple Silicon Metal GPU acceleration.

BIG-LL evaluates and recombines bitboard chromosomes directly in CPU registers and GPU ALUs without runtime heap allocations, pointer chasing, or heuristic repair operators.

---

## Technical Invariants

### 1. Zero Runtime Heap Allocation
All population slabs, offspring buffers, deduplication tables, linkage correlation matrices, and island migration rings are pre-allocated at initialization. Generational transitions execute via pointer swapping between dual population slabs (`pop_current`, `pop_next`). Runtime heap allocation in evolutionary loops is 0 bytes.

### 2. Cacheline Packing (64-Byte Alignment)
Chromosomes are stored as contiguous 64-bit word arrays aligned to 64 bytes:
```zig
pub fn Chromosome(comptime L: usize) type {
    const words_len = (L + 63) / 64;
    return struct {
        words: [words_len]u64 align(64),
        fitness: f64,
        hash: u64,
    };
}
```
Each chromosome occupies exactly one 64-byte physical cacheline, eliminating false sharing across worker threads. An island population of 1,024 individuals occupies 64 KB, fitting entirely within the 128 KB L1D cache of an Apple Firestorm P-core.

### 3. Asymmetric SoC Scheduling
Worker threads are pinned to CPU core clusters via QoS classes:
* P-cores (Firestorm, 3.2 GHz, 8-wide decode): `QOS_CLASS_USER_INTERACTIVE` (72% population share, 18% per core).
* E-cores (Icestorm, 2.06 GHz, 4-wide decode): `QOS_CLASS_UTILITY` (28% population share, 7% per core).
* Synchronization: Atomic acquire/release flags backed by a 128-iteration hardware spin loop (`std.atomic.spinLoopHint()`) before yielding.

### 4. Apple Silicon Metal GPU Acceleration (Zero-Copy UMA)
Host CPU and 1,024 Apple GPU ALUs operate on shared physical DRAM via `MTLResourceStorageModeShared`. Chromosome struct strides are passed directly to Metal Shading Language kernels, evaluating batches up to 4,194,304 individuals with zero PCIe transfer overhead.

### 5. Linkage Learning & Logic Multiplexing
* **Linkage Tracking**: Pairwise bit co-occurrence is accumulated over elite bitboards into correlation matrices to identify epistatic building blocks.
* **Logic Multiplexing Recombination**: Offspring pairs are generated via bitwise multiplexing across linkage mask `m`:
  ```text
  c1 = (p1 & m) | (p2 & ~m)
  c2 = (p2 & m) | (p1 & ~m)
  ```
* **Two-Tier Deduplication**: Clones are rejected in O(1) time prior to fitness evaluation via an L1 bitmask filter and an L2 two-choice open-addressing table.

---

## Performance Telemetry

Measured on Apple Silicon M1 (8 CPU cores: 4P + 4E; 8 GPU cores, 1,024 ALUs, Unified Memory).

### OneMax (L=256)

| Target / Runtime | Cores / Hardware | Population (N) | Latency / Gen (us) | Throughput (evals/sec) |
| :--- | :--- | :--- | :--- | :--- |
| DEAP (Python 3.12) | 1 CPU Core | 1,024 | 11,760.0 | ~85,000 |
| DEAP + Multiprocessing | 4 CPU Cores | 1,024 | 3,120.0 | ~320,000 |
| Jenetics 8.0 (Java 21) | 4 CPU Cores | 1,024 | 263.0 | ~3,800,000 |
| Pagmo2 (C++20 -O3) | 4 CPU Cores | 1,024 | 121.0 | ~8,200,000 |
| **BIG-LL (1 P-Core)** | 1 Firestorm P-Core | 2,048 | 339.4 | **6,034,669** |
| **BIG-LL (4 P-Cores)** | 4 Firestorm Islands | 4,096 | 210.1 | **19,497,960** |
| **BIG-LL (8 Hetero Cores, Asymmetric)** | 4P (72%) + 4E (28%) | 4,096 | 195.4 | **20,963,380** |
| **BIG-LL (Metal GPU UMA)** | 1,024 GPU ALUs | 65,536 | 482.8 | **135,737,583** |
| **BIG-LL (Metal GPU Zero-Copy)** | 1,024 GPU ALUs | 1,048,576 | 1,783.0 | **588,083,003** |
| **BIG-LL (Metal GPU Zero-Copy)** | 1,024 GPU ALUs | 4,194,304 | 5,892.8 | **711,762,145** |

### Markowitz Portfolio Optimization (16 Assets, Quadratic Form)

| Target / Runtime | Cores / Hardware | Batch Size (N) | Latency / Batch (us) | Throughput (evals/sec) |
| :--- | :--- | :--- | :--- | :--- |
| **BIG-LL (1 P-Core)** | 1 Firestorm P-Core | 1,024 | 117.8 | **8,689,625** |
| **BIG-LL (4 P-Cores)** | 4 Firestorm Islands | 1,024 | 39.2 | **26,150,431** |
| **BIG-LL (Metal GPU UMA)** | 1,024 GPU ALUs | 131,072 | 741.5 | **176,777,339** |
| **BIG-LL (Metal GPU Zero-Copy)** | 1,024 GPU ALUs | 4,194,304 | 4,620.8 | **907,695,920** |

---

## Domain Adapters

Decouples bitsliced chromosome words from problem domains via zero-heap decoders:

| Domain | Adapter | Decoder Logic | Benchmark Instance | Throughput (4 Islands) |
| :--- | :--- | :--- | :--- | :--- |
| **Quantitative Finance** | Reflected Gray Code | Continuous weight normalization and covariance quadratic form | S&P 500 Daily Returns | 26.15M evals/sec |
| **Power Grid** | MINLP Hybrid | Discrete generator status bitfield + Gray-coded continuous MW dispatch | IEEE RTS-96 (24 Units) | 22.10M evals/sec |
| **Bioinformatics** | Boolean Set | Hardware `@ctz` index extraction for multi-locus epistatic interaction | Marchini et al. GWAS | 22.70M evals/sec |
| **Logistics Routing** | Random Key | Fixed-point keys decoded via in-register `argsort` (repair-free permutations) | CVRPLIB A-n33-k5 | 2.24M evals/sec |
| **Airfoil Aerodynamics** | Reflected Gray Code | B-spline surface coordinates under boundary layer thickness constraints | UIUC NACA 0012 | 10.85M evals/sec |
| **5G MIMO Telecom** | Discrete Key | Channel allocation and power control for SINR sum-rate maximization | 3GPP TR 38.901 UMi | 10.31M evals/sec |

---

## Build & Test

### Requirements
* Zig `0.16.0` (or `0.16.0-dev`).
* macOS for Apple Silicon Metal GPU acceleration; Linux x86_64 / ARM64 for CPU engine.

### Commands
```bash
# Build release executable
zig build -Doptimize=ReleaseFast

# Run all 98 unit and integration tests
zig build test --summary all

# Verify 0 B runtime heap allocation invariant
zig build bench-mem

# Run automated reproduction benchmark suite
bash run_all_benchmarks.sh
```

---

## CLI Usage

```text
Usage: big_ll solve <input-file> [options]

Subcommands:
  solve <file>       Ingest dataset and execute genetic search

Options:
  --pop <n>          Population size per island (default: 1024)
  --gens <n>         Generation count (default: 200)
  --islands <n>      Number of island demes (default: 4)
  --seed <n>         PRNG seed (default: 42)
  --pretty, -p       Render interactive terminal convergence dashboard
  --plot, --svg <p>  Export publication-grade vector SVG report
  --json             Emit machine-readable JSON output with decoded phenotype
  --save-ir <p>      Compile and serialize raw input to zero-copy .bigd format
  --quiet, -q        Suppress generational terminal output
```

### Examples
```bash
# Interactive terminal solve on portfolio optimization:
./zig-out/bin/big_ll solve data/real/finance/sp500_portfolio.bigd --pretty

# Export publication-grade vector SVG report:
./zig-out/bin/big_ll solve data/real/bioinformatics/marchini_gwas.bigd --plot report.svg

# Headless execution with JSON output for pipeline integration:
./zig-out/bin/big_ll solve data/real/logistics/cvrplib_An33k5.bigd --json --quiet

# Compile raw problem instance to zero-copy .bigd cache:
./zig-out/bin/big_ll solve data/raw/logistics/A-n33-k5.vrp --save-ir /tmp/A-n33-k5.bigd
```
