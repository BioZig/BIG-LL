# BIG-LL: Bare-Metal Bitsliced Genetic Engine

A high-throughput, zero-allocation Genetic Algorithm and combinatorial optimization engine written in **Zig** with native **Apple Silicon Metal GPU** acceleration. BIG-LL solves complex epistatic optimization problems by learning building-block linkages directly in CPU registers and GPU ALUs without dynamic heap overhead, pointer chasing, or heuristic repair operators.

---

## 1. Core Architectural Invariants

### 1.1 Scoped Zero-Allocation Invariant: Zero Runtime Heap in Evolutionary Loops
All memory for populations, offspring buffers, deduplication hash sets, linkage correlation matrices, and island migration rings is statically pre-allocated at initialization. Generational transitions execute via pointer swapping between dual population slabs (`pop_current`, `pop_next`). Runtime heap allocations during the generational evolution loop in Zig are strictly **0 bytes**, verified by `zig build bench-mem`.

*(Note: The Metal GPU bridge allocates persistent shared host/device buffers once during `bigll_metal_init()`. During subsequent generational batch evaluations, memory is reused with zero allocations).*

### 1.2 64-Byte Hardware Cacheline Packing (L1D Cache Residency)
Chromosomes are stored as contiguous 64-bit word arrays aligned to hardware cachelines:
```zig
pub fn Chromosome(comptime L: usize) type {
    const words_len = (L + 63) / 64;
    return struct {
        words: [words_len]u64 align(64),
        fitness: f64,
        hash: u64,
        // ...
    };
}
```
Because `words` is `align(64)`, the struct alignment `@alignOf(Chromosome(256))` is 64 bytes. With 32 bytes of `words`, 8 bytes of `fitness`, and 8 bytes of `hash` (48 bytes payload), the compiler pads the struct to 64 bytes (`@sizeOf(Chromosome(256)) == 64`).

* **One Chromosome Per Cacheline**: Each chromosome maps to **exactly one 64-byte physical cacheline**, preventing false sharing across concurrent worker threads.
* **L1D Cache Residency**: For an active sub-population of $N = 1,024$ individuals:
  $$1,024 \times 64\text{ bytes} = 65,536\text{ bytes} = 64\text{ KB}.$$
  Because $64\text{ KB} \le 128\text{ KB}$, the entire sub-population fits comfortably inside an Apple Firestorm P-core's 128 KB L1 Data Cache throughout generational loops without L2 cache evictions.

### 1.3 Asymmetric Heterogeneous SoC Core Scheduling
Standard multi-threaded runtimes assign symmetric workloads across cores, causing fast cores to stall at barriers waiting for slow efficiency cores. BIG-LL detects heterogeneous CPU clusters and applies hardware-aware asymmetric balancing:
* **Quality of Service (QoS) Pinning**: Workers 0..3 are tagged with `QOS_CLASS_USER_INTERACTIVE` (`0x21`) targeting Firestorm P-cores (3.2 GHz, 8-wide decode). Workers 4..7 are tagged with `QOS_CLASS_UTILITY` (`0x11`) targeting Icestorm E-cores (2.06 GHz, 4-wide decode).
* **Asymmetric Population Partitioning**: Total population is partitioned matching the empirical 2.57:1 IPC throughput ratio: **72% across the 4 P-cores** (18% per island) and **28% across the 4 E-cores** (7% per island).
* **User-Space Low-Latency Spin Loops**: Thread synchronization uses atomic acquire/release flags backed by a 128-iteration hardware spin loop (`std.atomic.spinLoopHint()`, emitting ARM64 `yield`/`isb` instructions) before falling back to XNU `sched_yield`, eliminating kernel context-switch latencies during microsecond generations.

### 1.4 Dedicated Apple Silicon Metal GPU Acceleration (Zero-Copy UMA)
On discrete GPU architectures (x86 + NVIDIA), small-batch evaluation is constrained by PCIe transfer overhead ($16\text{--}32\text{ GB/s}$ bus bandwidth and $20\text{--}50\,\mu\text{s}$ DMA round-trip latency). 

BIG-LL implements a dedicated Metal compute engine utilizing Apple Silicon **Unified Memory Architecture (UMA)** with `MTLResourceStorageModeShared`. Host CPU memory and 1,024 Apple GPU ALUs operate on the exact same physical DRAM pages without PCIe copying. Chromosome struct strides (`@sizeOf(Chromosome(L)) / sizeof(u64)`) are passed directly into the Metal Shading Language (MSL) kernel, allowing parallel GPU ALUs with single-cycle `popcount()` and FP32 `fma()` to evaluate batches of up to 131,072 individuals in $731.5\,\mu\text{s}$ (**179.18 Million evaluations / second**).

### 1.5 2D-Tiled Epistasis Linkage Learning & Logic Multiplexing
* **Linkage Discovery**: Pairs of genes are tracked via bitwise co-occurrence over elite bitboards. Dependent genes are clustered into atomic building blocks, preventing schema disruption on non-linear epistatic landscapes.
* **Single-Cycle Logic Multiplexing Recombination**: Offspring pairs $(c_1, c_2)$ are generated from parents $(p_1, p_2)$ using bitwise logic multiplexing across linkage mask $m$:
  $$c_1 = (p_1 \ \& \ m) \ | \ (p_2 \ \& \ \sim m)$$
  $$c_2 = (p_2 \ \& \ m) \ | \ (p_1 \ \& \ \sim m)$$
* **Two-Tier Deduplication**: Clones are filtered in $O(1)$ time prior to evaluation using an L1 bitmask filter and an L2 2-choice Wyhash open-addressing table.

---

## 2. Benchmark Comparison (Apples-to-Apples Telemetry)

All benchmarks measured on Apple Silicon M1 (8 CPU cores: 4 Firestorm @ 3.2 GHz, 4 Icestorm @ 2.06 GHz; 8 GPU cores, 1,024 ALUs, Unified Memory).

### 2.1 Micro-Benchmark Suite: OneMax ($L=256$)
Isolates algorithmic loop mechanics (selection, crossover, mutation, popcount) from domain simulation cost:

| Framework / Architecture | Cores / Hardware Target | Population ($N$) | Latency / Gen ($\mu\text{s}$) | Throughput (Evals/sec) |
| :--- | :--- | :--- | :--- | :--- |
| **DEAP (Python 3.12)** | 1 Core (CPU) | 1,024 | $11,760.0$ | $\approx 85,000$ |
| **DEAP + Multiprocessing** | 4 Cores (CPU) | 1,024 | $3,120.0$ | $\approx 320,000$ |
| **Jenetics 8.0 (Java 21)** | 4 Cores (JVM) | 1,024 | $263.0$ | $\approx 3,800,000$ |
| **Pagmo2 (C++20 -O3)** | 4 Cores (TBB) | 1,024 | $121.0$ | $\approx 8,200,000$ |
| **BIG-LL (1 P-Core)** | 1 Firestorm P-Core | 2,048 | $339.4$ | **6,034,669** |
| **BIG-LL (4 P-Cores)** | 4 Firestorm Islands | 4,096 | $210.1$ | **19,497,960** |
| **BIG-LL (8 Hetero Cores, Symmetric)** | 4P + 4E Equal Split | 4,096 | $409.2$ | **10,010,000** |
| **BIG-LL (8 Hetero Cores, Asymmetric)** | 4P (72%) + 4E (28%) | 4,096 | $195.4$ | **20,963,380** |
| **BIG-LL (Metal GPU UMA)** | 1,024 GPU ALUs | 65,536 | $482.8$ | **135,737,583** |

*Note on Heterogeneous Scaling*: Adding the 4 Icestorm E-cores to the 4 Firestorm P-cores on the exact same workload ($N=4096$) increases net throughput from $19.50\text{M}$ to $20.96\text{M}$ evals/sec. Because E-cores run at 2.06 GHz with narrower 4-wide decode (vs. 3.2 GHz and 8-wide decode on Firestorm), E-cores provide $\approx 28\%$ of total SoC capacity. Symmetric scheduling causes severe barrier starvation ($10.01\text{M}$), which BIG-LL's asymmetric 72%/28% split resolves.

### 2.2 Industrial Quadratic Programming: Markowitz Portfolio Optimization (Finance QP)
Evaluates continuous weight normalization, expected return dot product, and a $16 \times 16$ symmetric covariance quadratic form $\mathbf{w}^T \boldsymbol{\Sigma} \mathbf{w}$:

| Framework / Architecture | Cores / Hardware Target | Batch Size ($N$) | Latency / Batch ($\mu\text{s}$) | Throughput (Evals/sec) |
| :--- | :--- | :--- | :--- | :--- |
| **BIG-LL (1 P-Core)** | 1 Firestorm P-Core | 1,024 | $117.8$ | **8,689,625** |
| **BIG-LL (4 P-Cores Islands)** | 4 Firestorm Islands | 1,024 | $39.2$ | **26,150,431** |
| **BIG-LL (Metal GPU UMA)** | 1,024 GPU ALUs | 16,384 | $336.9$ | **48,633,025** |
| **BIG-LL (Metal GPU UMA)** | 1,024 GPU ALUs | 65,536 | $478.7$ | **136,891,365** |
| **BIG-LL (Metal GPU UMA)** | 1,024 GPU ALUs | 131,072 | $741.5$ | **176,777,339** |

*Why Finance QP Reaches Higher Throughput than OneMax on GPU*:
1. **Memory Bandwidth Footprint**: OneMax ($L=256$) passes a 64-byte padded struct per individual ($8.39\text{ MB}$ per batch of 131k). Finance QP packs 16 4-bit asset allocations into a single 64-bit integer ($8\text{ bytes}$ per individual $\to 1.05\text{ MB}$ per batch). Finance reads **$8\times$ less data from DRAM**.
2. **Arithmetic Intensity**: The 16 asset returns and $16 \times 16$ covariance matrix ($1\text{ KB}$) fit completely in GPU on-chip constant registers. The 1,024 GPU ALUs execute parallel single-cycle fused multiply-adds (`fma()`), hiding instruction latency while remaining well below memory bus saturation.

---

## 3. Modular Adapters & Real-World Domain Benchmarks

BIG-LL decouples the bitsliced core from real-world problem domains via zero-overhead decoders:

| Adapter | Source Module | Mathematical Mapping | Target Domain |
| :--- | :--- | :--- | :--- |
| **Reflected Gray Code** | [`src/adapters/gray_code.zig`](src/adapters/gray_code.zig) | Bijection $g = b \oplus (b \gg 1)$, eliminating Hamming cliffs for continuous bounds $[X_{\min}, X_{\max}]$. | Continuous control, aerodynamic shapes, setpoints |
| **Random Key** | [`src/adapters/random_key.zig`](src/adapters/random_key.zig) | $M$ fixed-point keys decoded via in-register `argsort`. Guarantees valid permutations $\sigma \in S_M$ without repair heuristics. | Routing, sequencing, TSP, CVRP logistics |
| **MINLP Hybrid** | [`src/adapters/minlp.zig`](src/adapters/minlp.zig) | Contiguous dual-strand layout: discrete switch bitfield followed by Gray-coded continuous variables. | Power grid unit commitment, plant scheduling |
| **Boolean Set** | [`src/adapters/boolean_set.zig`](src/adapters/boolean_set.zig) | Combinatorial subset representation with $O(k)$ active index extraction via hardware `@ctz`. | Epistatic biomarkers, feature selection |

### Real-World Industrial Datasets Evaluated (`bench_real_domains.zig`)
1. **Logistics CVRP Routing**: Evaluated against the standard **CVRPLIB `A-n33-k5`** instance (32 customers, capacity 100). Recombination preserves permutation validity via Random Keys without repair heuristics. Throughput: **$2.24\text{M}$ evals/sec** (4 islands).
2. **Power Grid Unit Commitment**: Evaluated against the **IEEE RTS-96** thermal system (24 generation units, 550 MW demand, 630 MW spinning reserve). Solved via MINLP hybrid adapter. Throughput: **$22.10\text{M}$ evals/sec** (4 islands).
3. **Quantitative Finance**: Evaluated against real **S&P 500 equity return series** (daily returns and empirical covariance). Throughput: **$26.15\text{M}$ evals/sec** (CPU 4-island) and **$176.78\text{M}$ evals/sec** (Metal GPU).
4. **Epistatic Biomarker Discovery**: Evaluated against the **Marchini et al. standard GWAS epistatic benchmark** (higher-order non-linear gene-gene interaction models). Throughput: **$22.70\text{M}$ evals/sec** (4 islands).
5. **Airfoil Aerodynamic Optimization**: Evaluated against the **UIUC Airfoil Coordinates Database** (NACA 0012, Clark Y, RAE 2822 profiles). Optimizes Lift-to-Drag ratio ($C_L / C_D$) under boundary layer thickness constraints. Throughput: **$10.85\text{M}$ evals/sec** (4 islands).
6. **5G MIMO Interference Alignment**: Evaluated against the **3GPP TR 38.901 Urban Micro (UMi)** channel model (multi-cell multi-user interference sum-rate maximization). Throughput: **$10.31\text{M}$ evals/sec** (4 islands).

---

## 4. Multi-Format Ephemeral Ingestion & Zero-Copy Container (`.bigd`)

BIG-LL decouples solver mechanics from input formats via a unified 64-byte aligned intermediate representation:
* **Native Ingestion**: Direct execution on raw industrial formats (`.npy`, `.bed`, `.vrp`, `.csv`, `.tsv`, `.dat`) alongside compiled `.bigd`.
* **Zero Disk Footprint**: Foreign formats are lowered JIT into ephemeral hardware-aligned RAM buffers (`UnifiedBigDInstance`) without writing intermediate files to disk or polluting storage.
* **$1.8\,\mu\text{s}$ Fast-Path**: For compiled `.bigd` containers, POSIX `mmap()` maps binary payloads directly into virtual address space in microseconds.
* **Optional IR Persistence (`--save-ir <path>`)**: Users can compile and cache any raw dataset to `.bigd` on the first run for instantaneous repeated solves.

---

## 5. 20,480-Bit Dimensional Flexibility (512-Bit Cacheline Block Tiling)

To scale beyond small combinatorial bitboards up to whole-genome epistasis and ultra-dense combinatorial spaces, BIG-LL introduces a contiguous 512-bit block tiling architecture:
* **`BlockChromosome(comptime num_blocks: usize)`**: Organizes bits into contiguous $N \times 512$-bit (64-byte) blocks, each matching an exact L1D cacheline.
* **100% CPU Cache Residency**: At 20,480 bits ($40 \times 512$ bits = 2,560 bytes), an entire island population of 1,024 individuals occupies only **2.56 MB**—fitting completely within the 16–48 MB L2 / SLC cache on Apple Silicon.
* **Bitwise Block Crossover & Invariant Masking**: Single-cycle logic multiplexing operates across 512-bit cachelines with active bit masking (`cleanPadding`), guaranteeing zero schema corruption.

---

## 6. Distributed Multi-Node CPU Clustering (Dual-Transport)

BIG-LL scales across multi-node CPU clusters with lock-free atomic circular migration rings:
* **Dual Transport Architecture (`--transport <mpi|rdma|mesh|shm>`)**:
  * **Apple Silicon Thunderbolt PCIe Mesh (`mesh`)**: Zero-copy point-to-point ring buffers over 40–80 Gbps Thunderbolt interconnects for Apple Studio/Mac mini clusters, achieving sub-10 µs elite migration latencies.
  * **Linux Headless RDMA / MPI (`rdma`, `mpi`)**: High-performance InfiniBand `libibverbs` RDMA and standard MPI execution for large-scale x86_64 compute clusters.
  * **Local Multi-Process Shared Memory (`shm`)**: POSIX shared memory ring buffers for intra-host process isolation.
* **Platform Positioning**: Apple Silicon is the **Flagship Acceleration Tier** (UMA Metal GPU + Thunderbolt PCIe Mesh); Linux is the **Headless High-Core CPU Tier**.

---

## 7. Execution & Reproduction Guide

### Prerequisites
* Zig compiler `0.16.0` (or `0.16.0-dev`).
* macOS with Apple Silicon (M1/M2/M3/M4) for Metal GPU acceleration. On Linux/x86_64, the CPU engine, thread pools, and domain adapters build and run with native AVX2/NEON optimizations.

### Run All Benchmarks
```bash
# Execute the full automated verification and benchmark pipeline:
bash run_all_benchmarks.sh
```

### Individual Build Steps
```bash
# 1. Run all 93 unit and integration tests (21 test suites):
zig build test --summary all

# 2. Run memory audit (verifies strictly zero runtime heap allocations in evolutionary loops):
zig build bench-mem

# 3. Run CPU multi-core and heterogeneous island benchmarks:
zig build bench

# 4. Run dedicated Apple Silicon Metal GPU benchmark (UMA zero-copy):
zig build bench-metal

# 5. Run synthetic industrial domains benchmark (6 domains x 3 hardness tiers):
zig build bench-domains

# 6. Run real-world industrial datasets benchmark (CVRPLIB, IEEE RTS-96, S&P 500, etc.):
zig build bench-real

# 7. Run scaled multi-batch L2 cache tiling benchmark (L=256..8192):
zig build bench-scaled

# 8. Solve an optimization instance directly via CLI (ephemeral or .bigd):
./zig-out/bin/big_ll solve data/raw/airfoil/n0012.dat --gens 50 --json
./zig-out/bin/big_ll solve data/raw/logistics/A-n33-k5.vrp --save-ir /tmp/cached_An33.bigd
./zig-out/bin/big_ll solve data/real/finance/sp500_portfolio.bigd --pop 512 --gens 100 --json

# 9. Distributed Multi-Node Clustering:
./zig-out/bin/big_ll solve data/raw/logistics/A-n33-k5.vrp --cluster --transport mesh --node-id 0 --num-nodes 2
```

---

## 8. Directory Structure

```
big-ll/
├── benchmarks/
│   ├── bench_metal.zig         # Dedicated Apple Silicon Metal GPU UMA benchmarks
│   ├── bench_throughput.zig    # CPU multi-core & island scaling benchmarks
│   ├── bench_domains.zig       # Synthetic domain benchmark suite
│   ├── bench_real_domains.zig  # Real-world industrial dataset benchmarks
│   ├── bench_scaled.zig        # Scaled multi-batch L2 cache tiling benchmarks (L=256..8192)
│   └── bench_memory.zig        # Zero-allocation memory invariant verification
├── src/
│   ├── adapters/               # Gray Code, Random Key, MINLP, Boolean Set
│   ├── core/
│   │   ├── types.zig           # 64-byte aligned bitboard, BlockChromosome & DynamicPopulation
│   │   ├── engine.zig          # Dual-slab generational engine & linkage scheduler
│   │   ├── pool.zig            # Low-latency spin-wait pool with QoS affinity
│   │   ├── island.zig          # Heterogeneous asymmetric island topology
│   │   ├── linkage.zig         # 2D-tiled epistasis learning
│   │   ├── crossover.zig       # Logic multiplexing recombination
│   │   └── dedup.zig           # Two-tier L1/L2 clone deduplication
│   ├── domains/                # Finance, Logistics, Power Grid, Airfoil, Telecom, Bio
│   ├── metal/
│   │   ├── metal_bridge.h      # Clean C-ABI export for Metal compute
│   │   ├── metal_bridge.m      # Objective-C ARC implementation with MSL kernels
│   │   └── metal_engine.zig    # Zero-copy Zig wrapper for Metal compute
│   ├── data/                   # Zero-copy binary container (.bigd), mmap loader & JIT lowering
│   ├── ipc/                    # POSIX shm, distributed multi-node transport (mesh/rdma/mpi)
│   └── root.zig                # Library export root
├── tests/                      # 21 test files covering 93 test suites
├── run_all_benchmarks.sh       # Root 1-click reproduction script
└── build.zig                   # Build configuration
```
