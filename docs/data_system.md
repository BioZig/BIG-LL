# Zero-Copy Binary Dataset System (`.bigd`)

## 1. Architectural Overview & Design Rationale

In high-throughput heuristic and genetic optimization pipelines, parsing formatted text files (e.g., CSV, JSON, TSV) at initialization introduces significant latency, uncontrolled heap fragmentation, and non-deterministic cache footprints.

The BIG-LL Zero-Copy Binary Dataset System (`.bigd`) standardizes instance representations across all 6 industrial domains via a memory-mapped, binary container specification. Key architectural properties include:

* **Zero Runtime Heap Allocation:** All domain instances are loaded either via POSIX memory mapping (`mmap`) or static memory buffers. No dynamic allocation occurs during the generational evolutionary loop.
* **Hardware Cacheline Alignment:** The file header (`BigDHeader`) is aligned to 64 bytes and sized to exactly 128 bytes (occupying two L1 cache lines). All data sections (Section A, Section B, Section C) are aligned to 64-byte boundaries, enabling direct vectorization via ARM64 NEON and x86_64 AVX-512.
* **Precomputed Distance & Metric Matrices:** Expensive geometric or algebraic operations (e.g., $\sqrt{\Delta x^2 + \Delta y^2}$ in CVRP routing) are precomputed into contiguous row-major matrices stored in the binary payload, reducing inner-loop domain evaluation to $O(1)$ flat memory lookups.
* **Payload Integrity Verification:** Every binary container includes a Fletcher-32 checksum covering the entire payload beyond the header.

---

## 2. Container Specification: `BigDHeader`

The container begins with the 128-byte `BigDHeader` struct declared with `align(64)`:

```zig
pub const BigDHeader = extern struct {
    magic: u32 align(64) = 0x42494744, // "BIGD"
    version: u32 = 1,
    domain_tag: u32,
    flags: u32 = 0,
    total_bytes: u64,
    checksum: u32 = 0,
    param_u32_4: u32 = 0,
    param_u32_0: u32 = 0,
    param_u32_1: u32 = 0,
    param_u32_2: u32 = 0,
    param_u32_3: u32 = 0,
    param_f64_0: f64 = 0.0,
    param_f64_1: f64 = 0.0,
    param_f64_2: f64 = 0.0,
    param_f64_3: f64 = 0.0,
    offset_section_a: u64 = 128,
    offset_section_b: u64 = 0,
    offset_section_c: u64 = 0,
    reserved: [3]u64 = [_]u64{0} ** 3,
};
```

### Invariants:
1. `@sizeOf(BigDHeader) == 128` bytes.
2. `@alignOf(BigDHeader) == 64` bytes.
3. `magic == 0x42494744`.
4. `offset_section_a >= 128` and `offset_section_a % 64 == 0`.
5. `offset_section_b == 0` or (`offset_section_b >= offset_section_a` and `offset_section_b % 64 == 0`).
6. `offset_section_c == 0` or (`offset_section_c >= offset_section_b` and `offset_section_c % 64 == 0`).
7. `total_bytes` matches exact file/buffer length.
8. Fletcher-32 checksum matches data payload `buffer[128..total_bytes]`.

---

## 3. Supported Domains & Binary Payload Layouts

The `domain_tag` field identifies the problem domain:

| Domain | Tag Value | Section A Payload | Section B Payload | Section C Payload | Scalar Metadata |
| :--- | :---: | :--- | :--- | :--- | :--- |
| **Logistics (CVRP)** | 0 | `[M]Point` (Customer $(x, y)$) | `[M]f64` (Demands) | `[(M+1)*(M+1)]f64` (Distance Matrix) | `u32_0`: $M$, `f64_0`: Capacity, `f64_1..2`: Depot $(x,y)$ |
| **Power Grid (UC)** | 1 | `[N]GeneratorSpec` ($P_{\min}, P_{\max}, a, b, c$) | Empty (0) | Empty (0) | `u32_0`: $N$, `f64_0`: Demand, `f64_1`: Reserve |
| **Finance (Markowitz)** | 2 | `[N]f64` (Expected Returns) | `[N*N]f64` (Flat Covariance) | Empty (0) | `u32_0`: $N$ Assets, `u32_1`: $R$ Rules, `f64_0`: Risk-Free Rate |
| **Bioinformatics (GWAS)**| 3 | `[N]f64` (SNP Main Effects) | `[P]EpistasisPair` (Pair Epistasis) | `[T]EpistasisTriplet` (Triplet Epistasis) | `u32_0`: $N$, `u32_1`: TargetCard, `u32_2`: MaxCard, `u32_3`: $P$, `u32_4`: $T$, `f64_0`: Penalty W |
| **Airfoil (Aerodynamics)**| 4 | Empty (0) | Empty (0) | Empty (0) | `u32_0`: $N$ Points, `f64_0`: $C_L$ Factor, `f64_1`: Base $C_D$ |
| **Telecom (MIMO SINR)** | 5 | `[U*U]f64` (Flat Channel Gain Matrix)| Empty (0) | Empty (0) | `u32_0`: $U$ Users, `u32_1`: $C$ Channels, `f64_0`: Noise Floor |

---

## 4. Loader Architecture: Zero-Copy Ingestion

The `data_loader` module provides two distinct ingestion modes:

### 1. `MemoryMappedInstance`
Utilizes kernel-level virtual memory mapping via POSIX `openat`, `lseek`, and `mmap`:
```zig
var mmap_inst = try loader.MemoryMappedInstance.openFile("data/logistics_32.bigd");
defer mmap_inst.deinit();

const domain = try big_ll.LogisticsDomain.initFromInstance(&mmap_inst);
```
Pages are paged into memory on-demand by the OS kernel without user-space buffer allocation or data copying. When the instance is destroyed, `munmap` and `close` release the virtual address mapping.

### 2. `StaticBufferInstance`
Enables direct zero-copy wrapping of stack-allocated, static, or IPC shared-memory slices:
```zig
var inst = try loader.StaticBufferInstance.init(raw_buffer_aligned_64);
const domain = try big_ll.PowerGridDomain.initFromInstance(&inst);
```

Both loaders provide type-safe slice extraction via:
```zig
pub fn getSlice(self: *const Self, comptime T: type, offset: usize, count: usize) LoaderError![]const T
```
which guarantees strict pointer alignment, size boundary containment, and memory safety without allocations.

---

## 5. Synthetic Instance Generator

The `data_generator` module provides deterministic synthetic dataset generators for benchmarking and falsification testing:

* `generateLogistics(allocator, num_customers, capacity, depot_x, depot_y)`
* `generatePowerGrid(allocator, demand, reserve_target)`
* `generateFinance(allocator, num_assets, num_rules, risk_free_rate)`
* `generateBioinformatics(allocator, num_snps, target_card, max_card)`
* `generateAirfoil(allocator, num_points, cl_factor, base_cd)`
* `generateTelecom(allocator, num_users, num_channels, noise_power)`
* `saveToFile(path, buffer)`

---

## 6. Verification & Falsification Guarantees

The binary dataset system is verified by two test suites in the continuous integration pipeline:

1. `tests/test_data_loader.zig`:
   * Header size ($128\text{ B}$) and cache alignment ($64\text{ B}$) invariants.
   * Rejection of corrupted magic tokens (`0x12345678`), truncated buffers, and unaligned section offsets.
   * Fletcher-32 single-bit corruption detection.
   * File-backed `MemoryMappedInstance` roundtrip tests.

2. `tests/test_synthetic_domains.zig`:
   * End-to-end integration across all 6 domains using `GeneticEngine`.
   * Monotonic or non-decreasing generational fitness convergence.
   * Absolute verification of physical and structural domain invariants:
     * Logistics: Valid permutation with each customer visited exactly once.
     * Power Grid: Generation limits respected for all active units.
     * Finance: Strictly non-negative weights summing to 1.0.
     * Bioinformatics: Extracted biomarker panel strictly within cardinality bounds.
     * Airfoil: Physical thickness non-negative ($y_{\text{upper}} \ge y_{\text{lower}}$).
     * Telecom: Valid channel assignments and power bounds.

---

## 7. Multi-Format Ephemeral Ingestion & Zero-Disk JIT Lowering

To decouple solver mechanics from raw domain formats without compromising cache alignment or introducing disk clutter, BIG-LL implements a unified JIT lowering pipeline:

```
                            big-ll solve <input_file>
                                       │
                          Inspect First 16 Bytes / Magic
                                       │
             ┌─────────────────────────┼─────────────────────────┐
             ▼                         ▼                         ▼
     Magic == 0x42494744      Magic == 0x6C 0x1B 0x01      Magic == "\x93NUMPY"
         [ .bigd ]                 [ PLINK .bed ]              [ NumPy .npy ]
             │                         │                         │
       POSIX mmap()              JIT Lowering               JIT Lowering
    (Fast-path: ~1.8 µs)    (Stream into RAM buffer)   (Header parse + RAM buffer)
             │                         │                         │
             └─────────────────────────┼─────────────────────────┘
                                       ▼
                            Unified BigDInstance
                       (raw_slice: []align(64) const u8)
                                       │
                                       ▼
                        Zero-Allocation Solver Engine
                        (4-Island CPU / Metal GPU UMA)
```

### Architectural Properties:
1. **`UnifiedBigDInstance`**: Polymorphic container representing either a POSIX `mmap` backing slice or an ephemeral anonymous RAM buffer (`ephemeral_ram`).
2. **Zero Disk Footprint**: Raw files (`.bed`, `.npy`, `.vrp`, `.csv`, `.tsv`, `.dat`) are compiled into a 64-byte aligned in-memory `.bigd` buffer in RAM during initialization. No files are written to `/tmp` or disk.
3. **Preservation of the 0 B Runtime Heap Invariant**: JIT translation happens strictly during initialization. Once the evolutionary loop begins, the engine operates on the RAM buffer with zero heap allocations.
4. **`--save-ir <path>` Support**: Enables optional persistence of compiled in-memory IR to disk for recurring microsecond-level fast-path runs.
