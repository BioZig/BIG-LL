# Component Documentation: Simulation IPC and C-ABI Driver (`shm.zig`, `c_api.zig`)

## 1. Specification

The `ipc` module provides zero-copy inter-process communication (IPC) and a standard C-ABI export interface for coupling BIG-LL with external simulation codes (e.g., C, C++, Fortran, Python, and standalone physics/biological simulation binaries).

### Architectural Invariants

1. **Zero-Copy Evaluation Buffer:**
   * Chromosomes and fitness values are mapped in a single contiguous memory region or passed directly via pointers (`[*]const u64`).
   * No serialization, string parsing, or JSON marshaling occurs during fitness evaluation.

2. **64-Byte Cacheline Alignment:**
   * Header structures and data segment offsets are aligned to 64-byte boundaries (`align(64)`).
   * Contiguous chromosome bit words begin at `genes_offset` ($128$ bytes from base).
   * Fitness array begins at `fitness_offset` aligned to a multiple of 64 bytes.

3. **Lock-Free Atomic Status Signaling:**
   * External simulation workers synchronize with BIG-LL using atomic status flags:
     * `STATUS_IDLE (0)`: No evaluation pending.
     * `STATUS_EVAL_REQUEST (1)`: Published with release semantics by BIG-LL when population data is ready.
     * `STATUS_EVAL_COMPLETE (2)`: Published with release semantics by the simulator when all fitness values are written.
     * `STATUS_TERMINATED (3)`: Engine shutdown signal.
   * Eliminates system call latency or kernel semaphore contention in hot simulation loops.

4. **Hermetic Memory Mapping:**
   * Supports both anonymous memory allocation (`initAnonymous`) and file-backed POSIX shared memory (`createFileBacked`, `openFileBacked`) via `std.posix.mmap`.
   * Operates without requiring an external C runtime library dependency.

---

## 2. Shared Memory Layout

```text
+---------------------------+---------------------------------------------------+
| Byte Offset Range         | Segment Description                               |
+---------------------------+---------------------------------------------------+
| 0 .. 127                  | SharedPopulationHeader (Magic 0x4249474C, Ver 1)  |
| 128 .. fitness_offset - 1 | Chromosome Genes Data (N * W * 8 bytes)           |
| fitness_offset .. end     | Fitness Array (N * 8 bytes, f64)                  |
+---------------------------+---------------------------------------------------+
```

where:
* $N = \text{pop\_size}$
* $L = \text{chromosome\_bits}$
* $W = \lceil L / 64 \rceil$

---

## 3. C-ABI Interface (`include/big_ll.h`)

### Function Signatures

```c
// Creation & destruction
big_ll_handle_t big_ll_create(
    size_t chromosome_bits,
    size_t pop_size,
    uint64_t seed,
    big_ll_eval_fn eval_fn,
    void* user_data
);
big_ll_handle_t big_ll_create_custom(
    size_t chromosome_bits,
    size_t pop_size,
    uint64_t seed,
    big_ll_eval_fn eval_fn,
    void* user_data,
    big_ll_options_t options
);
void big_ll_destroy(big_ll_handle_t handle);

// Generational advancement
int big_ll_step(big_ll_handle_t handle);
int big_ll_run(big_ll_handle_t handle, size_t generations);

// Zero-copy state inspection
double big_ll_get_best_fitness(big_ll_handle_t handle);
size_t big_ll_get_best_genes(big_ll_handle_t handle, uint64_t* out_buffer, size_t buffer_words);
const uint64_t* big_ll_get_best_genes_ptr(big_ll_handle_t handle);
const uint64_t* big_ll_get_chromosome_words(big_ll_handle_t handle, size_t index);
double big_ll_get_chromosome_fitness(big_ll_handle_t handle, size_t index);
```

---

## 4. Distributed Multi-Node Clustering Protocol (`distributed.zig`)

BIG-LL provides a zero-allocation distributed transport protocol for multi-node CPU clustering:

### Wire Protocol: `FrameHeader` (128 Bytes, Cacheline Aligned)
The network wire protocol uses a fixed 128-byte header aligned to 64 bytes (`align(64)`):
* `magic: u32 = 0x4249474E` ("BIGN")
* `version: u32 = 1`
* `source_node: u32` / `target_node: u32`
* `generation: u64`
* `elite_count: u32`
* `chrom_bits: u32`
* `payload_bytes: u64`
* `checksum: u32` (CRC32 over payload)
* `reserved: [10]u64`

### Transports:
1. **Apple Silicon Thunderbolt PCIe Mesh (`mesh`)**: Zero-copy memory-mapped circular mailboxes across Thunderbolt 4/5 interconnects (40–80 Gbps) connecting Mac Studios and Mac minis with sub-10 µs elite migration latencies.
2. **InfiniBand RDMA (`rdma`)**: Low-latency, zero-copy remote direct memory access for Linux enterprise HPC clusters.
3. **Message Passing Interface (`mpi`)**: Standard asynchronous `MPI_Isend` / `MPI_Irecv` ring migration for distributed supercomputing partitions.
4. **Shared Memory (`shm`)**: Lock-free atomic circular mailboxes (`NetworkMailbox`) for multi-process clustering on a single workstation.

### Lock-Free Migration Invariant:
Ring buffers use atomic head and tail pointers (`std.atomic.Value(u32)`) with acquire/release semantics. Immigrants are polled non-blockingly during generational migration checkpoints without stalling island worker threads.
