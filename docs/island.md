# Component Documentation: Island Topology Engine (`island.zig`)

## 1. Specification

The `island.zig` module implements an island-parallel evolutionary framework `IslandEngine(L, Adapter, num_islands, migration_interval)` to scale genetic optimization over multiple CPU cores using thread-local cache execution and periodic ring migration.

### Algorithmic Pipeline

1. **Sub-Population Partitioning:**
   * Given total population size $N$ and island count $M$, each island instantiates an independent `GeneticEngineCustom(L, Adapter, .{ .enable_dedup = false, .num_threads = 1 })` controlling sub-population $N_{\text{sub}} = \lfloor N / M \rfloor$.
   * Each sub-population resides entirely within the private L1/L2 cache blocks of its assigned core.

2. **Thread-Local Generational Execution:**
   * During an epoch of length `E = migration_interval` generations, each island thread iterates uninterrupted:
     ```text
     for t in [0, E):
         island[m].step()
     ```
   * Zero shared memory accesses, zero lock contention, and zero atomic operations occur in the generational loop.

3. **Synchronous Ring Migration:**
   * At epoch completion (`G % E == 0`), islands synchronize and export their top `C` elites into a pre-allocated migration buffer:
     ```text
     B[m] <- top C chromosomes of island m
     ```
   * Unidirectional ring exchange routes elites:
     ```text
     island[m] receives B[(m + M - 1) % M]
     ```
   * Incoming elites replace the lowest-fitness individuals in `pop_current` if the incoming fitness strictly exceeds the local minimum.
   * Island metrics are recalculated and the next epoch initiates.

### Invariants

* **Zero Runtime Heap Allocation:** All migration buffers and thread contexts are statically sized at comptime.
* **Strict Elitism:** Global best fitness is monotonic non-decreasing across all epochs.
* **Independent PRNG Streams:** Each island PRNG is initialized with a distinct Weyl sequence seed offset.
