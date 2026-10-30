# Component Documentation: Generational Genetic Engine (`engine.zig`)

## 1. Specification

The `engine.zig` module implements the top-level evolutionary orchestration pipeline `GeneticEngine(L, Adapter)` utilizing zero-allocation slab memory and entropy-governed linkage scheduling.

### Algorithmic Pipeline

1. **Dual-Slab Flat Memory Arena:**
   * Allocates two flat population arrays (`pop_current`, `pop_next`) during `init`.
   * Generational step executes off `pop_current` into `pop_next`.
   * At the end of the generational cycle, slice pointers are swapped in $O(1)$ time without memory reallocation or copy.

2. **Elite Extraction:**
   * Extracts top $K \in \{64, 128, 256\}$ chromosomes using single-pass min-element replacement.
   * Transferred directly into `linkage_engine.ingestElites()`.
   * Exact copies placed into `pop_next[0..K]` for strict elitism guarantee ($f_{\text{best}}^{(t+1)} \ge f_{\text{best}}^{(t)}$).

3. **Linkage Graph Refresh & Permutation Shuffling:**
   * At epoch modulo 5 ($\text{gen} \equiv 0 \pmod 5$), triggers Fisher-Yates permutation shuffle across tiles.
   * Executes `linkage_engine.computeFullLinkage(allocator)` to construct the sparse linkage graph.

4. **Entropy-Triggered Adaptive Mutation:**
   * Inspects `pop_current.entropy` against threshold `entropy_floor` (default 0.15).
   * Dynamically alters mutation bit-depth:
     $$\text{depth} = \begin{cases} 4 & \text{if } H(t) < H_{\text{floor}} \ (\text{elevated diversity injection: } (0.5)^4 = 6.25\% \text{ bit-flip}) \\ 5 & \text{if } H(t) \ge H_{\text{floor}} \ (\text{standard sparse perturbation: } (0.5)^5 = 3.125\% \text{ bit-flip}) \end{cases}$$

5. **Offspring Production Loop:**
   * Tournament selection of size 2 ($k=2$) identifies parents $p_1, p_2$.
   * `CrossoverEngine.generateMask` walks the linkage graph from a random seed locus.
   * Logic multiplexing recombines parents into $c_1, c_2$.
   * Sparse mutation is applied to both offspring.
   * Evaluates offspring through `dedup_cache`; evaluated individuals are placed into `pop_next`.

### Invariants

* Zero heap allocations inside `step()`.
* Strict elitism: best fitness monotonic non-decreasing.
* Thread-safe independent PRNG state per engine instance.

---

### DynamicGeneticEngine(Adapter)

A runtime dynamic-length evolutionary engine operating on [`DynamicPopulation`](types.md#dynamicpopulation-pitch-stratified-arena) and [`DynamicLinkageEngine`](linkage.md).

* **Key Advancements:**
  1. **Dynamic $L$ Support:** Configurable bit-length at initialization (`init(allocator, bit_len, pop_size, ...)`).
  2. **Fused Streaming Recombination:** Utilizes `recombineAndMutateFused` to eliminate RAM round-trips for mutation masks.
  3. **Zero-Allocation Generational Loop:** All working buffers (`elite_bitboard`, `crossover_mask`, `pop_current`, `pop_next`) are pinned once during startup. Slice pointers are swapped at generation end in $O(1)$ time.
  4. **Branchless Tournament Selection:** Lowers to hardware conditional select (`csel` on ARM64, `cmov` on x86), eliminating branch misprediction penalties on fitness comparisons.
