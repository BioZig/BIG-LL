# Component Documentation: L1 Deduplication Cache (`dedup.zig`)

## 1. Specification

The `dedup.zig` module provides an $O(1)$ direct-mapped evaluation ring cache designed to filter redundant chromosome evaluations within the L1 CPU data cache.

### Algorithmic Pipeline

1. **Power-of-Two Ring Addressing:**
   * Cache capacity $N = 2^k$ (default: 8,192 slots).
   * Slot indexing via single bitwise AND:
     $$\text{slot\_index} = \text{hash} \ \& \ (N - 1)$$

2. **Murmur-Inspired 64-Bit Avalanche Hash:**
   * Ingests 64-bit aligned chromosome word slice $\mathbf{w} \in \mathbb{N}^{W}$.
   * Initial seed: `0x517cc1b727220a95`.
   * Word mixing:
     $$h \leftarrow (h \oplus w_i) \lll 31 \times \mu_1$$
   * Final avalanche mixing stages:
     $$h \leftarrow h \oplus (h \gg 33) \times \mu_2 \dots$$
   * Ensures $> 16$-bit flip divergence for single-bit chromosome alterations.

3. **Direct-Mapped Slot Overwrite:**
   * Slot layout: `struct { hash: u64, fitness: f64, valid: bool }`.
   * Lookups compare `slot.valid and slot.hash == hash`.
   * Hit returns cached `f64` fitness without invoking domain decoding or objective simulation.
   * Miss records cache miss and allows caller to compute and insert new fitness score.

4. **Telemetry:**
   * Tracks discrete hit and miss counters.
   * `hitRatio()` returns $\frac{\text{hits}}{\text{hits} + \text{misses}}$.

### Memory Footprint

* `L1Dedup`: 8,192 slots $\times$ 24 bytes $\approx$ 192 KB.
* `AdaptiveDedup`: 2-way associative sets $\times$ 32 bytes per slot + 1024-bit compact Bloom filter (128 bytes).
* Zero dynamic allocations during lookup or insertion.

## 2. Adaptive Set-Associative Deduplication (`AdaptiveDedup`)

The `AdaptiveDedup(NumSets, Assoc)` component prevents Birthday Paradox eviction thrashing and eliminates wasted hash cycles during exploration:

1. **Entropy-Gated Bypass**:
   * Evaluates population Shannon entropy $H(t)$.
   * When $H(t) > 0.35$ (exploration phase), all hashing and table lookups are bypassed ($0$ cycle penalty).
   * When $H(t) \le 0.35$ (convergence phase), table search and caching activate automatically.

2. **2-Way / 4-Way Set Associativity with LRU Eviction**:
   * Each set contains `Assoc` slots.
   * Colliding distinct hashes coexist in parallel ways rather than overwriting each other immediately.
   * Eviction uses an epoch-based LRU counter `age`.

3. **1024-Bit Compact Bloom Filter**:
   * Uses independent dual hash slices: $b_1 = h \bmod 1024$, $b_2 = (h \gg 32) \bmod 1024$.
   * Filters out novel chromosomes in a single bitwise check before probing the associative set.
