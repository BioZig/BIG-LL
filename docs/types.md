# Component Documentation: Core Types (`types.zig`)

## 1. Specification

The `types.zig` module provides hardware-aligned memory primitives for genetic algorithm bitstrings and population containers.

### Chromosome(comptime L: usize)

Fixed-length binary bitstring representation bounded by compile-time dimension `L`.

* **Memory Layout:**
  * Word Count: `W = ceil(L / 64)`.
  * Alignment: Strict 64-byte alignment (`align(64)`) to match modern x86/ARM CPU cache lines.
  * Padding: Bits beyond `L - 1` in the final word `W - 1` are zeroed via `cleanPadding()`.

* **Fields:**
  * `words: [W]u64 align(64)`: Contiguous bit-packed payload.
  * `fitness: f64`: Scalar objective evaluation score.
  * `hash: u64`: Precomputed Murmur3/BLAKE3 64-bit checksum for evaluation cache interception.

* **Operations:**
  * `getBit(idx: usize) u1`: Extracts bit at index $idx$. Complexity: $O(1)$ (1 shift, 1 mask).
  * `setBit(idx: usize, val: u1) void`: In-place mutation of bit at index $idx$. Complexity: $O(1)$.
  * `toggleBit(idx: usize) void`: In-place bitwise inversion ($x \oplus 1$) at index $idx$. Complexity: $O(1)$.
  * `popCount() usize`: Total active bits across all words using hardware `@popCount`. Complexity: $O(W)$.
  * `hammingDistance(other: *const Self) usize`: Exact bitwise distance $\sum_{i=0}^{W-1} \text{popCount}(w_i \oplus w_i')$. Complexity: $O(W)$.
  * `bitwiseEqual(other: *const Self) bool`: Word-for-word equality check. Complexity: $O(W)$.

### Population(comptime L: usize)

Flat contiguous memory container for $N$ individuals.

* **Fields:**
  * `individuals: []Chromosome(L)`: Contiguous memory buffer.
  * `best_idx: usize`: Index of maximal fitness individual.
  * `best_fitness: f64`: Highest fitness value recorded in current generation.
  * `mean_fitness: f64`: Arithmetic mean of population fitness.
  * `entropy: f64`: Normalized bitwise Shannon entropy:
    $$H = \frac{1}{L} \sum_{i=0}^{L-1} \left[ -p_i \log_2(p_i) - (1-p_i)\log_2(1-p_i) \right]$$
    where $p_i = \frac{1}{N} \sum_{j=0}^{N-1} \text{indiv}_j[i]$.

* **Lifecycle:**
  * `init(allocator, size)`: Allocates one contiguous slice of $N \times \text{sizeof(Chromosome(L))}$.
  * `deinit(allocator)`: Frees contiguous slice. Zero intermediate heap operations.

---

### DynamicChromosome

Non-allocating view over an arbitrary runtime slice of 64-bit words, supporting arbitrary bit lengths $L$.

* **Fields:**
  * `words: []u64`: Natural 8-byte aligned slice into population bitboard storage.
  * `bit_length: usize`: Exact active bit count $L$.
  * `fitness: f64`: Objective evaluation score.
  * `hash: u64`: Precomputed checksum.

* **Operations:**
  * `getBit(idx)`, `setBit(idx, val)`, `toggleBit(idx)`: $O(1)$ bit manipulation.
  * `popCount()`: Vectorizable hardware `@popCount` over slice words.
  * `hammingDistance(other)`, `bitwiseEqual(other)`: Word-for-word slice operations.
  * `cleanPadding()`: Masks off trailing inactive bits beyond $L - 1$ in the final word.

---

### DynamicPopulation (Pitch-Stratified Arena)

Structure-of-Arrays (SoA) contiguous memory container eliminating cache line waste on dynamic problem scales.

* **Memory Architecture:**
  * Single 64-byte aligned bitboard: `bitboard: []align(64) u64` of size $N \times \text{pitch}$.
  * Pitch: $\text{pitch} = \lceil L / 64 \rceil$ words per individual.
  * Multi-Individual Cache Line Packing: For $L \le 64$ ($\text{pitch} = 1$), a single 64-byte cache line holds **8 entire individuals**, guaranteeing 100% L1 Data Cache residency and zero wasted padding.
  * SoA Separation: `fitnesses: []f64` and `hashes: []u64` are stored in separate contiguous arrays, preventing cache pollution during selection and metric scans.

* **Methods:**
  * `init(allocator, size, bit_len)`: Pre-allocates aligned bitboard and SoA arrays.
  * `deinit(allocator)`: Releases buffers.
  * `get(idx)`: Returns a `DynamicChromosome` view over individual `idx` in $O(1)$ with zero heap allocation.
  * `getWords(idx)`, `getWordsMut(idx)`: Direct strided slice access.
  * `updateMetrics()`: $O(N)$ arithmetic mean and maximum fitness extraction.
  * `computeShannonEntropy()`: Parallelized bitwise population Shannon entropy across arbitrary $L$.
