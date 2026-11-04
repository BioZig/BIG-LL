# Component Documentation: Crossover Engine (`crossover.zig`)

## 1. Specification

The `crossover.zig` module provides linkage graph-cut recombination and bitwise mutation operators.

### Operators

1. **`generateMask(engine, rng, out_mask)`:**
   * Selects random seed locus `s` in `[0, L-1]` via uniform PRNG.
   * Traverses adjacent edges in the learned sparse linkage graph.
   * **Cluster Growth Governor:** Clamps cluster expansion at `S_max = floor(0.35 * L)`.
   * Prevents percolation collapse in continuous geometries, preserving open recombination.

2. **`recombine(p1, p2, mask)`:**
   * Generates two offspring via 128-bit SIMD vector multiplexing (`@Vector(2, u64)`):
     ```text
     c1 = (p1 & m) | (p2 & ~m)
     c2 = (p2 & m) | (p1 & ~m)
     ```
   * Maps directly to the ARM64 NEON **`bsl`** (Bitwise Select) instruction and x86 AVX-512 **`vpternlog`**, retiring recombination in 1 cycle per vector.
   * Cleans unused word padding.
   * Complexity: `O(L / 128)` vector cycles.

3. **`recombineAndMutateFused(p1, p2, mask, rng, depth, c1, c2, bit_length)`:**
   * **In-Register Fused Recombine-Mutate Kernel:**
     Combines the `bsl` vector recombination with in-register sparse random mutation:
     ```text
     c1 = ((p1 & m) | (p2 & ~m)) ^ r1
     c2 = ((p2 & m) | (p1 & ~m)) ^ r2
     ```
   * Eliminates the intermediate `mutation_mask` buffer round-trip to L1 cache, halving memory bandwidth.
   * Offspring are streamed directly into the destination population buffer in a single pass.

4. **`mutate(chrom, mut_mask)`:**
   * In-place bit-flip mutation via XOR:
     ```text
     c' = c ^ r
     ```
   * Complexity: `O(L / 64)` machine cycles.
