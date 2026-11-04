# Component Documentation: PRNG Engine (`prng.zig`)

## 1. Specification

The `prng.zig` module implements an in-register Xoshiro256++ (xor/shift/rotate) pseudo-random number generator optimized for SIMD bitmask synthesis.

### Algorithm: Xoshiro256++

* **State Space:** 256 bits (`s: [4]u64`).
* **Period:** $2^{256} - 1$.
* **Seeding:** Seeded via SplitMix64 generator to guarantee maximum initial entropy across all 4 state words.

### Public Interface

* `init(seed: u64) Self`: Initializes state words via SplitMix64 state expansion.
* `next() u64`: Produces 64 pseudorandom bits.
  $$\text{result} = \text{rotl}(s_0 + s_3, 23) + s_0$$
* `nextFloat() f64`: Generates standard uniform float $\in [0, 1.0)$ with 53 bits of precision ($(\text{next}() \gg 11) \times 2^{-53}$).
* `uintLessThan(bound: usize) usize`: Bounded integer generation using modulo reduction.
* `fillWords(words: []u64) void`: Fills memory slice with 64-bit random words for bulk crossover masks.
* `sparseMask(depth: u3) u64`: Generates sparse mutation bitmasks via logical AND intersections:
  ```text
  Mask = AND_{k=1..depth} next()
  ```
  * Density at `depth = 1`: 50.0% (Uniform Crossover).
  * Density at `depth = 4`: 6.25%.
  * Density at `depth = 6`: 1.5625% (~1/64 bits set; average 1 bit flip per 64-bit word).
* `fillSparseWords(words: []u64, depth: u3) void`: In-place generation of mutation vectors across arbitrary word counts.
