# Component Documentation: Modular Domain Adapters (`adapters/`)

## 1. Specification

The `adapters/` package provides zero-overhead bi-directional mappings between standard domain representations and BIG-LL bitsliced chromosomes.

### 1. Reflected Gray Code Adapter (`gray_code.zig`)

Continuous domain mapping $x \in [X_{\min}, X_{\max}] \subset \mathbb{R}^d$:

* **Bijection:**
  $$g = b \oplus (b \gg 1)$$
  $$b = \bigoplus_{i=0}^5 (g \gg 2^i)$$
* **Locality Preservation:** Adjacent quantization intervals differ by exactly $\text{Hamming}(g_k, g_{k+1}) = 1$, preventing Hamming cliffs in continuous search spaces.
* **Arbitrary Word-Spanning Bitfields:** `extractBits` and `insertBits` handle bitfields of arbitrary lengths $[1, 64]$ crossing 64-bit word boundaries without truncation or byte misalignment.
* **Vectorized Decoder:** `GrayDecoder(Params)` maps $d$-dimensional continuous parameter vectors to and from compact bitstrings with zero heap allocations.

### 2. Random Key Permutation Adapter (`random_key.zig`)

Combinatorial sequencing, vehicle routing, and scheduling:

* **Representation:** $M$ items represented by $M$ independent fixed-point keys $K_i \in [0, 2^{\text{KeyBits}} - 1]$.
* **Bijection Guarantee:** $\pi = \text{argsort}(\mathbf{K})$ yields a guaranteed valid permutation $\pi \in S_M$ without duplicate items, cycle omissions, or heuristic repair operators.
* **In-Register Sorting:** Uses `std.sort.block` over contiguous stack buffers.
* **Linkage Preservation:** Epistatic dependencies between item positions are preserved and learned directly in bitsliced co-occurrence space.

### 3. MINLP Hybrid Adapter & Dynamic Augmented Lagrangian (`minlp.zig`)

Mixed-Integer Non-Linear Programming:

* **Dual-Strand Architecture:**
  * Discrete Strand: $N_s$ binary switches or discrete state indicators mapped to bits $0 \dots N_s - 1$.
  * Continuous Strand: $N_c$ scalars decoded via Reflected Gray code starting at bit offset $N_s$.
* **Co-Evolution:** Epistatic linkage between discrete switches (e.g., turbine on/off) and continuous controls (e.g., fuel flow rate) is captured in a unified $L$-bit linkage graph.
* **Dynamic Augmented Lagrangian (`AugmentedLagrangianConstraint`):**
  * Tracks population constraint violation ratio $\rho_{\text{infeas}}$.
  * Dynamically scales quadratic penalty multiplier $\mu$:
    $$\mu_{t+1} = \begin{cases} \min(\mu_t \times 1.2, \mu_{\max}) & \text{if } \rho_{\text{infeas}} > 0.05 \\ \max(\mu_t \times 0.9, \mu_{\min}) & \text{if } \rho_{\text{infeas}} \le 0.01 \end{cases}$$
  * Prevents search collapse into illegal domains while avoiding exploration stalling at active boundaries.

### 4. Boolean Set Combinatorial Adapter (`boolean_set.zig`)

Feature selection, epistatic biomarker panels, and knapsack:

* **Representation:** Bitmask over universe size $N$. Bit $i = 1$ denotes selection of element $i$.
* **Fast Index Extraction:** $O(k)$ bit extraction via `w &= w - 1` and `@ctz` (Count Trailing Zeros), executing at hardware instruction speed.
* **Cardinality Invariants:** Penalizes excess selection over target cardinality constraint $K_{\max}$.

### 5. In-Register Permutation Decoder (`RandomKeyDecoder.decodeFast`)

* For $M \le 64$, replaces Quicksort block sorting with an in-register counting ranking algorithm:
  $$\text{rank}_i = \sum_{j=0}^{M-1} \mathbb{I}(K_j < K_i \lor (K_j = K_i \land j < i))$$
* Operates strictly in L1 registers with zero heap allocations, zero sorting stack frames, and deterministic branch execution.
