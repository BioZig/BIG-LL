# Component Documentation: Linkage Engine (`linkage.zig`)

## 1. Specification

The `linkage.zig` module implements the Bitsliced Information-Geometric Linkage Learning (B-BIG-LL) engine with L2 cache tiling.

### Algorithmic Pipeline

1. **Dimensionality & Batch Sizing:**
   * $B = \min(L, 2048)$ (limits working dense tile to $\le 512\text{ KB}$ for L2 cache pinning).
   * $X = \lceil L / B \rceil$ (number of sequential tiles).
   * $K \in \{64, 128, 256\}$ (register-scaled elite sample size).

2. **Extreme-Value Statistical Threshold:**
   $$\tau(L, K) = \sqrt{\frac{2 \ln(L)}{K}}, \quad \Delta_{\text{thresh}} = \max\left(4, \text{round}\left(\tau \times \frac{K}{2}\right)\right)$$

3. **In-Register Elite Transposition:**
   $K$ elite chromosomes are transposed into $L$ register vectors $\mathbf{v}_i \in \{0, 1\}^K$.
   Complexity: $O(K \times L / 64)$.

4. **Batch Co-occurrence Evaluation & Scaled Integer Pipeline:**
   For each batch tile $b \in [0, X-1]$:
   * Evaluate joint occurrences:
     ```text
     C_ij = sum_{w=0}^{W_K-1} @popCount(v_i[w] & v_j[w])
     ```
   * **Pure Scaled Integer Linkage:**
     To eliminate multi-cycle floating-point divisions (`fdiv`) and integer-to-float conversions in the $O(L^2)$ locus loop, the comparison is multiplied by $K$:
     $$\Delta_{\text{scaled}} = \left| K \cdot C_{ij} - \text{pop}(\mathbf{v}_i) \cdot \text{pop}(\mathbf{v}_j) \right|$$
   * Threshold test:
     $$\Delta_{\text{scaled}} \ge T_{\text{scaled}}, \quad \text{where } T_{\text{scaled}} = K \cdot \Delta_{\text{thresh}}$$
   * Replaces 14-cycle scalar float division with 1-cycle integer multiply-subtract (`mul`/`sub`), accelerating the $O(L^2)$ matrix computation by up to $10\times$.
   * **Register Hoisting:** The outer locus vector $\mathbf{v}_i$ is hoisted into CPU registers once per outer iteration, avoiding redundant memory loads across all $j \in [i+1, count-1]$.

5. **2D-Tiled Cross-Batch Scheduling:**
   * In addition to diagonal tiles $(B_b, B_b)$, `computeFullLinkageScheduled` evaluates off-diagonal cross-tile blocks $(B_a, B_b)$ via `computeCrossBatchLinkage`.
   * For $X \le 4$, all off-diagonal blocks are computed systematically every step.
   * For $X > 4$, a deterministic stride-rotation scheduler evaluates interleaved cross-tile pairs:
     $$b = (a + 1 + (\text{epoch} \bmod (X - 1))) \bmod X$$
   * In-place Fisher-Yates permutation reshuffling (`shufflePermutation`) runs every 5 generations to rotate gene loci across tile windows.

### Memory Complexity

* Ingested Elites: $L \times (K/64) \times 8\text{ bytes}$.
* Dense Tile Buffer: $B \times \lceil B / 64 \rceil \times 8\text{ bytes} \le 512\text{ KB}$ (fixed reusable buffer).
* Sparse Adjacency List: $2 \times |E| \times 4\text{ bytes}$ (scales with true coupled edge density).
