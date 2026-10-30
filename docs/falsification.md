# Component Documentation: Falsification Suite (`tests/test_falsification.zig`)

## 1. Specification

The `test_falsification.zig` suite provides an empirical benchmark designed to test and falsify the building-block hypothesis and linkage discovery capabilities of BIG-LL against deceptive epistatic fitness landscapes.

### The Deceptive Trap Function

1. **Mathematical Definition:**
   For a block of $k = 5$ bits with unitation $u = \sum_{i=0}^4 x_i$:
   $$\text{Trap}(u) = \begin{cases} 5.0 & \text{if } u = 5 \\ 4.0 - u & \text{if } u < 5 \end{cases}$$

2. **Deceptive Gradient:**
   * $u = 0 \implies 4.0$ (Local deceptive attractor)
   * $u = 1 \implies 3.0$
   * $u = 2 \implies 2.0$
   * $u = 3 \implies 1.0$
   * $u = 4 \implies 0.0$ (Deceptive valley)
   * $u = 5 \implies 5.0$ (Global optimum)

3. **Failure Mode of Standard Genetic Algorithms:**
   * Uniform and single-point crossover independently disrupt partially completed building blocks when $u \in \{1, 2, 3, 4\}$, driving the population downhill into the all-zeros local suboptimum ($\sum \text{Trap}(0) = 8 \times 4.0 = 32.0$).
   * Independent bit mutations encounter exponential scaling $O(2^k)$ to traverse the Hamming distance of 5 against the fitness slope.

4. **BIG-LL Resolution:**
   * In-register transposition and co-occurrence evaluation detect tight statistical correlation across the 5 loci within each block ($\Delta_{ij} \ge \tau \cdot K / 2$).
   * The linkage graph forms connected components across building blocks $\{5b, 5b+1, 5b+2, 5b+3, 5b+4\}$.
   * Recombination transfers intact 5-bit building blocks as indivisible units via logic multiplexing, preserving optimal blocks and escaping the $32.0$ deceptive attractor.

### Empirical Invariants Verified

* Exact unitation response: $u \in [0, 5]$ matches theoretical curve.
* 40-generation progression: Population successfully discovers building blocks and achieves fitness $> 32.0$, falsifying the hypothesis of premature deceptive convergence.
