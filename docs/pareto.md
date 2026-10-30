# Component Documentation: Pareto Multi-Objective Engine (`pareto.zig`)

## 1. Specification

The `pareto.zig` module provides non-dominated sorting and crowding distance calculations for multi-objective optimization over binary chromosomes ($L$ bits, $M$ objective values).

### Core Data Structures

1. **`ParetoChromosome(comptime L: usize, comptime M: usize)`**:
   * Bitset array representation: `genes: [words_len]u64`.
   * Multi-objective evaluation array: `objectives: [M]f64` (maximization semantics).
   * Non-dominated front rank: `rank: usize`.
   * Diversity metric: `crowding_distance: f64`.

2. **Dominance Operator (`dominates`)**:
   Vector $\mathbf{u} = (u_1, \dots, u_M)$ dominates vector $\mathbf{v} = (v_1, \dots, v_M)$ ($\mathbf{u} \succ \mathbf{v}$) if and only if:
   $$\forall k \in \{1, \dots, M\}: u_k \ge v_k \quad \land \quad \exists k \in \{1, \dots, M\}: u_k > v_k$$

3. **Fast Non-Dominated Sorting (`computeNonDominatedRanks`)**:
   * Partitions the population into Pareto fronts $\mathcal{F}_0, \mathcal{F}_1, \dots, \mathcal{F}_k$.
   * Calculates domination count $n_p$ (number of individuals dominating $p$) and set of dominated individuals $S_p$.
   * Assigns rank $= 0$ to non-dominated individuals ($n_p = 0$).
   * Iteratively decrements domination counts for members of $S_p$ to extract subsequent fronts.

4. **Crowding Distance Assignment (`computeCrowdingDistances`)**:
   * Evaluated per front $\mathcal{F}_k$.
   * Boundary elements for each objective $m \in \{1, \dots, M\}$ receive distance $= +\infty$.
   * Intermediate elements receive cumulative normalized cuboid distances:
     $$d_i = \sum_{m=1}^{M} \frac{f_m(i+1) - f_m(i-1)}{f_m^{\max} - f_m^{\min}}$$

5. **`ParetoEngine(comptime L: usize, comptime M: usize, comptime Adapter: type)`**:
   * Pre-allocates double buffers for parent and offspring populations (`pop_current`, `pop_next`).
   * Crowded tournament selection: selects based on $\text{rank}(a) < \text{rank}(b)$, breaking ties with $\text{crowding\_distance}(a) > \text{crowding\_distance}(b)$.
   * Dynamic mutation and crossover operators applied across generations.
   * Zero heap allocations occur in generational inner loops outside rank sort partitions.
