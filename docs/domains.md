# Component Documentation: Real-World Domain Implementations (`domains/`)

## 1. Specification Overview

The `domains/` module implements problem domains interfacing with the core BIG-LL genetic engine via the zero-overhead adapter architecture. All domain evaluators adhere to zero-allocation guarantees during generational loops and enforce exact physical and combinatorial invariants.

## 2. Airfoil Aerodynamic Optimization (`airfoil.zig`)

### Problem Formulation
Multi-objective aerodynamic design maximizing subsonic Lift-to-Drag ratio ($C_L / C_D$) subject to structural thickness and curvature smoothness constraints.

### Chromosome Representation
$L = 256$ bits ($16 \times 16$-bit Gray-coded parameters).
* 8 upper surface $y$-coordinates: $y_{\text{upper}} \in [0.0, 0.25]$.
* 8 lower surface $y$-coordinates: $y_{\text{lower}} \in [-0.20, 0.0]$.

### Objective Function
$$C_L = 2.5 \times \int_{0}^{1} \frac{y_{\text{upper}}(x) + y_{\text{lower}}(x)}{2} \, dx$$
$$C_D = 0.01 + 0.05 \int (y_{\text{upper}} - y_{\text{lower}}) \, dx + 50.0 \sum (\Delta^2 y)^2 + 10.0 |y_{\text{upper}}(1) - y_{\text{lower}}(1)|$$
$$\text{Fitness} = \frac{C_L}{C_D}$$

## 3. Fleet Logistics & CVRP (`logistics.zig`)

### Problem Formulation
Capacitated Vehicle Routing Problem (CVRP) with central depot (node 0), customer set $V_c = \{1, \dots, N\}$, and vehicle load capacity $Q = 35.0$.

### Chromosome Representation and In-Register Decoding
$L = 512$ bits ($32 \times 16$-bit Random Keys). In-register argsort maps continuous random keys to a valid permutation $\sigma \in S_{32}$ representing the giant tour without illegal duplicate visits.

### Prins' Split Dynamic Programming Formulation
Rather than relying on greedy trip cutting (which suffers from arbitrary subroute splitting and suboptimal vehicle boundary placement), `LogisticsDomain` evaluates the giant tour via Prins' Split Dynamic Programming algorithm.

An acyclic directed auxiliary graph $G = (V, A)$ is constructed over topological nodes $\{0, 1, \dots, N\}$, where node $i$ represents the state where customers $\sigma_1, \dots, \sigma_i$ have been served. An arc $(i, j)$ exists if and only if customer sequence $(\sigma_{i+1}, \dots, \sigma_j)$ respects vehicle capacity:
$$\sum_{k=i+1}^j d_{\sigma_k} \le Q$$
The cost of arc $(i, j)$ corresponds to the dedicated single-vehicle tour cost:
$$c_{ij} = d(0, \sigma_{i+1}) + \sum_{k=i+1}^{j-1} d(\sigma_k, \sigma_{k+1}) + d(\sigma_j, 0)$$

The optimal partitioning into valid vehicle routes corresponds to the Bellman shortest path from node 0 to node $N$:
$$V(j) = \min_{0 \le i < j, \sum_{k=i+1}^j d_{\sigma_k} \le Q} \{ V(i) + c_{ij} \}$$
with boundary condition $V(0) = 0$.

Because the auxiliary graph is topologically ordered by definition, the recurrence is solved in $O(N)$ expected time using an in-stack array `V: [NUM_CUSTOMERS + 1]f64` requiring exactly 264 bytes of stack space with zero runtime heap allocations.

### Granular $O(1)$ Delta-Evaluation 2-Opt Polish
To refine permutations without quadratic evaluation explosion, `polishTour` executes local 2-opt edge exchanges on the giant tour using exact $O(1)$ delta distance checks:
$$\Delta = (d(\sigma_i, \sigma_j) + d(\sigma_{i+1}, \sigma_{j+1})) - (d(\sigma_i, \sigma_{i+1}) + d(\sigma_j, \sigma_{j+1}))$$
Reversal of the sub-segment is performed in-place only when $\Delta < -10^{-6}$.

### Fitness Metric
$$\text{Fitness} = \frac{10000.0}{1.0 + D_{\text{optimal}}}$$
where $D_{\text{optimal}} = V(N)$.

## 4. Electrical Power Grid Unit Commitment (`power_grid.zig`)

### Problem Formulation
Mixed-Integer Non-Linear Programming (MINLP) for economic dispatch and unit commitment over 8 thermal generating units subject to strict grid power balance and spinning reserve limits.

### Chromosome Representation
$L = 136$ bits ($8 \times 1$-bit commitment switches + $8 \times 16$-bit continuous power generation outputs).

### Exact Physical Proportional Power Balance Projection
Pure stochastic recombination in MINLP domains often generates candidate states where committed generators fail to meet the instantaneous power demand:
$$\Delta P = \sum_{i=1}^8 u_i P_i - P_{\text{demand}} \ne 0$$
To prevent gradient stalling and erratic quadratic penalty oscillations, `PowerGridDomain.projectPowerBalance` applies an exact physical proportional generation shift across all active units prior to fitness assignment:
$$P_i \leftarrow \text{clamp}\left( P_i - \Delta P \cdot \frac{P_i^{\max}}{\sum_{j=1}^8 u_j P_j^{\max}}, P_i^{\min}, P_i^{\max} \right)$$
This projects the candidate directly onto the zero-imbalance manifold without altering discrete commitment topologies.

### Objective Function and Penalties
Generating costs follow quadratic heat rate curves:
$$\text{Cost} = \sum_{g=1}^8 u_g (a_g P_g^2 + b_g P_g + c_{0,g})$$
System-level constraint violations are quadratically and linearly penalized:
$$\text{ImbalancePenalty} = 25.0 \cdot \left(\sum_{g=1}^8 u_g P_g - P_{\text{demand}}\right)^2$$
$$\text{ReservePenalty} = 150.0 \cdot \max\left(0, P_{\text{reserve}} - \sum_{g=1}^8 u_g P_g^{\max}\right)$$
$$\text{Fitness} = \frac{100000.0}{1.0 + \text{Cost} + \text{ImbalancePenalty} + \text{ReservePenalty}}$$

## 5. Epistatic Biomarker Panel Selection (`bioinformatics.zig`)

### Problem Formulation
Identification of optimal epistatic biomarker panels from candidate single nucleotide polymorphisms (SNPs) under cardinality constraints.

### Chromosome Representation
$L = 128$ bits ($N = 128$ candidate loci).

### Hardware Bit-Parallel Decoding
Bitboard extraction utilizes compiler intrinsic hardware trailing zero count (`@ctz`) to unpack active loci indices directly into registers in $O(|S|)$ operations without traversing unselected loci.

### Objective Function
$$\text{Score} = \sum_{i \in S} w_i + 15.0 \sum_{(i,j) \in \mathcal{E}} \mathbb{I}(i, j \in S) + 35.0 \cdot \mathbb{I}(7, 42, 99 \in S) - \text{Penalty}(|S| > 12)$$

## 6. Algorithmic Trading & Portfolio Allocation (`finance.zig`)

### Problem Formulation
Joint discrete rule induction (Trend, Volatility, Momentum) coupled with continuous Markowitz mean-variance portfolio allocation across asset classes.

### Chromosome Representation
$L = 104$ bits (8 rule bits + $6 \times 16$-bit continuous asset weights).

### Objective Function
Given asset expected returns $\boldsymbol{\mu}$ and empirical covariance matrix $\boldsymbol{\Sigma}$:
$$\text{Fitness} = \frac{\mu_p - r_f}{\sigma_p} + 0.1 \cdot H(\mathbf{w})$$
where $H(\mathbf{w}) = -\sum_{i=1}^6 w_i \ln(w_i)$ imposes an informational entropy regularizer discouraging single-asset corner collapse.

## 7. MIMO Wireless Interference Alignment (`telecom.zig`)

### Problem Formulation
Simultaneous discrete channel assignment and continuous transmit power control for 16 interfering links over 4 orthogonal frequency bands.

### Chromosome Representation
$L = 160$ bits ($16 \times 2$-bit channel assignments + $16 \times 8$-bit power levels).

### Objective Function
Signal-to-Interference-plus-Noise Ratio (SINR) for link $i$:
$$\text{SINR}_i = \frac{P_i G_{ii}}{N_0 + \sum_{j \ne i, c_j = c_i} P_j G_{ji}}$$
System capacity sum-rate:
$$\text{Fitness} = \sum_{i=1}^{16} \log_2(1 + \text{SINR}_i) - 50.0 \sum_{i=1}^{16} \max(0, 0.25 - \text{SINR}_i)$$
