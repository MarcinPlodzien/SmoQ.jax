# Conventions of SmoQ.jax

Each statement below is what `src/smoqjax/engine.py` implements and what `tests/engine_suite.py` checks.

**Qubit ordering.** A state of $N$ qubits is a tensor `psi[s_0, s_1, ..., s_{N-1}]` with axis $q$ belonging to
qubit $q$. Flattened, the index is $i = s_0 2^{N-1} + s_1 2^{N-2} + \dots + s_{N-1}$, so qubit 0 is the most
significant bit, and `psi.reshape(-1)` agrees with the Kronecker product $v_0\otimes v_1\otimes\dots\otimes v_{N-1}$.

**Basis states.** $|0\rangle = (1,0)^T$ is the $+1$ eigenstate of $Z$ (spin up), $|1\rangle = (0,1)^T$ is spin down.

**Pauli matrices.** $X = \begin{pmatrix}0&1\\1&0\end{pmatrix}$, $Y = \begin{pmatrix}0&-i\\i&0\end{pmatrix}$,
$Z = \begin{pmatrix}1&0\\0&-1\end{pmatrix}$. Spin operators are $S^a = \sigma^a/2$; Hamiltonians built by
`heisenberg_terms` use Pauli matrices, $H = \sum_{\langle ij\rangle} (J_{xx} X_iX_j + J_{yy} Y_iY_j + J_{zz} Z_iZ_j)
+ \sum_i (h_x X_i + h_y Y_i + h_z Z_i)$.

**Ladder operators.** `SM` $=\sigma^- = |0\rangle\langle 1|$, `SP` $=\sigma^+ = |1\rangle\langle 0|$.

**Rotations.** $R_P(\theta) = \exp(-i\theta P/2)$ for $P \in \{X, Y, Z\}$, and $\exp(-i\theta PP/2)$ for the
two-qubit rotations `rxx`, `ryy`, `rzz`.

**Multi-qubit gates.** In `apply_gate(psi, U, qubits)` the first qubit listed is the most significant bit of the
$2^k\times 2^k$ matrix `U`, so `apply_gate(psi, CNOT, (c, t))` uses qubit `c` as control and `t` as target, for any
pair, adjacent or not, in either order.

**Density tensors.** A density operator of $N$ qubits is a rank-$2N$ tensor `rho[ket axes..., bra axes...]`;
`dm_matrix` reshapes it to a $2^N\times 2^N$ matrix and `to_dm` builds it from a state.

**Channels.** `kraus_depolarizing(p)`: $\rho\to(1-p)\rho + \tfrac{p}{3}(X\rho X+Y\rho Y+Z\rho Z)$, Bloch vector
shrinks by $1-4p/3$. `kraus_dephasing(p)` (= `kraus_phase_flip`): $\rho\to(1-p)\rho+pZ\rho Z$, coherences shrink by
$1-2p$. `kraus_amplitude_damping(g)`: decay $|1\rangle\to|0\rangle$ with probability $g$.

**Lindblad equation.** `lindblad_rhs` implements $\dot\rho = -i[H,\rho] + \sum_j \gamma_j\,(L_j\rho L_j^\dagger -
\tfrac12\{L_j^\dagger L_j,\rho\})$ with jumps given as `[(qubits, L, gamma), ...]` and $\hbar = 1$.

**Entropies** are in bits (base 2) unless `base` is given.
