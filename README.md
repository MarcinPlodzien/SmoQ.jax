# SmoQ.jax

**A matrix-free JAX engine for quantum many-body systems.**

SmoQ.jax stores the state of $N$ spins-1/2 (qubits) as a rank-$N$ tensor of shape `(2,)*N` and applies every
operator, whether a gate, a Hamiltonian term, a Kraus operator or a projector, through a single `einsum`
contraction. No $2^N\times 2^N$ matrix is ever built. Every function is written in `jax.numpy`, so whole simulations
can be compiled with `jit`, batched with `vmap`, looped with `lax.scan` and differentiated with `grad`, on CPU or GPU.

The engine is developed step by step, with every derivation and every check, in the lecture course
*Quantum Many-Body Simulation in JAX: lectures from scratch* by the same author.

## Installation

```bash
git clone https://github.com/MarcinPlodzien/SmoQ.jax
cd SmoQ.jax
pip install -e .            # for NVIDIA GPUs first: pip install -U "jax[cuda12]"
```

## Example

```python
import jax
import smoqjax as sq

psi = sq.zero_state(3)                          # |000>, a tensor of shape (2, 2, 2)
psi = sq.apply_gate(psi, sq.H, (0,))
psi = sq.apply_gate(psi, sq.CNOT, (0, 1))
psi = sq.apply_gate(psi, sq.CNOT, (1, 2))       # GHZ state
print(sq.sample_bitstrings(jax.random.PRNGKey(0), psi, shots=4))

terms = sq.heisenberg_terms(16)                 # Heisenberg chain as a list of local terms
E0, gs = sq.lanczos_ground_state(terms, 16)     # matrix-free Lanczos
```

`examples/starter.py` runs five such snippets: a circuit, a ground state, a quench with TEBD, a driven-dissipative
qubit with the Lindblad equation, and the gradient of a variational circuit.

## What is inside

| topic | functions |
|---|---|
| gates | `rx`, `ry`, `rz`, `rpp`, `rxx`, `ryy`, `rzz`, `controlled`, constants `X`, `Y`, `Z`, `H`, `S`, `T`, `CNOT`, `CZ`, `SWAP` |
| states | `zero_state`, `product_state`, `basis_state`, `ghz_state`, `w_state`, `dicke_state`, `cluster_state`, `bell_state`, `haar_state` |
| applying operators | `apply_gate`, `apply_gate_dm`, `apply_kraus_dm`, `apply_pauli_string`, `apply_gates` |
| observables and entanglement | `expect_local`, `expect_pauli_string`, `rdm`, `rdm_dm`, `purity`, `von_neumann_entropy`, `entanglement_entropy`, `schmidt_values`, `fidelity_pure`, `fidelity_dm`, `trace_distance`, `partial_transpose`, `negativity` |
| measurement | `measure_qubit`, `reset_qubit`, `sample_bitstrings` |
| channels | `kraus_depolarizing`, `kraus_dephasing`, `kraus_bit_flip`, `kraus_amplitude_damping`, `kraus_from_jump`, `apply_kraus_mcwf` |
| Hamiltonians | `heisenberg_terms`, `apply_hamiltonian`, `energy`, `dense_hamiltonian` |
| time evolution | `tebd_gates`, `tebd_evolve`, `exact_evolve`, `chebyshev_evolve`, `krylov_evolve`, `lanczos`, `lanczos_ground_state` |
| open systems | `lindblad_rhs`, `lindblad_rk4_step`, `lindblad_trotter_step_dm`, `lindblad_trotter_step_mcwf` |
| metrology | `qfi_pure`, `qfi_mixed`, `spin_moments`, `spin_squeezing`, `oat_evolve` |
| randomness and complexity | `haar_unitary`, `brickwall`, `single_qubit_cliffords`, `collect_pauli_shadows`, `shadow_estimate_pauli`, `stabilizer_renyi_entropy` |
| variational circuits | `hardware_efficient_ansatz`, `parameter_shift_grad`, `spsa_grad`, `adam_init`, `adam_update` |
| tensor networks | `state_to_mps`, `mps_to_state`, `mps_expect_sites`, `mps_entropies`, `xxz_mpo`, `dmrg`, `mps_tebd_evolve`, `mps_rdm2` |

The complete engine is one documented file, `src/smoqjax/engine.py`: every function carries the mathematics it
implements in its docstring. Conventions (qubit ordering, Pauli matrices, rotations, channels) are in
[CONVENTIONS.md](CONVENTIONS.md).

## Precision

Double precision (`float64`/`complex128`) is the default, and importing the package switches on
`jax_enable_x64`. For single precision set `QE_PRECISION=single` in the environment before the import.

## Tests

```bash
pip install -e ".[test]"
pytest                      # every primitive validated against dense linear algebra
```

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).
