"""SmoQ.jax in five snippets: a circuit, a ground state, unitary and dissipative dynamics, a variational gradient."""
import jax
import smoqjax as sq

key = jax.random.PRNGKey(0)

# 1. Quantum circuit: a three-qubit GHZ state, sampled
psi = sq.zero_state(3)                     # rank-3 tensor of shape (2, 2, 2)
psi = sq.apply_gate(psi, sq.H, (0,))
psi = sq.apply_gate(psi, sq.CNOT, (0, 1))
psi = sq.apply_gate(psi, sq.CNOT, (1, 2))
bits = sq.sample_bitstrings(key, psi, shots=6)
print("GHZ samples:", bits.tolist())       # only 000 and 111

# 2. Many-body physics: Heisenberg chain of 16 spins, Lanczos
N = 16
H_xxx = sq.heisenberg_terms(N)             # local terms, no 2^N x 2^N matrix
E0, gs = sq.lanczos_ground_state(H_xxx, N)
S_half = sq.entanglement_entropy(gs, list(range(N // 2)))
print(f"E0 = {E0:.6f},  half-chain entropy = {S_half:.4f} bits")

# 3. Unitary dynamics: a Neel state melts (2nd-order TEBD)
neel = sq.product_state("01" * (N // 2))
psi_t, z0 = sq.tebd_evolve(neel, H_xxx, dt=0.05, n_steps=40,
                           observe=lambda p: sq.expect_local(p, sq.Z, (0,)))
print("<Z_0>(t) every 10 steps:", [round(float(z), 3) for z in z0[::10]])

# 4. Dissipative dynamics: driven, decaying qubit (Lindblad)
rho = sq.to_dm(sq.zero_state(1))
drive = [((0,), 0.5 * sq.X)]
decay = [((0,), sq.SP, 0.2)]               # jump operator |0><1| = sigma^+ (decay |1> -> |0>), rate 0.2
for _ in range(200):
    rho = sq.lindblad_rk4_step(rho, drive, decay, dt=0.05)
print(f"excited population at t = 10: {sq.dm_matrix(rho)[1, 1].real:.4f}")

# 5. Variational circuit: energy gradient by autodiff
n, layers = 4, 2
H4 = sq.heisenberg_terms(n)
cost = lambda th: sq.energy(H4, sq.hardware_efficient_ansatz(th, n, layers))
theta = 0.1 * jax.random.normal(key, (sq.hea_num_params(n, layers),))
g = jax.grad(cost)(theta)
print(f"energy = {cost(theta):.4f}, gradient has {g.size} components")
