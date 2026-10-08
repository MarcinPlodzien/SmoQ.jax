"""Validate the matrix-free engine against dense linear algebra.  Run through pytest (tests/test_engine.py) or directly: python tests/engine_suite.py"""
import os, sys, functools

import numpy as np
import jax, jax.numpy as jnp
import smoqjax.engine as sq

key = jax.random.PRNGKey(7)
N = 5


def kron_embed(U, qubits, N):
    """Dense embedding by brute force (axis q = qubit q, big-endian)."""
    k = len(qubits)
    U = np.asarray(U).reshape((2,) * (2 * k))
    full = np.zeros((2,) * (2 * N), dtype=complex)
    for idx in np.ndindex(*(2,) * N):
        for new in np.ndindex(*(2,) * k):
            out = list(idx)
            for a, q in zip(new, qubits):
                out[q] = a
            full[tuple(out) + idx] += U[new + tuple(idx[q] for q in qubits)]
    return full.reshape(2 ** N, 2 ** N)


def check(name, a, b, tol=1e-10):
    err = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    print(f"{'OK ' if err < tol else 'FAIL'} {name:45s} err={err:.2e}")
    assert err < tol, name


psi = sq.haar_state(key, N)
v = np.asarray(psi).reshape(-1)

# gates
U1 = sq.haar_unitary(jax.random.PRNGKey(1), 2)
U2 = sq.haar_unitary(jax.random.PRNGKey(2), 4)
check("1q gate", sq.apply_gate(psi, U1, [3]).reshape(-1), kron_embed(U1, [3], N) @ v)
check("2q gate (non-adjacent, reversed)", sq.apply_gate(psi, U2, [4, 1]).reshape(-1), kron_embed(U2, [4, 1], N) @ v)
check("CNOT kron convention", kron_embed(sq.CNOT, [0, 1], 2), np.asarray(sq.CNOT))

# density matrices
rho = sq.to_dm(psi)
R = np.outer(v, v.conj())
E = kron_embed(U2, [0, 2], N)
check("gate on DM", sq.dm_matrix(sq.apply_gate_dm(rho, U2, [0, 2])), E @ R @ E.conj().T)
K = sq.kraus_amplitude_damping(0.3)
ref = sum(kron_embed(k, [2], N) @ R @ kron_embed(k, [2], N).conj().T for k in np.asarray(K))
check("Kraus channel on DM", sq.dm_matrix(sq.apply_kraus_dm(rho, K, [2])), ref)
for nm, Kf in [("depol", sq.kraus_depolarizing(0.2)), ("deph", sq.kraus_dephasing(0.2)), ("AD", K),
               ("jump", sq.kraus_from_jump(sq.SM, 0.05))]:
    check(f"Kraus completeness {nm}", jnp.einsum("mab,mac->bc", jnp.conj(Kf), Kf), np.eye(2))

# rdm / partial trace
Rt = R.reshape((2,) * (2 * N))
ref = np.einsum("abcdeAbcDe->adAD", Rt).reshape(4, 4)
check("rdm from psi", sq.rdm(psi, [0, 3]), ref)
check("rdm from DM", sq.rdm_dm(rho, [0, 3]), ref)
check("expect_local vs pauli string", sq.expect_local(psi, sq.XX, [1, 4]), sq.expect_pauli_string(psi, {1: "X", 4: "X"}))
check("entropy: SVD vs eigen", sq.entanglement_entropy(psi, [0, 1]), sq.von_neumann_entropy(sq.rdm(psi, [0, 1])))
check("GHZ entropy = 1 bit", sq.entanglement_entropy(sq.ghz_state(6), [0, 1, 2]), 1.0)
check("GHZ circuit", sq.ghz_circuit(6), sq.ghz_state(6))
neg, _ = sq.negativity(sq.to_dm(sq.bell_state()), [0])
check("Bell negativity = 1/2", neg, 0.5)

# Hamiltonian & evolution
terms = sq.heisenberg_terms(N, 1.0, 1.0, 0.7, hx=0.4, hz=0.1)
Hd = sum(kron_embed(h, q, N) for q, h in terms)
check("dense H from matrix-free action", sq.dense_hamiltonian(terms, N), Hd)
w, V = np.linalg.eigh(Hd)
t = 1.3
ref = (V * np.exp(-1j * t * w)) @ V.conj().T @ v
check("exact_evolve", sq.exact_evolve(psi, terms, t).reshape(-1), ref)
check("chebyshev", sq.chebyshev_evolve(psi, terms, t, sq.spectral_bounds(terms, N)).reshape(-1), ref, 1e-9)
check("krylov", sq.krylov_evolve(psi, terms, t, m=25).reshape(-1), ref, 1e-9)
errs = []
for order, tol in [(1, 5e-2), (2, 1e-3), (4, 1e-6)]:
    out, _ = sq.tebd_evolve(psi, terms, t / 130, 130, order=order)
    errs.append(float(jnp.linalg.norm(out.reshape(-1) - ref)))
    check(f"TEBD order {order}", out.reshape(-1), ref, tol)
print("     TEBD errors (orders 1,2,4):", errs)
E0, gs = sq.lanczos_ground_state(terms, N, m=30)
check("Lanczos ground energy", E0, w[0], 1e-8)

# Lindblad: RK4 vs Trotter-Kraus vs MCWF
Nn = 3
terms3 = sq.heisenberg_terms(Nn, 1.0, 1.0, 1.0, hx=0.5)
g, dt, steps = 0.2, 0.01, 100
jumps = [((q,), sq.SM, g) for q in range(Nn)]
rho0 = sq.to_dm(sq.product_state("1+0"))
r_rk = rho0
for _ in range(steps):
    r_rk = sq.lindblad_rk4_step(r_rk, terms3, jumps, dt)
gates = sq.tebd_gates(terms3, dt, 2)
jk = [((q,), sq.kraus_from_jump(sq.SM, g * dt)) for q in range(Nn)]
r_tr = rho0
for _ in range(steps):
    r_tr = sq.lindblad_trotter_step_dm(r_tr, gates, jk)
check("Lindblad trace", jnp.trace(sq.dm_matrix(r_rk)).real, 1.0, 1e-8)
check("Lindblad RK4 vs Trotter-Kraus", sq.dm_matrix(r_rk), sq.dm_matrix(r_tr), 5e-3)


@jax.jit
def traj(k):
    def step(p, kk):
        return sq.lindblad_trotter_step_mcwf(kk, p, gates, jk), None
    p, _ = jax.lax.scan(step, sq.product_state("1+0"), jax.random.split(k, steps))
    return jnp.stack([sq.expect_local(p, sq.Z, [q]) for q in range(Nn)])


zs = jax.vmap(traj)(jax.random.split(key, 2000)).mean(0)
zref = jnp.stack([sq.expect_local_dm(r_tr, sq.Z, [q]) for q in range(Nn)])
check("MCWF <Z> vs DM (2000 traj, statistical)", zs, zref, 0.06)

# metrology
check("QFI GHZ = N^2", sq.qfi_pure(sq.ghz_state(6), sq.Z), 36.0)
check("QFI |+>^N = N", sq.qfi_pure(sq.product_state("+" * 6), sq.Z), 6.0)
check("QFI mixed == pure on pure state", sq.qfi_mixed(sq.dm_matrix(sq.to_dm(psi)), sq.collective_dense(sq.Y, N)), sq.qfi_pure(psi, sq.Y), 1e-7)
check("CSS squeezing = 1", sq.spin_squeezing(sq.product_state("+" * 6)), 1.0)
Nq = 8
m = np.array([(Nq - 2 * bin(i).count("1")) / 2 for i in range(2 ** Nq)])
css = sq.product_state("+" * Nq)
check("OAT diagonal evolution", sq.oat_evolve(css, 0.3).reshape(-1), np.exp(-1j * 0.3 * m ** 2) * np.asarray(css).reshape(-1))
xi = float(sq.spin_squeezing(sq.oat_evolve(css, 0.15)))
print(f"     OAT squeezing xi^2(N=8, chi t=0.15) = {xi:.3f}  (<1 expected)"); assert xi < 1

# magic
p4 = sq.haar_state(jax.random.PRNGKey(3), 4)
check("SRE FWHT vs brute force", sq.stabilizer_renyi_entropy(p4), sq.stabilizer_renyi_entropy_bruteforce(p4), 1e-9)
check("SRE stabilizer state = 0", sq.stabilizer_renyi_entropy(sq.ghz_state(5)), 0.0, 1e-9)
tstate = functools.reduce(lambda p, q: sq.apply_gate(p, sq.T, [q]), range(3), sq.product_state("+++"))
check("SRE T-states = 3*log2(4/3)", sq.stabilizer_renyi_entropy(tstate), 3 * np.log2(4 / 3), 1e-9)

# Cliffords, measurements, shadows
C = sq.single_qubit_cliffords()
check("24 single-qubit Cliffords", C.shape[0], 24)
bits = sq.sample_bitstrings(key, sq.ghz_state(4), 2000)
check("GHZ samples all-equal", float(jnp.all(bits == bits[:, :1])), 1.0)
o, post = sq.measure_qubit(key, sq.ghz_state(4), 0)
check("collapse of GHZ", jnp.abs(post[(int(o),) * 4]), 1.0)
xs = sq.sample_bitstrings(key, sq.product_state("+r"), 100, bases="XY")
check("X/Y-basis sampling deterministic", xs.sum(), 0)
ghz = sq.ghz_state(4)
bases, sb = sq.collect_pauli_shadows(key, ghz, 40000)
for ops in [{0: "Z", 1: "Z"}, {0: "X", 1: "X", 2: "X", 3: "X"}, {2: "Z"}]:
    est, se = sq.shadow_estimate_pauli(bases, sb, ops)
    exact = sq.expect_pauli_string(ghz, ops)
    print(f"     shadow {ops}: {float(est):+.3f} +- {float(se):.3f}  exact {float(exact):+.3f}")
    assert abs(est - exact) < 5 * se + 1e-3
b2, s2 = sq.collect_pauli_shadows(key, sq.bell_state(), 20000)
rho_hat = sum(sq.shadow_snapshot_dm(b2[i], s2[i]) for i in range(3000)) / 3000
check("shadow DM reconstruction (Bell, 3000 snaps)", rho_hat, sq.dm_matrix(sq.to_dm(sq.bell_state())), 0.12)

# VQC gradients
termsv = sq.heisenberg_terms(4, 1, 1, 1, hx=0.5)
cost = lambda th: sq.energy(termsv, sq.hardware_efficient_ansatz(th, 4, 3))
th = jax.random.normal(key, (sq.hea_num_params(4, 3),))
check("jax.grad vs parameter shift", jax.grad(cost)(th), sq.parameter_shift_grad(cost, th), 1e-8)
check("parameter shift with shift pi/4 (divides by 2 sin s)", jax.grad(cost)(th), sq.parameter_shift_grad(cost, th, shift=jnp.pi / 4), 1e-8)
st = sq.adam_init(th); gfun = jax.jit(jax.value_and_grad(cost))
for _ in range(500):
    val, gr = gfun(th); th, st = sq.adam_update(th, gr, st, lr=0.05)
Eg = np.linalg.eigvalsh(np.asarray(sq.dense_hamiltonian(termsv, 4)))[0]
print(f"     VQE: E={float(val):.4f}  exact={Eg:.4f}"); assert val - Eg < 0.15
print("\nALL CORE TESTS PASSED")

# regression: non-integer Renyi index needs |<P>|^(2 alpha) (a signed power gives nonsense)
_st = sq.apply_gate(sq.product_state("-+"), sq.T, [1])
for _al in (0.5, 1.5, 2, 3):
    check(f"SRE alpha={_al}: WHT vs brute force", sq.stabilizer_renyi_entropy(_st, alpha=_al),
          sq.stabilizer_renyi_entropy_bruteforce(_st, alpha=_al), 1e-9)
print("SRE non-integer alpha regression passed")

# ── Matrix product states: exact conversion, DMRG ground state, TEBD against the state vector ───────────────
psi_r = sq.haar_state(jax.random.PRNGKey(5), 8)
B_r, lam_r, eps_r = sq.state_to_mps(psi_r, 16)                       # chi = 2^(N/2): exact
check("MPS <-> state vector (chi = 2^(N/2))", sq.mps_to_state(B_r), psi_r, 1e-12)
assert float(np.sum(np.asarray(eps_r))) < 1e-12
check("MPS entropies vs state vector", sq.mps_entropies(lam_r)[4],
      sq.entanglement_entropy(psi_r, range(4)), 1e-10)
check("MPS <Z_j> vs state vector", sq.mps_expect_sites(B_r, lam_r, sq.Z)[3],
      jnp.real(sq.expect_local(psi_r, sq.Z, [3])), 1e-10)

termsm = sq.heisenberg_terms(8, 1.0, 1.0, 1.0)                        # Heisenberg chain, 8 spins
E_lanczos, _ = sq.lanczos_ground_state(termsm, 8, m=60, restarts=3)
W8 = sq.xxz_mpo(8, 1.0, 1.0, 1.0)
check("MPO -> dense vs term list", sq.mpo_to_dense(W8), sq.dense_hamiltonian(termsm, 8), 1e-10)
B_d, lam_d, hist = sq.dmrg(W8, sq.product_mps("01" * 4, 32)[0], 32, 4)
check("DMRG ground-state energy vs Lanczos", hist[-1][3], E_lanczos, 1e-8)

h_bonds = sq.bond_hamiltonians(8, Jzz=-1.0, hx=-1.0)                  # critical TFIM quench, 8 spins
order = list(range(0, 7, 2)) + list(range(1, 7, 2))
psi_sv, _ = sq.tebd_evolve(sq.product_state("0" * 8), [((j, j + 1), h_bonds[j]) for j in order], 0.05, 20, order=2)
B_t, lam_t = sq.product_mps("0" * 8, 16)
(B_e, _), out_e = sq.mps_tebd_evolve(B_t, lam_t, h_bonds, 0.05, 20)
check("MPS-TEBD vs state-vector TEBD (chi = 2^(N/2))", sq.mps_to_state(B_e), psi_sv, 1e-10)
print("MPS / DMRG / TEBD tests passed")

# ── Regression (tensor-network cross-check 2026-10-07): lam must be the Schmidt values of the returned B ──────
# Before the fix, dmrg returned the lam recorded during the last right-to-left half sweep, which later local
# steps had made stale (|d<Z>| = 0.25 after one sweep), and state_to_mps with truncation stored lam before the
# next cut was truncated.  The references below are plain NumPy on the state vector that B encodes.
def _np_state(B):
    return np.asarray(sq.mps_to_state(B)).reshape(-1)


def _np_z(st, n):
    T = st.reshape([2] * n)
    return np.array([np.sum(np.abs(np.take(T, 0, axis=j)) ** 2 - np.abs(np.take(T, 1, axis=j)) ** 2) for j in range(n)])


def _np_schmidt(st, n, c):
    return np.linalg.svd(st.reshape(2 ** c, -1), compute_uv=False)


def _np_S(st, n):
    out = []
    for c in range(1, n):
        p = _np_schmidt(st, n, c) ** 2
        p = p[p > 1e-300]
        out.append(-np.sum(p * np.log2(p)))
    return np.array(out)


def _np_bond_energies(st, n, h_bonds):
    T = st.reshape([2] * n)
    out = []
    for j in range(n - 1):
        M = np.moveaxis(T, (j, j + 1), (0, 1)).reshape(4, -1)
        out.append(np.real(np.trace(np.asarray(h_bonds[j]) @ (M @ M.conj().T))))
    return np.array(out)


for _n, _chi, _sw in ((8, 16, 1), (10, 4, 1), (10, 4, 8)):        # one sweep at full chi; truncated chi, 1 and 8 sweeps
    _W = sq.xxz_mpo(_n, 1.0, 1.0, 0.5, hx=0.7)
    _B, _lam, _ = sq.dmrg(_W, sq.product_mps(("01" * _n)[:_n], _chi)[0], _chi, _sw)
    _st = _np_state(_B)
    check(f"dmrg N={_n} chi={_chi} {_sw} sweep(s): <Z_j> from lam vs B", sq.mps_expect_sites(_B, _lam, sq.Z), _np_z(_st, _n), 1e-10)
    check(f"dmrg N={_n} chi={_chi} {_sw} sweep(s): entropies from lam vs B", sq.mps_entropies(_lam)[1:_n], _np_S(_st, _n), 1e-10)
    _hb = sq.bond_hamiltonians(_n, 1.0, 1.0, 0.5, hx=0.7)
    check(f"dmrg N={_n} chi={_chi} {_sw} sweep(s): bond energies from lam vs B", sq.mps_expect_bonds(_B, _lam, _hb),
          _np_bond_energies(_st, _n, _hb), 1e-10)
    _B2, _lam2 = sq.mps_recanonicalise(_B)
    check(f"mps_recanonicalise N={_n} chi={_chi}: state unchanged", _np_state(_B2), _st, 1e-12)

for _chi in (2, 4):                                                    # state_to_mps with truncation
    _B, _lam, _eps = sq.state_to_mps(psi_r, _chi)
    assert float(np.max(_eps)) > 1e-3                                   # something was really discarded
    _st = _np_state(_B)
    check(f"state_to_mps chi={_chi} (truncated): lam vs Schmidt values of stored state",
          np.stack([_lam[c, :_chi] for c in range(1, 8)]), np.stack([np.pad(_np_schmidt(_st, 8, c), (0, _chi))[:_chi] for c in range(1, 8)]), 1e-10)
    check(f"state_to_mps chi={_chi} (truncated): <Z_j> from lam vs stored state", sq.mps_expect_sites(_B, _lam, sq.Z), _np_z(_st, 8), 1e-10)
    check(f"state_to_mps chi={_chi} (truncated): norm of stored state", np.vdot(_st, _st).real, 1.0, 1e-12)

for _i, _j in ((3, 3), (5, 2)):                                        # misuse of mps_correlator must raise
    try:
        sq.mps_correlator(B_r, lam_r, sq.Z, _i, sq.Z, _j)
    except ValueError:
        print(f"OK   mps_correlator(i={_i}, j={_j}) raises ValueError")
    else:
        print(f"FAIL mps_correlator(i={_i}, j={_j}) returned a number")
        raise AssertionError("mps_correlator accepted j <= i")
print("MPS lam-consistency regression tests passed")

# ── Regression (2026-10-08): DMRG collapse to the zero vector, and MPO start/end-state convention ─────────────
# Before the fix, split_two_site kept the arbitrary LAPACK completion of its zero singular values; some of these bond
# "states" had norm 0, H_eff had a spurious eigenvalue 0 on them, and from the Neel start of the ferromagnetic TFIM
# (all physical local energies positive) Lanczos landed there at N = 24: E = 0.0, ||psi|| = 1.7e-15, no error.
def _tfim_free_fermion(n, J=1.0, h=1.0):
    """Exact ground energy of H = -J sum Z_i Z_{i+1} - h sum X_i, open chain: E0 = -(1/2) sum_k Lambda_k."""
    A, Bm = 2 * h * np.eye(n), np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = A[i + 1, i] = -J
        Bm[i, i + 1], Bm[i + 1, i] = -J, J
    return -0.5 * np.sum(np.linalg.svd(A - Bm, compute_uv=False))


def _np_kron_chain(ops):
    out = np.eye(1)
    for o in ops:
        out = np.kron(out, np.asarray(o))
    return out


def _np_dense(n, bonds, sites):
    """sum_j sum_(c,P,Q) c P_j Q_{j+1} + sum_j sum_(c,P) c P_j, by Kronecker products (big-endian, site 0 first)."""
    H = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for j in range(n):
        for c, P in sites:
            H += c * _np_kron_chain([P if k == j else np.eye(2) for k in range(n)])
        if j < n - 1:
            for c, P, Q in bonds:
                H += c * _np_kron_chain([P if k == j else Q if k == j + 1 else np.eye(2) for k in range(n)])
    return H


_ff8 = _tfim_free_fermion(8)
_Wt = sq.xxz_mpo(8, 0.0, 0.0, -1.0, hx=-1.0)
check("free-fermion TFIM energy vs dense ED (N=8)", _ff8, np.linalg.eigvalsh(np.asarray(sq.mpo_to_dense(_Wt)))[0], 1e-10)
_W24 = sq.xxz_mpo(24, 0.0, 0.0, -1.0, hx=-1.0)
_B24, _lam24, _h24 = sq.dmrg(_W24, sq.product_mps("01" * 12, 32)[0], 32, 4)
check("dmrg TFIM N=24 from Neel start: exact energy", _h24[-1][3], _tfim_free_fermion(24), 1e-8)
check("dmrg TFIM N=24 from Neel start: <H> of returned B", sq.mpo_expectation(_B24, _W24), _tfim_free_fermion(24), 1e-8)
check("dmrg TFIM N=24 from Neel start: <psi|psi> = 1", jnp.real(sq.mps_overlap(_B24, _B24)), 1.0, 1e-10)

_W10 = sq.xxz_mpo(10, 1.0, 1.0, 1.0)                                    # all-up is an eigenstate of this S^z-conserving H
_E10 = np.linalg.eigvalsh(np.asarray(sq.mpo_to_dense(_W10)))[0]
_B10, _, _h10 = sq.dmrg(_W10, sq.product_mps("0" * 10, 32)[0], 32, 4)
check("dmrg Heisenberg N=10 from the eigenstate |0..0>: ground energy", _h10[-1][3], _E10, 1e-8)
check("dmrg Heisenberg N=10 from |0..0>: <psi|psi> = 1", jnp.real(sq.mps_overlap(_B10, _B10)), 1.0, 1e-10)

_Hs = np.diag([3.0, 1.0, -2.0, 0.5])                                    # Lanczos started on an eigenvector (beta_0 = 0)
_El, _xl = sq.lanczos_lowest(lambda u: jnp.asarray(_Hs, sq.CDTYPE) @ u, jnp.asarray([1.0, 0, 0, 0], sq.CDTYPE), 6)
check("lanczos_lowest started on an eigenvector: lowest eigenvalue", _El, -2.0, 1e-12)
check("lanczos_lowest started on an eigenvector: eigenvector", abs(np.asarray(_xl)[2]), 1.0, 1e-12)

try:                                                                    # a zero start must raise, not return a number
    sq.dmrg(_Wt, jnp.zeros((8, 4, 2, 4), dtype=sq.CDTYPE), 4, 1)
except ValueError:
    print("OK   dmrg with a zero start MPS raises ValueError")
else:
    raise AssertionError("dmrg accepted a zero start MPS")

# MPO convention: start state = last index D-1, end state = index 0, for any D (mpo_to_dense used row 4 before)
_n = 6
_X, _Z, _I = np.asarray(sq.X), np.asarray(sq.Z), np.eye(2)
_W3 = np.zeros((3, 3, 2, 2), dtype=complex)                             # hand-built TFIM MPO, D = 3: 2 start, 1 Z placed, 0 end
_W3[2, 2] = _W3[0, 0] = _I
_W3[2, 1], _W3[1, 0], _W3[2, 0] = -1.0 * _Z, _Z, -0.7 * _X
_Ws3 = np.stack([_W3] * _n)
_Ws3[0] = np.zeros_like(_W3); _Ws3[0][2] = _W3[2]
_Ws3[-1] = np.zeros_like(_W3); _Ws3[-1][:, 0] = _W3[:, 0]
_Ws3 = jnp.asarray(_Ws3, sq.CDTYPE)
_H3 = _np_dense(_n, [(-1.0, _Z, _Z)], [(-0.7, _X)])
check("mpo_to_dense, D=3 hand-built TFIM MPO vs kron", sq.mpo_to_dense(_Ws3), _H3, 1e-12)
_W5 = np.asarray(sq.xxz_mpo(_n, 1.0, 0.8, 0.5, hx=0.3, hz=-0.2))       # D = 5 -> D = 6: unused state 4, start state moved to 5
_W6 = np.zeros((_n, 6, 6, 2, 2), dtype=complex)
_perm = [0, 1, 2, 3, 5]
_W6[:, np.ix_(_perm, _perm)[0], np.ix_(_perm, _perm)[1]] = _W5
_W6 = jnp.asarray(_W6, sq.CDTYPE)
_H5 = _np_dense(_n, [(1.0, _X, _X), (0.8, np.asarray(sq.Y), np.asarray(sq.Y)), (0.5, _Z, _Z)], [(0.3, _X), (-0.2, _Z)])
check("mpo_to_dense, D=5 xxz_mpo vs kron", sq.mpo_to_dense(jnp.asarray(_W5, sq.CDTYPE)), _H5, 1e-12)
check("mpo_to_dense, D=6 (padded, start state D-1) vs kron", sq.mpo_to_dense(_W6), _H5, 1e-12)
_Bp = sq.product_mps("01+-0+", 8)[0]
check("mpo_expectation, D=6 vs D=5", sq.mpo_expectation(_Bp, _W6), sq.mpo_expectation(_Bp, jnp.asarray(_W5, sq.CDTYPE)), 1e-12)
_, _, _h6 = sq.dmrg(_W6, sq.product_mps("01" * 3, 8)[0], 8, 3)
_, _, _h3 = sq.dmrg(_Ws3, sq.product_mps("01" * 3, 8)[0], 8, 3)
check("dmrg with the D=6 MPO: ground energy", _h6[-1][3], np.linalg.eigvalsh(_H5)[0], 1e-8)
check("dmrg with the D=3 MPO: ground energy", _h3[-1][3], np.linalg.eigvalsh(_H3)[0], 1e-8)
print("DMRG-collapse and MPO-convention regression tests passed")
