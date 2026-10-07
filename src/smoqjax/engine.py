"""
quantum_engine.py -- a matrix-free state-vector simulator built on jax.numpy.einsum
====================================================================================

Author : Marcin Plodzien, Institute of Theoretical Physics, Jagiellonian University
         ORCID https://orcid.org/0000-0002-0835-1644 | https://github.com/MarcinPlodzien

This file is SmoQ.jax, a matrix-free JAX engine for quantum many-body systems, developed
in and used throughout the lecture course "Quantum Many-Body Simulation: from a single spin to
quantum machine learning".  Every notebook is self-contained:
tools/build_nb.py copies the functions a notebook uses from this file into its
"Engine recap" cell, so all notebooks share exactly the same code, and tests/test_core.py
validates every function against dense linear algebra.  The functions are derived step
by step in the notebooks of Chapters 1-3; here every docstring repeats the essential
math so that the file can also be read on its own.


################################################################################
#  0. EINSUM PRIMER  -- read this first
################################################################################

0.1  Index notation
-------------------
Physics formulas are sums over repeated indices.  `einsum` is exactly that notation,
executable:  you NAME the indices of every input, and say which indices SURVIVE.

      "input_1 , input_2 , ...  ->  output"

  Rule 1  a letter that appears in the inputs but NOT in the output is SUMMED over;
  Rule 2  a letter that appears in the output is kept (free index);
  Rule 3  the same letter in two inputs ties those axes together (they must have equal size).

0.2  The classic examples
-------------------------
  matrix-vector   y_i   = sum_j A_ij x_j            einsum("ij,j->i",   A, x)
  matrix-matrix   C_ik  = sum_j A_ij B_jk           einsum("ij,jk->ik", A, B)
  inner product   s     = sum_i a_i^* b_i           einsum("i,i->",     a.conj(), b)
  outer product   M_ij  = a_i b_j                   einsum("i,j->ij",   a, b)      (nothing summed)
  trace           t     = sum_i A_ii                einsum("ii->",      A)         (repeated letter in ONE input)
  transpose       B_ji  = A_ij                      einsum("ij->ji",    A)
  batched matmul  C_bik = sum_j A_bij B_bjk         einsum("bij,bjk->bik", A, B)
  Tr(ABC)         t     = sum_ijk A_ij B_jk C_ki    einsum("ij,jk,ki->", A, B, C)

0.3  Quantum states ARE tensors
-------------------------------
An N-qubit state  |psi> = sum_{s_0..s_{N-1}} psi[s_0,...,s_{N-1}] |s_0 ... s_{N-1}>
is a rank-N array of shape (2,2,...,2): ONE AXIS PER QUBIT.  The familiar "vector of
length 2^N" is just `psi.reshape(-1)`; with C-ordering the flat index is
      i = s_0 2^(N-1) + s_1 2^(N-2) + ... + s_{N-1} 2^0     (qubit 0 = most significant bit).

0.4  A gate on qubit q  =  contraction with axis q
-------------------------------------------------
Textbook:   (1 x ... x U x ... x 1)|psi>   -- a 2^N x 2^N matrix: 4^N numbers, 4^N work.
Index form: psi'[s_0,..,a,..,s_{N-1}] = sum_b U[a,b] psi[s_0,..,b,..,s_{N-1}]
            For N=3, q=1:  einsum("ab,xbz->xaz", U, psi)       -- 2*2^N work, NO big matrix.
Two qubits: reshape the 4x4 matrix to U[a,b,c,d] (a,b = outputs; c,d = inputs):
            For N=4 on qubits (1,3): einsum("abcd,xcyd->xayb", U4, psi)
The qubits need not be neighbours and no SWAPs are needed: einsum addresses any axes.

0.5  Reduced density matrices = contraction of psi with psi^*
------------------------------------------------------------
rho_A = Tr_B |psi><psi| ;  keep the axes of A open on both copies, tie the axes of B:
            N=3, A={0}:   rho[a,a'] = sum_{y,z} psi[a,y,z] psi^*[a',y,z]
                          einsum("ayz,Ayz->aA", psi, psi.conj())
For a density TENSOR rho[s_0..s_{N-1}; s'_0..s'_{N-1}] the partial trace is a repeated
letter inside a single operand:   N=2, trace out qubit 1:  einsum("abAb->aA", rho).

0.6  Why this is "matrix-free"
------------------------------
Memory is O(2^N) (the state) and each k-qubit gate costs O(2^k 2^N).  Hamiltonians,
channels, measurements, time evolution are all expressed as sums/products of such local
contractions -- the 2^N x 2^N operator never exists.

0.7  JAX practices used everywhere
----------------------------------
  * pure functions (no in-place mutation): state in -> state out;
  * `jax.jit`      compile a whole circuit / time step into one fused XLA program;
  * `jax.lax.scan` time loops that compile once instead of unrolling;
  * `jax.vmap`     trajectories, snapshots, parameter batches without Python loops;
  * `jax.grad`     exact gradients of variational circuits;
  * explicit PRNG keys (`jax.random.split`) -> reproducible stochastic simulations;
  * float64 / complex128 (enabled below) -- quantum dynamics needs the precision.
  Qubit labels (`qubits` arguments) are static Python ints: they define the einsum
  string at trace time.  Angles, gates and states are traced arrays.
"""
from __future__ import annotations

import os
import string
from functools import partial

import math
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax

# Precision is configurable: "double" = float64/complex128 (default; needed for long time evolutions and
# tight validation), "single" = float32/complex64 (half the memory, faster on consumer GPUs).
# In a notebook PRECISION is defined in the Configuration cell; for this module use the env var QE_PRECISION.
PRECISION = globals().get("PRECISION", os.environ.get("QE_PRECISION", "double"))
jax.config.update("jax_enable_x64", PRECISION == "double")
RDTYPE = jnp.float64 if PRECISION == "double" else jnp.float32       # real dtype
CDTYPE = jnp.complex128 if PRECISION == "double" else jnp.complex64  # complex dtype of all states/operators
TOL = 1e-10 if PRECISION == "double" else 1e-4                       # tolerance for sanity checks (asserts)
_LETTERS = string.ascii_letters  # 52 einsum labels; a call needs (axes + new labels) <= 52: apply_gate on a state
                                 # N + k <= 52 (N <= 51 for k = 1), apply_gate_dm 2N + k <= 52 (N <= 25 for k = 1, 2),
                                 # apply_kraus_dm 2N + 2k + 1 <= 52 (N <= 24 for k = 1, N <= 23 for k = 2)
# --- END PREAMBLE ---

# ==============================================================================
# 1. Elementary matrices: Paulis, Clifford generators, rotations
# ==============================================================================
# Basis convention: |0> = (1,0)^T is the +1 eigenstate of Z ("spin up"), |1> = (0,1)^T.
I2 = jnp.eye(2, dtype=CDTYPE)
X = jnp.array([[0, 1], [1, 0]], dtype=CDTYPE)
Y = jnp.array([[0, -1j], [1j, 0]], dtype=CDTYPE)
Z = jnp.array([[1, 0], [0, -1]], dtype=CDTYPE)
H = jnp.array([[1, 1], [1, -1]], dtype=CDTYPE) / jnp.sqrt(2.0)      # Hadamard: Z-basis <-> X-basis
S = jnp.array([[1, 0], [0, 1j]], dtype=CDTYPE)                      # phase gate, S^2 = Z
SDG = S.conj().T
T = jnp.array([[1, 0], [0, jnp.exp(1j * jnp.pi / 4)]], dtype=CDTYPE)  # pi/8 gate, T^2 = S (non-Clifford!)
TDG = T.conj().T
SM = jnp.array([[0, 1], [0, 0]], dtype=CDTYPE)  # sigma^- = |0><1| : lowers |1> -> |0>
SP = SM.conj().T                                # sigma^+ = |1><0|
P0 = jnp.array([[1, 0], [0, 0]], dtype=CDTYPE)  # projector |0><0|
P1 = jnp.array([[0, 0], [0, 1]], dtype=CDTYPE)  # projector |1><1|
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}

# Two-qubit matrices in the basis |00>,|01>,|10>,|11>  (FIRST listed qubit = left factor of kron)
CNOT = jnp.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=CDTYPE)
CZ = jnp.diag(jnp.array([1, 1, 1, -1], dtype=CDTYPE))
SWAP = jnp.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=CDTYPE)
XX = jnp.kron(X, X)
YY = jnp.kron(Y, Y)
ZZ = jnp.kron(Z, Z)


def rx(theta):
    """Rotation about x:  Rx(theta) = exp(-i theta X/2) = cos(theta/2) 1 - i sin(theta/2) X.

    Derivation: for any operator with P^2 = 1 the exponential series splits into even/odd
    parts,  exp(-i a P) = cos(a) 1 - i sin(a) P.   Here a = theta/2.
    `theta` may be a traced JAX scalar, so the gate is differentiable (jax.grad) and jit-able.
    """
    c, s = jnp.cos(theta / 2), jnp.sin(theta / 2)
    return jnp.array([[c, -1j * s], [-1j * s, c]], dtype=CDTYPE)


def ry(theta):
    """Rotation about y:  Ry(theta) = exp(-i theta Y/2) = [[cos, -sin], [sin, cos]](theta/2)  (real matrix)."""
    c, s = jnp.cos(theta / 2), jnp.sin(theta / 2)
    return jnp.array([[c, -s], [s, c]], dtype=CDTYPE)


def rz(theta):
    """Rotation about z:  Rz(theta) = exp(-i theta Z/2) = diag(e^{-i theta/2}, e^{+i theta/2})."""
    e = jnp.exp(-0.5j * theta)
    return jnp.array([[e, 0], [0, jnp.conj(e)]], dtype=CDTYPE)


def rpp(theta, PP):
    """Two-qubit Pauli rotation  exp(-i theta PP/2)  for any product PP of two Pauli matrices (XX, YY, ZZ, XY, ...).

    Because (P x Q)^2 = 1 the same closed form as for single-qubit rotations holds:
        exp(-i theta PP/2) = cos(theta/2) 1_4 - i sin(theta/2) PP.
    These are the native entangling gates of trapped ions (XX) and the building blocks
    of Trotterised spin-chain evolution.
    """
    return jnp.cos(theta / 2) * jnp.eye(4, dtype=CDTYPE) - 1j * jnp.sin(theta / 2) * PP


def rxx(theta):
    """exp(-i theta X(x)X / 2)."""
    return rpp(theta, XX)


def ryy(theta):
    """exp(-i theta Y(x)Y / 2)."""
    return rpp(theta, YY)


def rzz(theta):
    """exp(-i theta Z(x)Z / 2)  -- diagonal: diag(e^{-i t/2}, e^{+i t/2}, e^{+i t/2}, e^{-i t/2})."""
    return rpp(theta, ZZ)


def controlled(U):
    """Controlled-U = |0><0| (x) 1 + |1><1| (x) U  as a 4x4 block matrix (control = first qubit)."""
    return jnp.block([[jnp.eye(2, dtype=CDTYPE), jnp.zeros((2, 2), CDTYPE)],
                      [jnp.zeros((2, 2), CDTYPE), U]])


# ==============================================================================
# 2. THE core primitive: a k-qubit operator acting on a tensor through ONE einsum
# ==============================================================================
def apply_gate(psi, U, qubits):
    """Apply a k-qubit operator U to the axes `qubits` of the tensor `psi`.

    MATH
    ----
    One qubit (k=1), target q:
        psi'[s_0,..,a,..,s_{N-1}] = sum_b  U[a,b] psi[s_0,..,b,..,s_{N-1}]
    Two qubits (k=2), targets (q1,q2); the 4x4 matrix is reshaped to U[a1,a2,b1,b2]:
        psi'[.., a1, .., a2, ..] = sum_{b1,b2} U[a1,a2,b1,b2] psi[.., b1, .., b2, ..]
    This IS the action of (1 x .. x U x .. x 1) -- written without ever building it.

    EINSUM TRANSLATION  (generated automatically below)
    ------------------
        N=3, q=1        :  "db,abc->adc"          d = new (output) index of qubit 1
        N=4, q=(1,3)    :  "efbd,abcd->aecf"      e,f = new indices of qubits 1 and 3
    Recipe: label the axes of psi with letters a,b,c,...; give every target qubit a NEW
    letter (the next unused ones); U carries (new letters)+(old letters of the targets); the
    output is psi's labels with the target letters replaced by the new ones.

    IMPLEMENTATION NOTES
    --------------------
    * reshape of U:  (2^k,2^k) -> (2,)*2k.  Row index = (a1..ak), column = (b1..bk), and
      the FIRST qubit in `qubits` is the most significant bit of the small matrix, i.e. it
      corresponds to the LEFT factor of kron(A,B).
    * `qubits` may be any distinct axes in any order: non-neighbouring gates cost the same.
    * U need not be unitary: projectors, Kraus operators and Hamiltonian terms reuse this.
    * psi may be ANY tensor with those axes of size 2: a state (rank N), a density tensor
      (rank 2N: ket axes 0..N-1, bra axes N..2N-1), ...
    COST   O(2^k * 2^N) operations, O(2^N) memory  (dense matrix: O(4^N) both).
    JAX    `qubits` must be static Python ints (they define the einsum string); psi and U are
           traced, so the function can sit inside jit / grad / vmap / scan.
    """
    qubits = tuple(int(q) for q in qubits)
    k, n = len(qubits), psi.ndim
    U = jnp.asarray(U, dtype=psi.dtype).reshape((2,) * (2 * k))     # matrix -> rank-2k tensor
    inp = list(_LETTERS[:n])                                         # labels of psi's axes
    new = _LETTERS[n:n + k]                                          # fresh labels for the outputs
    out = inp.copy()
    for a, q in zip(new, qubits):
        out[q] = a                                                   # target axes get the new labels
    sub = f"{''.join(new)}{''.join(inp[q] for q in qubits)},{''.join(inp)}->{''.join(out)}"
    return jnp.einsum(sub, U, psi)


def apply_gate_dm(rho, U, qubits):
    """Unitary on a density tensor:  rho -> U rho U^dagger.

    MATH   rho'[a..;a'..] = sum_{b,b'} U[a,b] rho[b..;b'..] U^*[a',b']
           i.e. U acts on the KET axis q and the complex conjugate U^* on the BRA axis N+q.
           (A density matrix is "a state vector of 2N qubits": vectorisation |rho>> and
           U (x) U^* acting on it.)
    IMPLEMENTATION  two calls of `apply_gate` -- the same einsum machinery, no new code.
    COST   O(2^k 4^N).
    """
    N = rho.ndim // 2
    rho = apply_gate(rho, U, qubits)                                         # U    on ket axes
    return apply_gate(rho, jnp.conj(jnp.asarray(U)), [N + q for q in qubits])  # U^*  on bra axes


def apply_kraus_dm(rho, kraus, qubits):
    """Quantum channel in Kraus form:  rho -> sum_m K_m rho K_m^dagger,   sum_m K_m^dag K_m = 1.

    EINSUM TRANSLATION  (N=2, channel on qubit 0; g = Kraus index, summed!)
        "gea,gfc,abcd->ebfd"   with operands  K, K^*, rho      [a=ket0, b=ket1, c=bra0, d=bra1; e, f = new ket0, bra0]
    All Kraus branches are summed INSIDE one einsum -- no Python loop over m.
    `kraus` has shape (M, 2^k, 2^k).
    """
    qubits = tuple(int(q) for q in qubits)
    k, N = len(qubits), rho.ndim // 2
    K = jnp.asarray(kraus, dtype=rho.dtype).reshape((-1,) + (2,) * (2 * k))
    n = rho.ndim
    inp = list(_LETTERS[:n])
    new_k, new_b, m = _LETTERS[n:n + k], _LETTERS[n + k:n + 2 * k], _LETTERS[n + 2 * k]
    out = inp.copy()
    for a, b, q in zip(new_k, new_b, qubits):
        out[q], out[N + q] = a, b
    ket_old = "".join(inp[q] for q in qubits)
    bra_old = "".join(inp[N + q] for q in qubits)
    sub = f"{m}{new_k}{ket_old},{m}{new_b}{bra_old},{''.join(inp)}->{''.join(out)}"
    return jnp.einsum(sub, K, jnp.conj(K), rho)


# ==============================================================================
# 3. State preparation
# ==============================================================================
def zero_state(N):
    """|00...0>: a rank-N tensor of zeros with a single 1 at index (0,0,...,0).
    JAX arrays are immutable: `.at[idx].set(v)` returns an updated COPY (functional style)."""
    return jnp.zeros((2,) * N, dtype=CDTYPE).at[(0,) * N].set(1.0)


_KETS = {"0": [1, 0], "1": [0, 1], "+": [1, 1], "-": [1, -1], "r": [1, 1j], "l": [1, -1j]}


def product_state(spec):
    """Product state from a string, one character per qubit:  product_state('01+-') = |0>|1>|+>|->.
    ('r','l' = the +1/-1 eigenstates of Y.)

    MATH    psi[s_0,...,s_{N-1}] = v0[s_0] v1[s_1] ... v_{N-1}[s_{N-1}]
    EINSUM  an outer product = einsum with NO summed index:  "a,b,c->abc".
    A product state needs only 2N numbers; entanglement is what makes all 2^N necessary.
    """
    vecs = [jnp.array(_KETS[c], dtype=CDTYPE) / jnp.linalg.norm(jnp.array(_KETS[c], dtype=CDTYPE))
            for c in spec]
    n = len(vecs)
    sub = ",".join(_LETTERS[i] for i in range(n)) + "->" + _LETTERS[:n]
    return jnp.einsum(sub, *vecs)


def basis_state(bits):
    """Computational basis state |b_0 b_1 ... b_{N-1}> from a list of bits."""
    bits = tuple(int(b) for b in bits)
    return jnp.zeros((2,) * len(bits), dtype=CDTYPE).at[bits].set(1.0)


def ghz_state(N):
    """GHZ state (|0...0> + |1...1>)/sqrt(2): two non-zero entries of the rank-N tensor."""
    psi = jnp.zeros((2,) * N, dtype=CDTYPE)
    return psi.at[(0,) * N].set(1 / jnp.sqrt(2.0)).at[(1,) * N].set(1 / jnp.sqrt(2.0))


def ghz_circuit(N):
    """The same GHZ state, prepared the way hardware does it: H on qubit 0, then a CNOT cascade.
        |0..0> -H-> (|0>+|1>)|0..0>/sqrt2 -CNOTs-> (|0..0>+|1..1>)/sqrt2."""
    psi = apply_gate(zero_state(N), H, [0])
    for q in range(1, N):
        psi = apply_gate(psi, CNOT, [q - 1, q])
    return psi


def dicke_state(N, k):
    """Symmetric Dicke state |D_N^k>: equal superposition of all basis states with exactly k ones.
    Implementation: mask of the flat indices with popcount == k, normalised by sqrt(binom(N,k))."""
    idx = np.arange(2 ** N)
    pop = np.array([bin(i).count("1") for i in idx])
    v = (pop == k).astype(np.complex128)
    return jnp.asarray(v / np.linalg.norm(v), dtype=CDTYPE).reshape((2,) * N)


def w_state(N):
    """W state = Dicke state with one excitation: (|10..0> + |01..0> + ... + |0..01>)/sqrt(N)."""
    return dicke_state(N, 1)


def cluster_state(N, periodic=False):
    """1D cluster (graph) state: |+>^N followed by CZ on every bond.  Stabilisers K_i = Z_{i-1} X_i Z_{i+1}."""
    psi = product_state("+" * N)
    for q in range(N - 1):
        psi = apply_gate(psi, CZ, [q, q + 1])
    if periodic and N > 2:
        psi = apply_gate(psi, CZ, [N - 1, 0])
    return psi


def bell_state(kind="phi+"):
    """The four Bell states as (2,2) tensors: phi+- = (|00> +- |11>)/sqrt2, psi+- = (|01> +- |10>)/sqrt2."""
    a = 1 / np.sqrt(2)
    v = {"phi+": [a, 0, 0, a], "phi-": [a, 0, 0, -a], "psi+": [0, a, a, 0], "psi-": [0, a, -a, 0]}[kind]
    return jnp.array(v, dtype=CDTYPE).reshape(2, 2)


def haar_state(key, N):
    """Haar-random pure state: a complex Gaussian vector, normalised.
    (The Gaussian measure is unitarily invariant, so the direction is uniformly distributed.)
    `key` is an explicit JAX PRNG key -- same key, same state: reproducible science."""
    k1, k2 = jax.random.split(key)
    v = jax.random.normal(k1, (2,) * N) + 1j * jax.random.normal(k2, (2,) * N)
    return (v / jnp.linalg.norm(v)).astype(CDTYPE)


def to_dm(psi):
    """Pure state -> density TENSOR  rho[s..; s'..] = psi[s..] psi^*[s'..]   (rank 2N).
    An outer product: einsum "ab,AB->abAB" -- here via the equivalent tensordot(axes=0)."""
    return jnp.tensordot(psi, jnp.conj(psi), axes=0)


def dm_matrix(rho):
    """View a rank-2N density tensor as an ordinary 2^N x 2^N matrix (a reshape, no copy of meaning)."""
    d = int(round(np.sqrt(rho.size)))
    return rho.reshape(d, d)


def normalize(psi):
    """psi / ||psi||  (Frobenius norm of the tensor = 2-norm of the state vector)."""
    return psi / jnp.linalg.norm(psi)


# ==============================================================================
# 4. Reduced density matrices & observables (matrix-free)
# ==============================================================================
def rdm(psi, keep):
    """Reduced density matrix of the qubits `keep` from a PURE state:  rho_A = Tr_B |psi><psi|.

    MATH
        rho_A[a, a'] = sum_{b}  psi[a, b] psi^*[a', b]          (a: indices of A, b: of the rest B)
    EINSUM TRANSLATION   N=3, keep=(0,):   "abc,Abc->aA"   with operands psi, psi^*
        * letters of B (b,c) appear in BOTH operands and not in the output  -> summed = traced out
        * the kept axis gets a different letter in the second operand       -> stays open (a and A)
    The full |psi><psi| (4^N numbers) is never formed; cost O(2^N 2^|A|).
    Returns a (2^|A|, 2^|A|) matrix with row/column order following `keep`.
    """
    keep = tuple(int(q) for q in keep)
    n, k = psi.ndim, len(keep)
    a = list(_LETTERS[:n])
    b = a.copy()
    for new, q in zip(_LETTERS[n:n + k], keep):
        b[q] = new
    sub = f"{''.join(a)},{''.join(b)}->{''.join(a[q] for q in keep)}{''.join(b[q] for q in keep)}"
    return jnp.einsum(sub, psi, jnp.conj(psi)).reshape(2 ** k, 2 ** k)


def rdm_dm(rho, keep):
    """Partial trace of a density TENSOR (rank 2N), keeping the qubits `keep`.

    MATH      rho_A[a,a'] = sum_b rho[a,b ; a',b]
    EINSUM    N=2, keep=(0,):  "abAb->aA"   -- the letter b is REPEATED inside one operand and absent
              from the output: that is a trace over that pair of axes.
    """
    keep = tuple(int(q) for q in keep)
    N = rho.ndim // 2
    ket = list(_LETTERS[:N])
    bra = [(_LETTERS[N + q] if q in keep else ket[q]) for q in range(N)]
    sub = f"{''.join(ket)}{''.join(bra)}->{''.join(ket[q] for q in keep)}{''.join(bra[q] for q in keep)}"
    return jnp.einsum(sub, rho).reshape(2 ** len(keep), 2 ** len(keep))


def expect_local(psi, O, qubits):
    """<psi| O_qubits |psi> for a k-local operator O (2^k x 2^k matrix).

    MATH   <O> = Tr(rho_A O)  with A = `qubits`: a local observable only sees the local state.
    Cost O(2^N) for the RDM + O(4^k) for the small trace.  One RDM serves ALL observables on A
    (e.g. <X>,<Y>,<Z> from a single 2x2 matrix).
    """
    return jnp.real(jnp.trace(rdm(psi, qubits) @ jnp.asarray(O, dtype=CDTYPE)))


def expect_local_dm(rho, O, qubits):
    """Tr(rho O_qubits) for a density tensor, via its reduced density matrix."""
    return jnp.real(jnp.trace(rdm_dm(rho, qubits) @ jnp.asarray(O, dtype=CDTYPE)))


def apply_pauli_string(psi, ops):
    """Apply a Pauli string P = P_0 x P_1 x ... to psi.
    `ops` is a dict {qubit: 'X'|'Y'|'Z'} or a string like 'XIZY' (one letter per qubit; 'I' skipped).
    N single-qubit einsums -- cost O(N 2^N) -- instead of one 2^N x 2^N matrix."""
    if isinstance(ops, str):
        ops = {q: p for q, p in enumerate(ops) if p != "I"}
    for q, p in ops.items():
        psi = apply_gate(psi, PAULI[p], [q])
    return psi


def expect_pauli_string(psi, ops):
    """<psi|P|psi> = <psi | (P psi)> : apply the string, then ONE inner product (jnp.vdot conjugates its
    first argument).  Works for strings of any weight, e.g. the N-body correlator <X X ... X>."""
    return jnp.real(jnp.vdot(psi, apply_pauli_string(psi, ops)))


def expect_pauli_string_dm(rho, ops):
    """Tr(P rho): apply P to the KET axes only, then take the full trace."""
    return jnp.real(jnp.trace(dm_matrix(apply_pauli_string(rho, ops))))


def all_local_expectations(psi):
    """Bloch vectors of all qubits: array (N,3) with rows (<X_q>, <Y_q>, <Z_q>), from the 1-qubit RDMs."""
    out = []
    for q in range(psi.ndim):
        r = rdm(psi, [q])
        out.append(jnp.stack([jnp.real(jnp.trace(r @ P)) for P in (X, Y, Z)]))
    return jnp.stack(out)


# ==============================================================================
# 5. State characteristics: entropies, fidelity, negativity
# ==============================================================================
def purity(rho_mat):
    """Tr(rho^2): 1 for pure states, 1/d for the maximally mixed state in dimension d."""
    return jnp.real(jnp.trace(rho_mat @ rho_mat))


def von_neumann_entropy(rho_mat, base=2.0):
    """S(rho) = -Tr(rho log rho) = -sum_i lambda_i log lambda_i  (in bits for base=2).
    Eigenvalues are clipped at 1e-16 because 0*log(0) := 0 but log(0) = -inf numerically."""
    lam = jnp.clip(jnp.linalg.eigvalsh(rho_mat), 1e-16, None)
    return -jnp.sum(lam * jnp.log(lam)) / jnp.log(base)


def schmidt_values(psi, subsystem):
    """Schmidt coefficients of the bipartition A|B with A = `subsystem`.

    MATH   Group the tensor axes into a matrix  M[(a),(b)] = psi[a,b]  of shape (2^|A|, 2^|B|).
           Its SVD  M = U diag(lambda) V^dag  IS the Schmidt decomposition
               |psi> = sum_k lambda_k |u_k>_A |v_k>_B ,     rho_A has eigenvalues lambda_k^2.
    IMPLEMENTATION   transpose (A axes first) -> reshape -> svd.   No density matrix needed.
    """
    A = tuple(int(q) for q in subsystem)
    B = tuple(q for q in range(psi.ndim) if q not in A)
    M = jnp.transpose(psi, A + B).reshape(2 ** len(A), -1)
    return jnp.linalg.svd(M, compute_uv=False)


def entanglement_entropy(psi, subsystem, base=2.0):
    """Entanglement entropy S_A = -sum_k p_k log p_k with p_k = lambda_k^2 (squared Schmidt values).
    Product state: 0.  Bell/GHZ across any cut: 1 bit.  Random state: ~ |A| - 1/(2 ln2) 2^{|A|-|B|} (Page)."""
    p = jnp.clip(schmidt_values(psi, subsystem) ** 2, 1e-16, None)
    return -jnp.sum(p * jnp.log(p)) / jnp.log(base)


def renyi2_entropy(psi, subsystem, base=2.0):
    """Second Renyi entropy S_2 = -log Tr(rho_A^2) = -log sum_k lambda_k^4  (experimentally accessible)."""
    p = schmidt_values(psi, subsystem) ** 2
    return -jnp.log(jnp.sum(p ** 2)) / jnp.log(base)


def fidelity_pure(psi, phi):
    """F = |<psi|phi>|^2 between two pure states."""
    return jnp.abs(jnp.vdot(psi, phi)) ** 2


def _sqrtm_psd(A):
    """Square root of a positive semi-definite matrix via eigendecomposition: V sqrt(w) V^dag."""
    w, v = jnp.linalg.eigh(A)
    return (v * jnp.sqrt(jnp.clip(w, 0, None))) @ v.conj().T


def fidelity_dm(rho, sigma):
    """Uhlmann fidelity F(rho,sigma) = (Tr sqrt( sqrt(rho) sigma sqrt(rho) ))^2 for dense matrices."""
    s = _sqrtm_psd(rho)
    w = jnp.linalg.eigvalsh(s @ sigma @ s)
    return jnp.sum(jnp.sqrt(jnp.clip(w, 0, None))) ** 2


def trace_distance(rho, sigma):
    """D = (1/2)||rho - sigma||_1 = (1/2) sum_i |eigenvalue_i(rho - sigma)|  (Hermitian difference)."""
    return 0.5 * jnp.sum(jnp.abs(jnp.linalg.eigvalsh(rho - sigma)))


def as_dm_tensor(rho):
    """Accept a density matrix in EITHER form and return the rank-2N tensor (2,)*2N.

    Both conventions are in use: `rdm` and `dm_matrix` return a (2^N, 2^N) MATRIX, while
    `to_dm` and the channel functions work with the rank-2N TENSOR (N ket axes, then N bra axes).
    A (2^N, 2^N) matrix with N > 1 is unambiguous, because a density tensor has every axis of length 2.
    Passing the matrix form straight to `partial_transpose` used to transpose the WHOLE matrix, which
    leaves the eigenvalues alone and silently reports zero negativity -- hence this conversion.
    """
    rho = jnp.asarray(rho)
    if rho.ndim == 2 and rho.shape[0] > 2:                  # (2^N, 2^N) matrix with N > 1
        d = rho.shape[0]
        N = int(round(math.log2(d)))
        if 2 ** N != d:
            raise ValueError(f"density matrix of dimension {d} is not a power of two")
        return rho.reshape((2,) * (2 * N))
    if any(s != 2 for s in rho.shape) or rho.ndim % 2:
        raise ValueError(f"not a density matrix or density tensor: shape {rho.shape}")
    return rho


def partial_transpose(rho, qubits):
    """Partial transpose on subsystem A = `qubits`, for a density tensor OR a (2^N, 2^N) matrix.
    In tensor form it is trivial: SWAP the ket axis q with the bra axis N+q for q in A.
    (With a 2^N x 2^N matrix this requires painful index gymnastics -- here: one `transpose`.)
    RETURNS the rank-2N tensor; wrap in `dm_matrix` for the matrix form."""
    rho = as_dm_tensor(rho)
    N = rho.ndim // 2
    perm = list(range(2 * N))
    for q in qubits:
        perm[q], perm[N + q] = perm[N + q], perm[q]
    return jnp.transpose(rho, perm)


def negativity(rho, qubits):
    """Entanglement negativity (Peres-Horodecki / PPT criterion).
        N(rho) = (||rho^{T_A}||_1 - 1)/2 = sum of |negative eigenvalues| of the partial transpose,
        E_N    = log2 ||rho^{T_A}||_1 = log2(2N+1)      (logarithmic negativity).
    N > 0  =>  entangled (for 2x2 and 2x3 systems also <=).   Returns (N, E_N)."""
    lam = jnp.linalg.eigvalsh(dm_matrix(partial_transpose(rho, qubits)))
    neg = jnp.sum(jnp.abs(lam) - lam) / 2
    return neg, jnp.log2(2 * neg + 1)


# ==============================================================================
# 6. Measurements (Born rule, collapse, sampling)
# ==============================================================================
_BASIS_ROT = {"Z": I2, "X": H, "Y": H @ SDG}  # U maps the eigenbasis of the Pauli to |0>,|1>:  P = U^dag Z U


def measure_qubit(key, psi, q, basis="Z"):
    """Projective measurement of qubit q in the eigenbasis of X, Y or Z.  Returns (outcome, post-state).

    MATH   Born rule      p(0) = <psi|P0_q|psi> = (rho_q)[0,0]          (only the 1-qubit RDM is needed)
           collapse       |psi> -> P_m|psi> / sqrt(p(m))
           other bases    measuring P = U^dag Z U  ==  rotate with U, measure Z, rotate back with U^dag
                          (X: U = H;   Y: U = H S^dag).
    outcome 0 <-> eigenvalue +1,   outcome 1 <-> eigenvalue -1.
    JAX    no Python `if` on random data: `jnp.where` selects the projector, so the function is
           jit/vmap-able (vmap over keys = many independent shots).
    """
    U = _BASIS_ROT[basis]
    psi = apply_gate(psi, U, [q])
    p0 = jnp.real(rdm(psi, [q])[0, 0])
    outcome = (jax.random.uniform(key) >= p0).astype(jnp.int32)
    proj = jnp.where(outcome == 0, P0, P1)
    psi = apply_gate(psi, proj, [q])
    psi = psi / jnp.linalg.norm(psi)
    return outcome, apply_gate(psi, U.conj().T, [q])


def reset_qubit(key, psi, q):
    """Active reset: measure qubit q in Z and apply X if the outcome was 1 -> qubit ends in |0>,
    disentangled from the rest (used in teleportation-like protocols, autoencoders, reservoirs)."""
    outcome, psi = measure_qubit(key, psi, q)
    return apply_gate(psi, jnp.where(outcome == 1, X, I2), [q])


def sample_bitstrings(key, psi, shots, bases=None):
    """Sample `shots` outcomes of measuring ALL qubits (no collapse bookkeeping needed).

    MATH   p(s) = |psi[s]|^2 ; optional per-qubit bases string like 'XZY' rotates first.
    IMPLEMENTATION   `jax.random.categorical` draws flat indices from log-probabilities; the bits are
    unpacked with shifts:  s_q = (i >> (N-1-q)) & 1   (qubit 0 = most significant bit).
    Returns an int array of shape (shots, N).
    """
    N = psi.ndim
    if bases is not None:
        for q, b in enumerate(bases):
            psi = apply_gate(psi, _BASIS_ROT[b], [q])
    logp = jnp.log(jnp.clip(jnp.abs(psi.reshape(-1)) ** 2, 1e-300, None))
    idx = jax.random.categorical(key, logp, shape=(shots,))
    return (idx[:, None] >> jnp.arange(N - 1, -1, -1)[None, :]) & 1


# ==============================================================================
# 7. Noise channels (Kraus operators): density matrix and quantum-trajectory forms
# ==============================================================================
def kraus_depolarizing(p):
    """Depolarising channel  rho -> (1-p) rho + (p/3)(X rho X + Y rho Y + Z rho Z).
    Kraus: sqrt(1-p) 1, sqrt(p/3) X, sqrt(p/3) Y, sqrt(p/3) Z.   Bloch vector shrinks by 1 - 4p/3."""
    return jnp.stack([jnp.sqrt(1 - p) * I2, jnp.sqrt(p / 3) * X, jnp.sqrt(p / 3) * Y, jnp.sqrt(p / 3) * Z])


def kraus_dephasing(p):
    """Dephasing (phase-flip) channel  rho -> (1-p) rho + p Z rho Z : coherences rho_01 -> (1-2p) rho_01."""
    return jnp.stack([jnp.sqrt(1 - p) * I2, jnp.sqrt(p) * Z])


def kraus_bit_flip(p):
    """Bit-flip channel  rho -> (1-p) rho + p X rho X."""
    return jnp.stack([jnp.sqrt(1 - p) * I2, jnp.sqrt(p) * X])


def kraus_phase_flip(p):
    """Phase-flip channel (identical to dephasing)."""
    return kraus_dephasing(p)


def kraus_amplitude_damping(g):
    """Amplitude damping (T1 decay |1> -> |0> with probability g):
        K0 = [[1,0],[0,sqrt(1-g)]]   (no decay -- but the amplitude of |1> shrinks: 'no click' is information)
        K1 = [[0,sqrt(g)],[0,0]]     (decay event = sqrt(g) sigma^-)."""
    K0 = jnp.array([[1, 0], [0, jnp.sqrt(1 - g)]], dtype=CDTYPE)
    K1 = jnp.array([[0, jnp.sqrt(g)], [0, 0]], dtype=CDTYPE)
    return jnp.stack([K0, K1])


def kraus_from_jump(L, gamma_dt):
    """Kraus pair generated by a Lindblad jump operator L acting for a short time dt (rate gamma):
        K1 = sqrt(gamma dt) L                      'a jump happened'
        K0 = sqrt(1 - gamma dt L^dag L)            'no jump'  (~ exp(-gamma dt L^dag L / 2))
    Completeness K0^dag K0 + K1^dag K1 = 1 holds EXACTLY whenever gamma dt ||L^dag L|| <= 1, so that
    1 - gamma dt L^dag L is positive semidefinite and its square root exists; the map is then trace preserving.
    For larger steps the negative eigenvalues are clipped by _sqrtm_psd and completeness FAILS (L = sigma^-,
    gamma dt = 1.5: error 0.5).  It reproduces the Lindblad dissipator up to O(dt^2)."""
    L = jnp.asarray(L, dtype=CDTYPE)
    K0 = _sqrtm_psd(jnp.eye(L.shape[0], dtype=CDTYPE) - gamma_dt * (L.conj().T @ L))
    return jnp.stack([K0, jnp.sqrt(gamma_dt) * L])


def apply_kraus_mcwf(key, psi, kraus, qubits):
    """Apply a channel to a PURE state stochastically (quantum trajectory / Monte-Carlo wave function).

    MATH   choose branch m with probability  p_m = ||K_m psi||^2 = Tr(K_m rho_A K_m^dag)
           then  |psi> -> K_m|psi> / sqrt(p_m).
           Averaging |psi><psi| over many trajectories gives exactly  sum_m K_m rho K_m^dag.
    WHY    memory O(2^N) instead of O(4^N): open systems at the price of closed ones (times #trajectories,
           which are embarrassingly parallel -> jax.vmap over keys).
    IMPLEMENTATION   probabilities come from the SMALL reduced density matrix (einsum "mab,bc,mac->m");
           `K[m]` with a traced integer m is a gather, so everything stays jit-compatible.
    """
    K = jnp.asarray(kraus, dtype=CDTYPE)
    r = rdm(psi, qubits)
    probs = jnp.real(jnp.einsum("mab,bc,mac->m", K, r, jnp.conj(K)))
    m = jax.random.categorical(key, jnp.log(jnp.clip(probs, 1e-300, None)))
    psi = apply_gate(psi, K[m], qubits)
    return psi / jnp.linalg.norm(psi)


# ==============================================================================
# 8. Hamiltonians as lists of local terms;  H|psi>  matrix-free
# ==============================================================================
def heisenberg_terms(N, Jxx=1.0, Jyy=1.0, Jzz=1.0, hx=0.0, hy=0.0, hz=0.0, periodic=False, bonds=None):
    """Spin-1/2 Hamiltonian in Pauli convention
        H = sum_<ij> (Jxx X_i X_j + Jyy Y_i Y_j + Jzz Z_i Z_j) + sum_i (hx X_i + hy Y_i + hz Z_i)
    returned as a LIST of local terms [(qubits, small_matrix), ...]  -- this list *is* our Hamiltonian.
      Heisenberg: Jxx=Jyy=Jzz     XXZ: Jxx=Jyy!=Jzz     XY: Jzz=0     TFIM: only Jzz and hx.
    `bonds` overrides the default open chain (any graph: ladders, 2D lattices, all-to-all, ...)."""
    if bonds is None:
        bonds = [(i, i + 1) for i in range(N - 1)] + ([(N - 1, 0)] if periodic and N > 2 else [])
    terms = [((i, j), Jxx * XX + Jyy * YY + Jzz * ZZ) for (i, j) in bonds]
    if hx != 0 or hy != 0 or hz != 0:
        terms += [((i,), hx * X + hy * Y + hz * Z) for i in range(N)]
    return terms


def apply_hamiltonian(terms, psi):
    """H|psi> = sum_k h_k|psi>, every term applied with `apply_gate` (h_k is Hermitian, not unitary --
    the einsum does not care).  Cost O(#terms * 2^N).  This matrix-vector product is ALL that Lanczos,
    Chebyshev and Krylov methods need, which is why they scale to N ~ 25-30."""
    out = jnp.zeros_like(psi)
    for qubits, h in terms:
        out = out + apply_gate(psi, h, qubits)
    return out


def energy(terms, psi):
    """<psi|H|psi> = Re <psi | (H psi)>."""
    return jnp.real(jnp.vdot(psi, apply_hamiltonian(terms, psi)))


def dense_hamiltonian(terms, N):
    """Dense 2^N x 2^N matrix for VALIDATION on small N -- built from the matrix-free action:
    column j of H is H|e_j>.   `jax.vmap` applies H to all basis vectors at once."""
    eye = jnp.eye(2 ** N, dtype=CDTYPE).reshape((2 ** N,) + (2,) * N)
    cols = jax.vmap(lambda e: apply_hamiltonian(terms, e).reshape(-1))(eye)
    return cols.T


# ==============================================================================
# 9. Time evolution: TEBD / Trotter-Suzuki, Chebyshev, Krylov, exact
# ==============================================================================
def _expm_herm(h, tau):
    """exp(-i tau h) for a SMALL Hermitian matrix: h = V w V^dag  =>  V exp(-i tau w) V^dag."""
    w, v = jnp.linalg.eigh(jnp.asarray(h, dtype=CDTYPE))
    return (v * jnp.exp(-1j * tau * w)) @ v.conj().T


def tebd_gates(terms, dt, order=2):
    """Gate sequence of ONE Trotter-Suzuki (TEBD) step for H = sum_k h_k.

    MATH   exp(-i H dt) cannot be split exactly because [h_k, h_l] != 0, but (BCH formula):
        order 1:  prod_k e^{-i h_k dt}                                       local error O(dt^2), global O(dt)
        order 2:  prod_k e^{-i h_k dt/2} * prod_k(reversed) e^{-i h_k dt/2}  local O(dt^3), global O(dt^2)
        order 4:  S2(s dt)^2 S2((1-4s) dt) S2(s dt)^2,  s = 1/(4 - 4^{1/3})  (Suzuki)   global O(dt^4)
    Each factor is a 2x2 or 4x4 unitary -> applied by einsum; this is TEBD on a full state vector.
    Every order is exactly unitary: the norm is conserved to machine precision; the ERROR is in the phase
    /direction of the state, and it is controlled by dt.
    """
    if order == 1:
        return [(q, _expm_herm(h, dt)) for q, h in terms]
    if order == 2:
        half = [(q, _expm_herm(h, dt / 2)) for q, h in terms]
        return half + half[::-1]
    if order == 4:
        s = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
        seq = []
        for w in (s, s, 1 - 4 * s, s, s):
            seq += tebd_gates(terms, w * dt, order=2)
        return seq
    raise ValueError("order must be 1, 2 or 4")


def apply_gates(psi, gates):
    """Apply a list [(qubits, U), ...] in order -- a quantum circuit is a Python loop over einsums.
    Under `jax.jit` the loop is unrolled at trace time and XLA fuses it into one program."""
    for qubits, U in gates:
        psi = apply_gate(psi, U, qubits)
    return psi


def apply_gates_dm(rho, gates):
    """Same circuit on a density tensor (U on kets, U^* on bras)."""
    for qubits, U in gates:
        rho = apply_gate_dm(rho, U, qubits)
    return rho


def tebd_evolve(psi, terms, dt, n_steps, order=2, observe=None):
    """Evolve for n_steps*dt with TEBD and record `observe(psi)` after every step.

    JAX    `lax.scan(step, carry, xs, length)` compiles the step ONCE and loops inside XLA
           (a Python for-loop under jit would unroll n_steps copies of the graph).
           Returns (final_state, stacked_observables).
    """
    gates = tebd_gates(terms, dt, order)

    def step(psi, _):
        psi = apply_gates(psi, gates)
        return psi, (observe(psi) if observe is not None else 0.0)

    return lax.scan(step, psi, None, length=n_steps)


def lanczos(matvec, v0, m=60, reorth=True):
    """m-step Lanczos iteration: orthonormal basis V of the Krylov space span{v, Hv, H^2 v, ...}
    in which H is TRIDIAGONAL:   T = V^dag H V = tridiag(beta, alpha, beta).

    Recurrence     w = H v_j - alpha_j v_j - beta_{j-1} v_{j-1},   alpha_j = <v_j|H|v_j>,   beta_j = ||w||.
    `reorth=True`  re-orthogonalises against all previous vectors (cures the loss of orthogonality of
                   floating-point Lanczos at the price of O(m^2) inner products).
    Only `matvec` (H|psi>) is needed -> matrix-free.  Returns (alphas, betas, V[m, 2,..,2]).
    """
    v = v0 / jnp.linalg.norm(v0)
    V, alphas, betas = [v], [], []
    w = matvec(v)
    for j in range(m):
        a = jnp.real(jnp.vdot(V[j], w))
        alphas.append(a)
        w = w - a * V[j] - (betas[-1] * V[j - 1] if j > 0 else 0.0)
        if reorth:
            for u in V:
                w = w - jnp.vdot(u, w) * u
        b = jnp.linalg.norm(w)
        if j == m - 1 or float(b) < 1e-12:
            break
        betas.append(b)
        V.append(w / b)
        w = matvec(V[-1])
    return np.array(alphas, dtype=float), np.array(betas, dtype=float), jnp.stack(V)


def lanczos_ground_state(terms, N, m=80, key=None, restarts=3):
    """Ground state (E_0, |psi_0>) from restarted Lanczos: diagonalise the small T, map its lowest
    eigenvector back with V, use it as the next start vector.  Extremal eigenvalues converge first."""
    key = jax.random.PRNGKey(0) if key is None else key
    v = haar_state(key, N)
    mv = jax.jit(lambda p: apply_hamiltonian(terms, p))
    E = None
    for _ in range(restarts):
        a, b, V = lanczos(mv, v, m)
        w, S_ = np.linalg.eigh(np.diag(a) + np.diag(b, 1) + np.diag(b, -1))
        v = jnp.tensordot(jnp.asarray(S_[:, 0], dtype=CDTYPE), V, axes=1)
        v = v / jnp.linalg.norm(v)
        E = w[0]
    return float(E), v


def spectral_bounds(terms, N, m=40, key=None):
    """Cheap estimate of (E_min, E_max) from a short Lanczos run -- needed to rescale H into [-1,1]
    for the Chebyshev expansion."""
    key = jax.random.PRNGKey(1) if key is None else key
    a, b, _ = lanczos(jax.jit(lambda p: apply_hamiltonian(terms, p)), haar_state(key, N), m)
    w = np.linalg.eigvalsh(np.diag(a) + np.diag(b, 1) + np.diag(b, -1))
    return float(w[0]), float(w[-1])


def chebyshev_evolve(psi, terms, t, bounds, tol=1e-13, safety=1.02):
    """exp(-iHt)|psi> by Chebyshev expansion (Tal-Ezer & Kosloff) -- no Trotter error.

    MATH   rescale  Ht = (H - b)/a  with a = (Emax-Emin)/2, b = (Emax+Emin)/2   => spectrum in [-1,1]
           Jacobi-Anger:   e^{-i a t x} = sum_k (2 - delta_k0) (-i)^k J_k(a t) T_k(x)
           =>  e^{-iHt}|psi> = e^{-i b t} sum_k c_k |phi_k>,    c_k = (2-delta_k0)(-i)^k J_k(a t)
           recurrence     |phi_0> = |psi>, |phi_1> = Ht|psi>, |phi_{k+1}> = 2 Ht|phi_k> - |phi_{k-1}>.
    CONVERGENCE   J_k(at) decays super-exponentially for k > a t: the number of terms is ~ a t + O(10),
           and the error can be pushed to machine precision.  Memory: 3 vectors.
    JAX    the recurrence is a `lax.scan` whose carry is (phi_{k-1}, phi_k, accumulated sum).
    """
    from scipy.special import jv
    Emin, Emax = bounds
    a, b = safety * (Emax - Emin) / 2.0, (Emax + Emin) / 2.0
    K = int(np.ceil(a * abs(t))) + 20
    while abs(jv(K, a * t)) > tol:
        K += 10
    coef = jnp.asarray([(1 if k == 0 else 2) * ((-1j) ** k) * jv(k, a * t) for k in range(K + 1)], dtype=CDTYPE)

    def Ht(p):
        return (apply_hamiltonian(terms, p) - b * p) / a

    T0, T1 = psi, Ht(psi)
    acc0 = coef[0] * T0 + coef[1] * T1

    def body(carry, c):
        Tm, Tc, acc = carry
        Tn = 2 * Ht(Tc) - Tm
        return (Tc, Tn, acc + c * Tn), None

    (_, _, acc), _ = lax.scan(body, (T0, T1, acc0), coef[2:])
    return jnp.exp(-1j * b * t) * acc


def krylov_evolve(psi, terms, t, m=30):
    """exp(-iHt)|psi> in an m-dimensional Krylov space:
        |psi(t)> ~ ||psi|| * V exp(-i T t) e_1 ,      T = V^dag H V  (tridiagonal, from Lanczos).
    Exact for polynomials of degree < m in H; accurate as long as  |t| * (spectral width)  <~  m.
    For long times: chain several short Krylov steps."""
    nrm = jnp.linalg.norm(psi)
    a, b, V = lanczos(jax.jit(lambda p: apply_hamiltonian(terms, p)), psi, m)
    w, S_ = np.linalg.eigh(np.diag(a) + np.diag(b, 1) + np.diag(b, -1))
    c = S_ @ (np.exp(-1j * t * w) * S_[0, :])
    return nrm * jnp.tensordot(jnp.asarray(c, dtype=CDTYPE), V, axes=1)


def exact_evolve(psi, terms, t):
    """Dense reference for small N: full diagonalisation H = V w V^dag, |psi(t)> = V e^{-i w t} V^dag |psi>.
    O(8^N) time, O(4^N) memory -- the wall that matrix-free methods avoid; used only to validate them."""
    N = psi.ndim
    w, v = jnp.linalg.eigh(dense_hamiltonian(terms, N))
    return ((v * jnp.exp(-1j * t * w)) @ (v.conj().T @ psi.reshape(-1))).reshape((2,) * N)


# ==============================================================================
# 10. Open systems: Lindblad master equation (density tensor) and quantum trajectories (MCWF)
# ==============================================================================
def lindblad_rhs(rho, terms, jumps):
    """Right-hand side of the Lindblad (GKSL) master equation, matrix-free:
        d rho/dt = -i[H, rho] + sum_j gamma_j ( L_j rho L_j^dag - (1/2){L_j^dag L_j, rho} ).
    `jumps` = [(qubits, L, gamma), ...].

    KEY TRICK   left multiplication  O rho  acts on KET axes with O;
                right multiplication rho O  acts on BRA axes with O^T     [ (rho O)_{ab} = sum_c rho_{ac} O_{cb} ].
                So every term is again `apply_gate` on a rank-2N tensor.
    """
    N = rho.ndim // 2
    out = jnp.zeros_like(rho)
    for q, h in terms:
        bq = [N + x for x in q]
        out = out - 1j * (apply_gate(rho, h, q) - apply_gate(rho, jnp.asarray(h).T, bq))
    for q, L, g in jumps:
        L = jnp.asarray(L, dtype=CDTYPE)
        bq = [N + x for x in q]
        LdL = L.conj().T @ L
        out = out + g * (apply_gate(apply_gate(rho, L, q), jnp.conj(L), bq)
                         - 0.5 * apply_gate(rho, LdL, q) - 0.5 * apply_gate(rho, LdL.T, bq))
    return out


def lindblad_rk4_step(rho, terms, jumps, dt):
    """Classical 4th-order Runge-Kutta step for d rho/dt = L(rho): local error O(dt^5).
    TRACE   conserved to round-off: every term of the Lindblad generator is traceless and the step is a linear
            combination of generator evaluations.  This holds even beyond the stability limit, so a correct trace
            does NOT certify a healthy run.
    POSITIVITY   not guaranteed: small negative eigenvalues appear for moderate dt, and the run blows up once
            dt * ||L|| exceeds the RK4 stability bound."""
    f = lambda r: lindblad_rhs(r, terms, jumps)
    k1 = f(rho)
    k2 = f(rho + 0.5 * dt * k1)
    k3 = f(rho + 0.5 * dt * k2)
    k4 = f(rho + dt * k3)
    return rho + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def lindblad_trotter_step_dm(rho, gates, jump_kraus):
    """Trotterised Lindblad step on a density tensor: unitary TEBD gates, then one Kraus channel per
    jump operator (`jump_kraus` = [(qubits, kraus_from_jump(L, gamma*dt)), ...]).
    Completely positive and trace preserving by construction, first order in the dissipator."""
    rho = apply_gates_dm(rho, gates)
    for q, K in jump_kraus:
        rho = apply_kraus_dm(rho, K, q)
    return rho


def lindblad_trotter_step_mcwf(key, psi, gates, jump_kraus):
    """The SAME step unravelled on a pure state (Monte-Carlo wave function): each Kraus channel is
    sampled with `apply_kraus_mcwf`.  The trajectory average equals `lindblad_trotter_step_dm` exactly,
    so DM-vs-MCWF comparisons test only the statistical error ~ 1/sqrt(n_traj).
    JAX: scan over time steps, vmap over trajectory keys."""
    psi = apply_gates(psi, gates)
    keys = jax.random.split(key, max(len(jump_kraus), 1))
    for k, (q, K) in zip(keys, jump_kraus):
        psi = apply_kraus_mcwf(k, psi, K, q)
    return psi


# ==============================================================================
# 11. Collective spins, spin squeezing, quantum Fisher information
# ==============================================================================
def apply_collective(psi, P, qubits=None):
    """(sum_q P_q)|psi> for a single-qubit operator P -- N einsums, O(N 2^N).  Note: NO factor 1/2 here;
    the collective spin is J_a = (1/2) sum_q sigma^a_q."""
    qubits = range(psi.ndim) if qubits is None else qubits
    out = jnp.zeros_like(psi)
    for q in qubits:
        out = out + apply_gate(psi, P, [q])
    return out


def qfi_pure(psi, P=Z, qubits=None):
    """Quantum Fisher information of a pure state for the phase imprinted by exp(-i theta J),
    J = (1/2) sum_q P_q:
        F_Q = 4 Var(J) = <G^2> - <G>^2   with G = sum_q P_q = 2J.
    Cramer-Rao: Delta theta >= 1/sqrt(M F_Q).  Product states F_Q <= N (standard quantum limit);
    GHZ: F_Q = N^2 (Heisenberg limit).   Matrix-free: |phi> = G|psi>,  <G^2> = <phi|phi>,  <G> = <psi|phi>."""
    phi = apply_collective(psi, P, qubits)
    return jnp.real(jnp.vdot(phi, phi)) - jnp.real(jnp.vdot(psi, phi)) ** 2


def qfi_mixed(rho_mat, G_mat, tol=1e-12):
    """QFI of a MIXED state (symmetric-logarithmic-derivative formula), rho = sum_m l_m |m><m|:
        F_Q = 2 sum_{m,n : l_m+l_n>0} (l_m - l_n)^2/(l_m + l_n) |<m|G|n>|^2       (G = generator, e.g. J).
    Reduces to 4 Var(G) for pure states.  Dense eigendecomposition -> use on small / reduced states."""
    lam, v = jnp.linalg.eigh(rho_mat)
    G = v.conj().T @ G_mat @ v
    num = (lam[:, None] - lam[None, :]) ** 2
    den = lam[:, None] + lam[None, :]
    return 2 * jnp.sum(jnp.where(den > tol, num / jnp.where(den > tol, den, 1.0), 0.0) * jnp.abs(G) ** 2)


def collective_dense(P, n):
    """Dense J = (1/2) sum_q P_q on n qubits (input of `qfi_mixed`), built from the matrix-free action
    on the basis vectors (vmap), exactly like `dense_hamiltonian`."""
    eye = jnp.eye(2 ** n, dtype=CDTYPE).reshape((2 ** n,) + (2,) * n)
    return 0.5 * jax.vmap(lambda e: apply_collective(e, P).reshape(-1))(eye).T


def spin_moments(psi):
    """Mean collective spin <J_a> (shape 3) and symmetrised covariance matrix
        C_ab = (1/2)<J_a J_b + J_b J_a> - <J_a><J_b>          (shape 3x3).
    Trick: |phi_a> = J_a|psi>  =>  <J_a J_b> = <phi_a|phi_b>, and its real part is the symmetrised moment."""
    phis = [0.5 * apply_collective(psi, P) for P in (X, Y, Z)]
    mean = jnp.stack([jnp.real(jnp.vdot(psi, p)) for p in phis])
    second = jnp.stack([jnp.stack([jnp.real(jnp.vdot(a, b)) for b in phis]) for a in phis])
    return mean, second - jnp.outer(mean, mean)


def spin_squeezing(psi):
    """Wineland spin-squeezing parameter
        xi^2_R = N * min_{n_perp} Var(J_{n_perp}) / |<J>|^2 .
    xi^2 < 1: phase sensitivity beyond the standard quantum limit (and the state is entangled).
    Implementation: project the 3x3 covariance onto the plane perpendicular to the mean spin and take
    the smaller eigenvalue of the resulting 2x2 matrix."""
    N = psi.ndim
    mean, cov = spin_moments(psi)
    L = jnp.linalg.norm(mean)
    n = mean / L
    a = jnp.where(jnp.abs(n[2]) < 0.9, jnp.array([0.0, 0.0, 1.0]), jnp.array([1.0, 0.0, 0.0]))
    e1 = jnp.cross(n, a)
    e1 = e1 / jnp.linalg.norm(e1)
    e2 = jnp.cross(n, e1)
    B = jnp.stack([e1, e2])
    vmin = jnp.linalg.eigvalsh(B @ cov @ B.T)[0]
    return N * vmin / L ** 2


def oat_evolve(psi, chi_t):
    """One-axis twisting  exp(-i chi t J_z^2)  applied EXACTLY.
    J_z is diagonal in the computational basis with eigenvalue m(s) = sum_q (1/2 - s_q), so the evolution
    multiplies each amplitude by a phase exp(-i chi t m^2): O(2^N), no Trotter error, no gates.
    m(s) is assembled by BROADCASTING N arrays of shape (1,..,2,..,1) -- a tensor-network-free way of
    writing a sum over sites.   [Note: chi sum_{i<j} Z_i Z_j = 2 chi J_z^2 - chi N/2.]"""
    N = psi.ndim
    m = sum(jnp.array([0.5, -0.5]).reshape([2 if a == q else 1 for a in range(N)]) for q in range(N))
    return psi * jnp.exp(-1j * chi_t * m ** 2)


# ==============================================================================
# 12. Random unitaries, Clifford group, brick-wall circuits
# ==============================================================================
def haar_unitary(key, d):
    """Haar-random d x d unitary (Mezzadri's recipe): QR-decompose a complex Ginibre matrix and fix the
    phases of R's diagonal -- without the phase fix the distribution is NOT Haar."""
    k1, k2 = jax.random.split(key)
    G = (jax.random.normal(k1, (d, d)) + 1j * jax.random.normal(k2, (d, d))) / jnp.sqrt(2.0)
    Q, R = jnp.linalg.qr(G)
    ph = jnp.diagonal(R) / jnp.abs(jnp.diagonal(R))
    return Q * ph[None, :]


def brickwall(key, psi, depth, periodic=False):
    """Random brick-wall circuit: layers of Haar-random 2-qubit gates on even bonds, then odd bonds, ...
    Depth ~ N turns a product state into a (nearly) Haar-random state: volume-law entanglement (Page)."""
    N = psi.ndim
    for layer in range(depth):
        bonds = [(i, i + 1) for i in range(layer % 2, N - 1, 2)]
        if periodic and N % 2 == 0 and layer % 2 == 1:
            bonds.append((N - 1, 0))
        key, *ks = jax.random.split(key, len(bonds) + 1)
        for k, b in zip(ks, bonds):
            psi = apply_gate(psi, haar_unitary(k, 4), b)
    return psi


def single_qubit_cliffords():
    """The 24 single-qubit Clifford gates (modulo global phase), generated by closing {H, S} under
    multiplication (breadth-first search; a canonical phase makes matrices comparable)."""
    found = [np.eye(2, dtype=complex)]
    gens = [np.asarray(H), np.asarray(S)]

    def canon(U):
        i = np.flatnonzero(np.abs(U.reshape(-1)) > 1e-9)[0]
        return U * np.exp(-1j * np.angle(U.reshape(-1)[i]))

    frontier = [found[0]]
    while frontier:
        nxt = []
        for U in frontier:
            for g in gens:
                V = canon(g @ U)
                if not any(np.allclose(V, W) for W in found):
                    found.append(V)
                    nxt.append(V)
        frontier = nxt
    return jnp.asarray(np.stack(found), dtype=CDTYPE)


# ==============================================================================
# 13. Classical shadows (random Pauli measurements)
# ==============================================================================
def collect_pauli_shadows(key, psi, n_shadows):
    """Randomised-measurement data set (Huang, Kueng, Preskill 2020).
    Each snapshot: draw a basis b_q in {X,Y,Z} for every qubit, rotate, measure all qubits once.
    Returns (bases, bits), int arrays of shape (n_shadows, N); basis code 0=X, 1=Y, 2=Z.
    JAX: one snapshot is a pure function of its key  ->  `vmap` over keys gives all snapshots in parallel;
         `rots[bases[q]]` selects the rotation with a traced index (gather), so no Python branching."""
    N = psi.ndim
    rots = jnp.stack([_BASIS_ROT["X"], _BASIS_ROT["Y"], _BASIS_ROT["Z"]])

    def one(k):
        kb, ks = jax.random.split(k)
        bases = jax.random.randint(kb, (N,), 0, 3)
        phi = psi
        for q in range(N):
            phi = apply_gate(phi, rots[bases[q]], [q])
        logp = jnp.log(jnp.clip(jnp.abs(phi.reshape(-1)) ** 2, 1e-300, None))
        idx = jax.random.categorical(ks, logp)
        bits = (idx >> jnp.arange(N - 1, -1, -1)) & 1
        return bases, bits

    return jax.vmap(one)(jax.random.split(key, n_shadows))


def shadow_estimate_pauli(bases, bits, ops):
    """Estimate <P> of a Pauli string `ops` = {qubit: 'X'|'Y'|'Z'} from Pauli shadows.
        single-snapshot estimator   o = prod_{q in supp(P)}  3 * [b_q == P_q] * (-1)^{bit_q}
    (the factor 3 per qubit inverts the measurement channel; a snapshot contributes only if all its bases
    match on the support -- probability 3^{-k} -- hence variance ~ 3^k for weight-k strings).
    Returns (mean, standard error)."""
    code = {"X": 0, "Y": 1, "Z": 2}
    val = jnp.ones(bases.shape[0])
    for q, p in ops.items():
        val = val * 3.0 * (bases[:, q] == code[p]) * (1 - 2 * bits[:, q])
    return jnp.mean(val), jnp.std(val) / jnp.sqrt(val.shape[0])


def shadow_snapshot_dm(bases_row, bits_row):
    """Single-snapshot state estimate  rho_hat = (x)_q ( 3 U_q^dag |b_q><b_q| U_q - 1 )  (dense; small N).
    E[rho_hat] = rho: averaging snapshots reconstructs the state."""
    rots = [_BASIS_ROT["X"], _BASIS_ROT["Y"], _BASIS_ROT["Z"]]
    out = jnp.ones((1, 1), dtype=CDTYPE)
    for b, s in zip(np.asarray(bases_row), np.asarray(bits_row)):
        U = rots[int(b)]
        ket = U.conj().T[:, int(s)]
        out = jnp.kron(out, 3 * jnp.outer(ket, ket.conj()) - I2)
    return out


# ==============================================================================
# 14. Stabilizer Renyi entropy ("magic") via the fast Walsh-Hadamard transform
# ==============================================================================
def _wht_all_axes(f):
    """Unnormalised Walsh-Hadamard transform over all N binary axes:
        F[b] = sum_s (-1)^{b.s} f[s]
    In tensor language it is simply [[1,1],[1,-1]] applied to EVERY axis: N einsums, O(N 2^N)."""
    Hun = jnp.array([[1, 1], [1, -1]], dtype=f.dtype)
    for q in range(f.ndim):
        f = apply_gate(f, Hun, [q])
    return f


def stabilizer_renyi_entropy(psi, alpha=2, batch=256):
    """Stabilizer Renyi entropy  M_alpha = (1-alpha)^{-1} log2( sum_P <P>^{2 alpha} / 2^N )  (sum over 4^N Paulis).
    M = 0 iff stabilizer state; M > 0 quantifies non-Clifford resources ("magic").

    ALGORITHM   write P = i^{a.b} X^a Z^b (bit strings a,b).  Then
                    <P_{a,b}> = phase * sum_s (-1)^{b.s} psi^*[s XOR a] psi[s]  =  phase * WHT_b[ f_a ],
                f_a[s] = psi^*[s XOR a] psi[s].   One WHT yields all 2^N Z-patterns for a given X-pattern a
                => O(N 4^N) instead of O(8^N); |phase| = 1 drops out of <P>^{2 alpha}.
    JAX         vmap over a batch of X-patterns a (index XOR is a gather), python loop over batches to cap memory."""
    N = psi.ndim
    flat = psi.reshape(-1)
    idx = jnp.arange(2 ** N)

    def for_a(a):
        f = (jnp.conj(flat[idx ^ a]) * flat).reshape((2,) * N)
        return jnp.sum(jnp.abs(_wht_all_axes(f)) ** (2 * alpha))

    total = 0.0
    fa = jax.jit(jax.vmap(for_a))
    for start in range(0, 2 ** N, batch):
        total = total + jnp.sum(fa(idx[start:start + batch]))
    return jnp.log2(total / 2 ** N) / (1 - alpha)


def stabilizer_renyi_entropy_bruteforce(psi, alpha=2):
    """The definition, literally: loop over all 4^N Pauli strings (tiny N only).  Used to validate the WHT version."""
    import itertools
    N = psi.ndim
    tot = 0.0
    for s in itertools.product("IXYZ", repeat=N):
        tot += abs(float(expect_pauli_string(psi, "".join(s)))) ** (2 * alpha)   # (<P>^2)^alpha = |<P>|^(2 alpha): abs matters for non-integer alpha
    return float(np.log2(tot / 2 ** N) / (1 - alpha))


# ==============================================================================
# 15. Variational circuits: ansatz, gradients, optimizers
# ==============================================================================
def hea_num_params(N, layers):
    """Number of angles of `hardware_efficient_ansatz`: (layers+1) rotation blocks x N qubits x 2 angles."""
    return 2 * N * (layers + 1)


def hardware_efficient_ansatz(theta, N, layers, entangler=CZ, periodic=False):
    """Hardware-efficient ansatz
        |psi(theta)> = R(theta_L) * prod_{l<L} [ ENT * R(theta_l) ] |0...0>,   R = prod_q Rz(.) Ry(.),
    ENT = a chain of CZ (or any 2-qubit gate).  `theta` is a flat vector of length 2N(L+1).
    The function is a pure map theta -> state, hence jit-able and differentiable end to end."""
    theta = theta.reshape(layers + 1, N, 2)
    psi = zero_state(N)
    for l in range(layers + 1):
        for q in range(N):
            psi = apply_gate(psi, ry(theta[l, q, 0]), [q])
            psi = apply_gate(psi, rz(theta[l, q, 1]), [q])
        if l < layers:
            for q in range(N - 1):
                psi = apply_gate(psi, entangler, [q, q + 1])
            if periodic and N > 2:
                psi = apply_gate(psi, entangler, [N - 1, 0])
    return psi


def parameter_shift_grad(f, theta, shift=jnp.pi / 2):
    """Parameter-shift rule: for gates exp(-i theta_k P/2) with P^2 = 1 the cost is a sinusoid in theta_k,
        f = a + b cos(theta_k) + c sin(theta_k),   so for any shift s with sin(s) != 0
        d f / d theta_k = [ f(theta + s e_k) - f(theta - s e_k) ] / (2 sin s)          (EXACT, not finite differences)
    and the default s = pi/2 gives the familiar [ f(+pi/2) - f(-pi/2) ] / 2.
    -- the gradient a real quantum computer can measure: 2 circuit evaluations per parameter.
    JAX: vmap over the unit vectors e_k evaluates all shifted circuits in one batched call."""
    eye = jnp.eye(theta.size).reshape((theta.size,) + theta.shape)
    denom = 2 * jnp.sin(shift)
    return jax.vmap(lambda e: (f(theta + shift * e) - f(theta - shift * e)) / denom)(eye).reshape(theta.shape)


def spsa_grad(key, f, theta, c=0.1):
    """SPSA gradient estimate (Spall 1992): perturb ALL parameters at once along a random +-1 direction Delta,
        g = [f(theta + c Delta) - f(theta - c Delta)] / (2c) * Delta        (Delta_k^{-1} = Delta_k)
    Two evaluations regardless of the number of parameters; unbiased up to O(c^2); noisy but cheap --
    the method of choice for shot-noise-limited / stochastic cost functions."""
    delta = jax.random.rademacher(key, theta.shape).astype(theta.dtype)
    return (f(theta + c * delta) - f(theta - c * delta)) / (2 * c) * delta


def adam_init(theta):
    """Adam state: (first moment m, second moment v, step counter t)."""
    return (jnp.zeros_like(theta), jnp.zeros_like(theta), 0)


def adam_update(theta, g, state, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
    """One Adam step (Kingma & Ba):  m <- b1 m + (1-b1) g,  v <- b2 v + (1-b2) g^2,
        theta <- theta - lr * m_hat / (sqrt(v_hat) + eps),   hats = bias-corrected moments.
    Momentum (m) averages noisy gradients (great with SPSA); v gives a per-parameter step size."""
    m, v, t = state
    t = t + 1
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g ** 2
    theta = theta - lr * (m / (1 - b1 ** t)) / (jnp.sqrt(v / (1 - b2 ** t)) + eps)
    return theta, (m, v, t)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# MATRIX PRODUCT STATES: canonical forms, observables, DMRG and TEBD
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# Derived step by step in notebook 18 (Chapter 7) and copied here VERBATIM so that
# later notebooks can reach beyond the state vector without re-deriving the algorithm.  The storage is the padded,
# fixed-bond-dimension format: B (N, chi, 2, chi) right-canonical tensors plus lam (N+1, chi) Schmidt values,
# lam[j] belonging to the cut on the LEFT of site j.
_SPIN_MPS = {"0": [1, 0], "1": [0, 1], "+": [1, 1], "-": [1, -1]}      # same characters as product_state
LAM_CUT = 100 * float(jnp.finfo(RDTYPE).eps)                           # Schmidt values below this are padding
# LAM_CUT is 2.2e-14 in double precision but 1.2e-5 with PRECISION="single": there `two_site_update` zeroes Schmidt
# values below 1.2e-5 (weight up to 1.4e-10 each) and that weight is NOT counted in the discarded weight eps.


def state_to_mps_left(psi, cut=1e-13):
    """Exact left-canonical MPS of a state tensor psi[s_0..s_{N-1}].

    MATH
        psi = A[0]^{s_0} A[1]^{s_1} ... A[N-1]^{s_{N-1}},   A[j] has shape (chi_j, 2, chi_{j+1}), chi_0 = chi_N = 1.
        Sweep j = 0..N-2:   R (chi_j * 2, rest)  = U S V^dag ;  A[j] = U ;  R <- S V^dag.
    IMPLEMENTATION
        NumPy, because every iteration has a different shape.  Singular values below `cut` are dropped, so
        chi_{j+1} equals the Schmidt rank of the cut between sites j and j+1.
    RETURNS   (list of A tensors, list of the Schmidt-value arrays of the N-1 cuts)
    COST      dominated by the first SVDs: O(2^N * chi^2) -- this is a conversion tool for SMALL N only.
    """
    psi = np.asarray(psi)
    N = psi.ndim
    tensors, schmidt = [], []
    R, chi = psi.reshape(1, -1), 1                         # remainder, with a dummy left bond of dimension 1
    for j in range(N - 1):
        M = R.reshape(chi * 2, -1)                         # rows: (a_j, s_j)   columns: (s_{j+1}, ..., s_{N-1})
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        keep = max(1, int(np.sum(S > cut)))                # numerical Schmidt rank
        tensors.append(U[:, :keep].reshape(chi, 2, keep))  # A[j][a_j, s_j, a_{j+1}]
        schmidt.append(S[:keep])
        R, chi = S[:keep, None] * Vh[:keep], keep          # remainder  S V^dag
    tensors.append(R.reshape(chi, 2, 1))                   # the last remainder is the last tensor
    return tensors, schmidt

def mps_list_to_state(tensors):
    """Contract a list of rank-3 tensors back into the state tensor (validation only: O(2^N) memory).
    MATH   Eq. (1): the bond legs are summed.  `tensordot(.., axes=([-1],[0]))` ties the last leg of what we
    have so far to the first leg of the next tensor."""
    psi = tensors[0]
    for A in tensors[1:]:
        psi = np.tensordot(psi, A, axes=([-1], [0]))
    return psi[0, ..., 0]                                  # strip the two dummy boundary legs

def state_to_mps(psi, chi, cut=1e-13):
    """State tensor -> right-canonical MPS with bond dimension <= chi, stored in fixed-shape padded arrays.

    MATH
        psi ~ B[0]^{s_0} B[1]^{s_1} ... B[N-1]^{s_{N-1}},    sum_s B^s B^s^dag = 1  (right-canonical, Eq. 3)
        Sweep j = N-1..1:   R (rest, 2*chi_{j+1}) = U S V^dag ;  B[j] = V^dag (first chi rows) ;
                            lam[j] = S[:chi]/||S[:chi]|| ;  eps[j] = sum_{k>=chi} S_k^2 / sum_k S_k^2 ;  R <- U S.
    RETURNS
        B    complex array (N, chi, 2, chi)   B[j][a, s, b]; unused entries are exactly zero if nothing is truncated
        lam  real array    (N+1, chi)         Schmidt values of the cut left of site j; lam[0]=lam[N]=(1,0,..)
        eps  real array    (N+1,)             discarded weight at every cut
    TRUNCATION  lam[j] is recorded when cut j is truncated, but truncating cut j-1 afterwards changes the left
        block of cut j, so the recorded values are not the Schmidt values of the stored state.  Whenever anything
        was discarded (some eps > 0), the result is therefore passed through `mps_recanonicalise`, and the returned
        lam are the exact Schmidt values of the returned (truncated, renormalised) state on every cut.  eps keeps the
        weights discarded by the sweep.  Without truncation the sweep is exact and is returned as it is.
    COST   O(2^N chi^2): conversion tool for small N (validation, initial states from exact methods).
    """
    psi = np.asarray(psi)
    N = psi.ndim
    B = np.zeros((N, chi, 2, chi), dtype=psi.dtype)
    lam = np.zeros((N + 1, chi))
    eps = np.zeros(N + 1)
    lam[0, 0] = lam[N, 0] = 1.0
    R, r = psi.reshape(-1, 1), 1                           # remainder with a dummy right bond of dimension 1
    for j in range(N - 1, 0, -1):
        M = R.reshape(-1, 2 * r)                           # rows: (s_0..s_{j-1})   columns: (s_j, b_{j+1})
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        keep = max(1, min(chi, int(np.sum(S > cut))))
        eps[j] = np.sum(S[keep:] ** 2) / np.sum(S ** 2)
        S = S[:keep] / np.linalg.norm(S[:keep])            # truncate + renormalise
        B[j, :keep, :, :r] = Vh[:keep].reshape(keep, 2, r)
        lam[j, :keep] = S
        R, r = U[:, :keep] * S[None, :], keep              # remainder  U S
    B[0, 0, :, :r] = R.reshape(2, r)
    B, lam = jnp.asarray(B, dtype=CDTYPE), jnp.asarray(lam, dtype=RDTYPE)
    if np.any(eps > 0):                                    # later truncations changed the cuts stored earlier
        B, lam = mps_recanonicalise(B)
    return B, lam, eps

@jax.jit
def mps_to_state(B):
    """Padded MPS -> state tensor (validation only, O(2^N)).  The dummy boundary bonds are index 0.
    JAX: the shapes are static, so the whole chain of N contractions is compiled once per (N, chi)."""
    psi = B[0][0]                                          # (2, chi): left boundary leg fixed to 0
    for j in range(1, B.shape[0]):
        psi = jnp.tensordot(psi, B[j], axes=([-1], [0]))
    return psi[..., 0]                                     # right boundary leg fixed to 0

def product_mps(spec, chi):
    """Product state |spec[0]> |spec[1]> ... as a padded MPS  (B[N,chi,2,chi], lam[N+1,chi]).
    MATH   B[j][0, s, 0] = v_j[s], all other entries zero;  every cut has the single Schmidt value 1.
    '0' = spin up (Z=+1), '1' = spin down (Z=-1), '+'/'-' = eigenstates of X."""
    N = len(spec)
    B = np.zeros((N, chi, 2, chi), dtype=complex)
    for j, c in enumerate(spec):
        v = np.array(_SPIN_MPS[c], dtype=complex)
        B[j, 0, :, 0] = v / np.linalg.norm(v)
    lam = np.zeros((N + 1, chi))
    lam[:, 0] = 1.0
    return jnp.asarray(B, dtype=CDTYPE), jnp.asarray(lam, dtype=RDTYPE)

def mps_expect_sites(B, lam, O):
    """<O_j> for ALL sites j at once.   MATH  theta_j = lam[j] B[j];  <O_j> = sum theta*[a,t,b] O[t,s] theta[a,s,b].
    COST O(N chi^2).  The letter j in the einsum is a batch index (one independent contraction per site)."""
    theta = lam[:-1, :, None, None] * B
    return jnp.real(jnp.einsum("jatb,ts,jasb->j", jnp.conj(theta), jnp.asarray(O, dtype=B.dtype), theta))

def mps_expect_bonds(B, lam, h_bonds):
    """<h_j> for all N-1 nearest-neighbour operators h_j (array (N-1,4,4)) at once; sum = energy.
    MATH  theta[a,s,t,b] = lam[j][a] B[j][a,s,m] B[j+1][m,t,b];   <h> = sum theta*[a,s,t,b] h[s,t,u,v] theta[a,u,v,b]."""
    theta = jnp.einsum("ja,jasm,jmtb->jastb", lam[:-2].astype(B.dtype), B[:-1], B[1:])
    h = jnp.asarray(h_bonds, dtype=B.dtype).reshape(-1, 2, 2, 2, 2)
    return jnp.real(jnp.einsum("jastb,jstuv,jauvb->j", jnp.conj(theta), h, theta))

def mps_entropies(lam):
    """Entanglement entropy (bits) of every cut, directly from the stored Schmidt values:
    S_j = -sum_k lam[j,k]^2 log2 lam[j,k]^2   (0 log 0 := 0 handles the zero padding)."""
    p = lam ** 2
    return -jnp.sum(jnp.where(p > 0, p * jnp.log2(jnp.where(p > 0, p, 1.0)), 0.0), axis=1)

def mps_overlap(B_phi, B_psi):
    """<phi|psi> of two padded MPS (bond dimensions may differ) by a transfer-matrix sweep.
    MATH  E_0 = e_0 e_0^T (dummy left bonds);  E <- sum_s conj(Bphi^s)^T E Bpsi^s ;  result E[0,0].
    JAX   all site tensors have the same shape -> the sweep is a lax.scan over the site axis.  COST O(N chi^3)."""
    E0 = jnp.zeros((B_phi.shape[1], B_psi.shape[1]), dtype=B_psi.dtype).at[0, 0].set(1.0)

    def push(E, tensors):
        Bp, Bq = tensors
        return jnp.einsum("ac,asb,csd->bd", E, jnp.conj(Bp), Bq), None

    E, _ = lax.scan(push, E0, (B_phi, B_psi))
    return E[0, 0]

def mps_correlator(B, lam, O, i, P, j):
    """<O_i P_j> for i < j (static ints) with a transfer-matrix sweep between the two sites.  COST O((j-i) chi^3).
    O and P must be Hermitian: the product O_i P_j on two different sites is then Hermitian and its expectation value
    real, which is what is returned (the real part; a non-Hermitian pair such as sigma+ sigma- would lose its
    imaginary part).  i >= j raises ValueError (i == j is the single-site <O P>, use mps_expect_sites)."""
    i, j = int(i), int(j)
    if not 0 <= i < j < B.shape[0]:                                   # j <= i silently returned a wrong number
        raise ValueError(f"mps_correlator needs 0 <= i < j < {B.shape[0]}, got i={i}, j={j}")
    O, P = jnp.asarray(O, dtype=B.dtype), jnp.asarray(P, dtype=B.dtype)
    theta = lam[i][:, None, None] * B[i]
    E = jnp.einsum("atb,ts,asc->bc", jnp.conj(theta), O, theta)       # open legs: right bonds (bra b, ket c)
    for k in range(i + 1, j):
        E = jnp.einsum("bc,bsd,cse->de", E, jnp.conj(B[k]), B[k])     # push through site k
    return jnp.real(jnp.einsum("bc,btd,ts,csd->", E, jnp.conj(B[j]), P, B[j]))

def xxz_mpo(N, Jxx=1.0, Jyy=1.0, Jzz=1.0, hx=0.0, hz=0.0):
    """MPO of H = sum_i (Jxx X X + Jyy Y Y + Jzz Z Z)_{i,i+1} + sum_i (hx X_i + hz Z_i), open chain, Pauli convention.

    RETURNS  W of shape (N, D, D, 2, 2), D = 5:  W[j, v, w, s_out, s_in] = the operator at site j for the move v -> w.
             States of the finite-state machine: 4 = nothing placed, 1/2/3 = X/Y/Z placed on the previous site, 0 = complete.
             Site 0 keeps only row 4 and site N-1 only column 0 (all other entries zero), so all sites have the same shape.
    """
    D = 5
    W = jnp.zeros((D, D, 2, 2), dtype=CDTYPE)
    W = W.at[0, 0].set(I2).at[4, 4].set(I2)                                 # stay: identities
    W = W.at[1, 0].set(X).at[2, 0].set(Y).at[3, 0].set(Z)                   # second half of a bond -> complete
    W = W.at[4, 1].set(Jxx * X).at[4, 2].set(Jyy * Y).at[4, 3].set(Jzz * Z)  # first half of a bond
    W = W.at[4, 0].set(hx * X + hz * Z)                                     # a single-site term, complete at once
    Ws = jnp.stack([W] * N)
    Ws = Ws.at[0].set(jnp.zeros_like(W).at[4].set(W[4]))                   # left boundary: only row 4
    Ws = Ws.at[N - 1].set(jnp.zeros_like(W).at[:, 0].set(W[:, 0]))         # right boundary: only column 0
    return Ws

def mpo_to_dense(Ws):
    """Contract an MPO into the dense 2^N x 2^N matrix (validation only, small N).
    MATH  M_(j)[w] = sum_v M_(j-1)[v] (x) W[j][v, w]   starting from row 4 of site 0; the result is column 0 after the last site."""
    N, D = Ws.shape[0], Ws.shape[1]
    M = Ws[0][4]                                                            # (w, s_out, s_in) after site 0
    for j in range(1, N):
        M = jnp.einsum("vab,vwcd->wacbd", M, Ws[j]).reshape(D, 2 ** (j + 1), 2 ** (j + 1))
    return M[0]

def left_env_update(L, A, W):
    """L_{j+1}[b,w,b'] = sum L_j[a,v,a'] A[a,s,b] W[v,w,t,s] conj(A)[a',t,b']      Eq. (12).

    Letters  a v x = L_j (x = a', the bra bond);  a s b = A (ket);  v w t s = W (t = s_out, s = s_in);  x t c = conj(A) (bra, c = b').
             The BRA tensor carries the OUTPUT leg of W, because <psi|H|psi> = sum conj(A) (W A).
    """
    return jnp.einsum("avx,asb,vwts,xtc->bwc", L, A, W, jnp.conj(A))

def right_env_update(R, B, W):
    """R_j[a,v,a'] = sum B[a,s,b] W[v,w,t,s] conj(B)[a',t,b'] R_{j+1}[b,w,b']        Eq. (13).

    Letters  a s b = B (ket);  v w t s = W (t = s_out, s = s_in);  x t c = conj(B) (bra, x = a', c = b');  b w c = R_{j+1}.
    """
    return jnp.einsum("asb,vwts,xtc,bwc->avx", B, W, jnp.conj(B), R)

def mpo_expectation(B, W):
    """<psi|H|psi> of a normalised MPS (any gauge) for an MPO: contract all sites with Eq. (13) and read R_0[0, D-1, 0]."""
    chi, D = B.shape[1], W.shape[1]
    R = jnp.zeros((chi, D, chi), dtype=CDTYPE).at[0, 0, 0].set(1.0)
    for j in range(B.shape[0] - 1, -1, -1):
        R = right_env_update(R, B[j], W[j])
    return jnp.real(R[0, D - 1, 0])

def boundary_environments(chi, D):
    """L_0 and R_N in the padded format (chi, D, chi): L_0[0, D-1, 0] = 1 (nothing placed), R_N[0, 0, 0] = 1 (complete)."""
    L0 = jnp.zeros((chi, D, chi), dtype=CDTYPE).at[0, D - 1, 0].set(1.0)
    RN = jnp.zeros((chi, D, chi), dtype=CDTYPE).at[0, 0, 0].set(1.0)
    return L0, RN

def heff_apply(L, W1, W2, R, theta):
    """(H_eff theta)[a',s',t',b'] = sum L[a,v,a'] W1[v,w,s',s] W2[w,u,t',t] R[b,u,b'] theta[a,s,t,b]      Eq. (14).
    Letters: a v x = L (x = a'), v w y s = W1 (y = s'), w u z t = W2 (z = t'), b u q = R (q = b'), a s t b = theta.
    COST O(chi^3 D d^2) with the contraction order chosen by optimize="optimal"; the (4 chi^2)^2 matrix is never formed."""
    return jnp.einsum("avx,vwys,wuzt,buq,astb->xyzq", L, W1, W2, R, theta, optimize="optimal")

def lanczos_lowest(matvec, v0, m):
    """Lowest eigenpair (E, x) of a Hermitian operator given only matvec, from m Lanczos steps started at v0.

    MATH    beta_j v_{j+1} = H v_j - alpha_j v_j - beta_{j-1} v_{j-1}  (notebook 11), full reorthogonalisation (twice);
            Ritz vector x = sum_j s_j v_j of the lowest eigenvector s of the tridiagonal T.
    JAX     fixed m (lax.scan); after a breakdown (beta ~ 0) the unused rows of T get the diagonal 1e6, so that their
            spurious zero eigenvalues are pushed to the top of the spectrum.
    """
    shape = v0.shape
    v0 = (v0 / jnp.linalg.norm(v0)).reshape(-1)
    V0 = jnp.zeros((m, v0.size), dtype=v0.dtype).at[0].set(v0)

    def body(carry, j):
        V, v, v_prev, beta_prev, alive = carry
        w = matvec(v.reshape(shape)).reshape(-1)
        alpha = jnp.real(jnp.vdot(v, w))
        w = w - alpha * v - beta_prev * v_prev
        w = w - jnp.conj(V @ jnp.conj(w)) @ V                   # w <- w - sum_i v_i <v_i|w>; V holds the v_i as ROWS, hence the two conjugations
        w = w - jnp.conj(V @ jnp.conj(w)) @ V                   # ... once more (rounding)
        beta = jnp.linalg.norm(w)
        ok = alive & (beta > 1e-12)
        v_next = jnp.where(ok, w / jnp.where(ok, beta, 1.0), 0.0)
        V = V.at[j + 1].set(v_next, mode="drop")                 # at j = m-1 the index j+1 is out of range; "drop" discards that write
        alpha = jnp.where(alive, alpha, 1e6)                     # breakdown guard: a zero row of T would give a spurious eigenvalue 0
        return (V, v_next, v, jnp.where(ok, beta, 0.0), ok), (alpha, jnp.where(ok, beta, 0.0))

    init = (V0, v0, jnp.zeros_like(v0), jnp.zeros((), dtype=RDTYPE), jnp.array(True))
    (V, *_), (alphas, betas) = lax.scan(body, init, jnp.arange(m))
    evals_T, evecs_T = jnp.linalg.eigh(jnp.diag(alphas) + jnp.diag(betas[:-1], 1) + jnp.diag(betas[:-1], -1))   # Ritz values and vectors of T
    x = evecs_T[:, 0].astype(V.dtype) @ V                    # Ritz vector of the lowest Ritz value, in the big space
    return evals_T[0], (x / jnp.linalg.norm(x)).reshape(shape)

def split_two_site(theta, chi, move_right):
    """theta[a,s,t,b] (padded, shape (chi,2,2,chi)) -> two tensors (chi,2,chi), Schmidt values, discarded weight.

    MATH   theta_{(as),(tb)} = U S V^dag, keep chi values:  S <- S[:chi]/||S[:chi]||,  eps = sum_{k>=chi} S_k^2 / sum_k S_k^2
           move_right:  left = U (left-canonical),     right = S V^dag (carries the centre)
           move_left :  left = U S (carries centre),   right = V^dag (right-canonical)
    ARGUMENTS  chi must equal theta.shape[0] (the storage size): the routine always keeps exactly chi of the 2*chi values.
    """
    U, S, Vh = jnp.linalg.svd(theta.reshape(2 * chi, 2 * chi), full_matrices=False)
    weight = S ** 2
    eps = jnp.sum(weight[chi:]) / jnp.sum(weight)
    S = S[:chi] / jnp.sqrt(jnp.sum(weight[:chi]))
    U, Vh = U[:, :chi], Vh[:chi]
    if move_right:
        return U.reshape(chi, 2, chi), (S[:, None] * Vh).reshape(chi, 2, chi), S, eps
    return (U * S[None, :]).reshape(chi, 2, chi), Vh.reshape(chi, 2, chi), S, eps

_LOCAL_STEP_CACHE = {}                                             # one XLA compilation per (chi, m, direction)


def make_local_step(chi, m, move_right):
    """Compiled DMRG step on the bond (j, j+1): contract, solve H_eff theta = E theta (Lanczos, m steps), split.

    The compiled step depends only on (chi, m, direction) -- never on the site, the Hamiltonian or the field -- so
    it is built once and cached.  Without the cache every call of `dmrg`, hence every point of a parameter scan,
    pays a fresh XLA compilation, which dominates the run time although the arithmetic is identical."""
    key = (chi, m, move_right)
    if key in _LOCAL_STEP_CACHE:
        return _LOCAL_STEP_CACHE[key]

    @jax.jit
    def step(L, W1, W2, R, M1, M2):
        theta = jnp.einsum("asm,mtb->astb", M1, M2)                                   # 1. two-site tensor
        E, theta = lanczos_lowest(lambda t: heff_apply(L, W1, W2, R, t), theta, m)     # 2. local ground state
        left, right, S, eps = split_two_site(theta, chi, move_right)                  # 3. split, move the centre
        return left, right, S, eps, E

    _LOCAL_STEP_CACHE[key] = step
    return step

def mps_recanonicalise(B):
    """Exact right-canonical form and the TRUE Schmidt values of every cut, for a padded MPS whose site 0 carries the norm.

    MATH
        Left sweep  j = 0..N-2:   A[j]_{(a s), b} = Q R   (QR);   A[j] <- Q (left-canonical);   A[j+1] <- R A[j+1].
        After it, sites 0..N-2 are left-canonical and the whole norm sits on site N-1.
        Right sweep j = N-1..1:   A[j]_{a, (s b)} = U S V^dag   (SVD);   B[j] <- V^dag;   lam[j] <- S;   A[j-1] <- A[j-1] U S.
        At step j everything left of the cut is left-canonical and everything right of it right-canonical, so
        S are exactly the Schmidt values of the cut left of site j (no stored value from an earlier, different state).
        No truncation can occur: every matrix has rank <= chi, and all chi values are kept.  lam[0] = lam[N] = (1,0,..).
    WHY    `dmrg` records lam[j] at the split of bond j during the right-to-left half sweep; the later steps at bonds
           j-1, ..., 1 change the left block of that cut, so the recorded values describe an intermediate state.
           Likewise `state_to_mps` with truncation stores lam[j] before cut j-1 is truncated.  This pass makes
           (B, lam) consistent, so that the lam-based observables (mps_expect_sites, mps_expect_bonds, mps_entropies,
           mps_correlator) are those of the state that B encodes.  The state itself is unchanged (to rounding).
    ARGUMENTS  B (N, chi, 2, chi) any padded MPS whose left boundary leg is index 0 and right boundary leg index 0.
    RETURNS    B (N, chi, 2, chi) right-canonical on sites 1..N-1 (site 0 = centre, holds the norm), lam (N+1, chi).
    COST       N QR and N SVD of (2chi x chi) matrices: O(N chi^3).
    """
    N, chi = B.shape[0], B.shape[1]
    A = list(B)
    for j in range(N - 1):                                             # left sweep: QR, push R to the right
        Q, R = jnp.linalg.qr(A[j].reshape(chi * 2, chi))
        A[j] = Q.reshape(chi, 2, chi)
        A[j + 1] = jnp.einsum("ab,bsc->asc", R, A[j + 1])
    lam = jnp.zeros((N + 1, chi), dtype=RDTYPE).at[0, 0].set(1.0).at[N, 0].set(1.0)
    for j in range(N - 1, 0, -1):                                      # right sweep: SVD, push U S to the left
        U, S, Vh = jnp.linalg.svd(A[j].reshape(chi, 2 * chi), full_matrices=False)
        A[j] = Vh.reshape(chi, 2, chi)
        lam = lam.at[j].set(S.astype(RDTYPE))
        A[j - 1] = jnp.einsum("asb,bc->asc", A[j - 1], U * S[None, :].astype(U.dtype))
    A[0] = A[0].at[1:].set(0.0)                                        # unused rows of the dummy left bond: exactly zero
    return jnp.stack(A), lam

def dmrg(W, M, chi, n_sweeps, m=20, trace=False):
    """Two-site DMRG for the MPO W (N, D, D, 2, 2), starting from the right-canonical padded MPS M (N, chi, 2, chi).

    RETURNS  B (N, chi, 2, chi) right-canonical, lam (N+1, chi) Schmidt values of all bonds, history list of
             (sweep, direction, bond j, energy, discarded weight, number of non-zero Schmidt values) for every local step.
    FINAL PASS  The S of the split at bond j+1 is recorded during the right-to-left half sweep, but the later steps
             at bonds j, ..., 0 change the left block of that cut, so the recorded values belong to an intermediate
             state; they are exact only after convergence at a chi where nothing is truncated.  `dmrg` therefore ends
             with `mps_recanonicalise` (exact, O(N chi^3), state unchanged), which recomputes lam from the returned B.
    """
    N, D = W.shape[0], W.shape[1]
    step_right, step_left = make_local_step(chi, m, True), make_local_step(chi, m, False)
    L, R = [None] * (N + 1), [None] * (N + 1)
    L[0], R[N] = boundary_environments(chi, D)
    for j in range(N - 1, 1, -1):                                  # right environments of the initial state, Eq. (13)
        R[j] = right_env_update(R[j + 1], M[j], W[j])
    M = list(M)
    lam = jnp.zeros((N + 1, chi), dtype=RDTYPE).at[0, 0].set(1.0).at[N, 0].set(1.0)
    history = []
    for sweep in range(n_sweeps):
        for j in range(0, N - 1):                                  # ---- left -> right
            M[j], M[j + 1], S, eps, E = step_right(L[j], W[j], W[j + 1], R[j + 2], M[j], M[j + 1])
            L[j + 1] = left_env_update(L[j], M[j], W[j])           # the only environment that changes
            history.append((sweep, "->", j, float(E), float(eps), int(jnp.sum(S > 1e-12))))
        for j in range(N - 2, -1, -1):                             # ---- right -> left
            M[j], M[j + 1], S, eps, E = step_left(L[j], W[j], W[j + 1], R[j + 2], M[j], M[j + 1])
            R[j + 1] = right_env_update(R[j + 2], M[j + 1], W[j + 1])
            lam = lam.at[j + 1].set(S)                             # Schmidt values of the bond left of site j+1
            history.append((sweep, "<-", j, float(E), float(eps), int(jnp.sum(S > 1e-12))))
        if trace:
            last = [h for h in history if h[0] == sweep]
            print(f"   sweep {sweep}:  E = {last[-1][3]:.12f}   largest discarded weight {max(h[4] for h in last):.1e}   "
                  f"largest bond dimension {max(h[5] for h in last)}")
    B, lam = mps_recanonicalise(jnp.stack(M))                     # the lam recorded above is stale: recompute it from B
    return B, lam, history

def expm_herm(h, tau):
    """exp(-i tau h) for a small Hermitian matrix via its eigendecomposition h = V w V^dag."""
    w, v = jnp.linalg.eigh(jnp.asarray(h, dtype=CDTYPE))
    return (v * jnp.exp(-1j * tau * w)) @ v.conj().T

def two_site_update(lamL, B1, B2, G):
    """Apply a two-site gate to neighbouring MPS tensors and truncate back to bond dimension chi.

    MATH
        C[a,s,t,b]      = sum_{m,s',t'} G[s,t,s',t'] B1[a,s',m] B2[m,t',b]        (merge + gate)
        theta[a,s,t,b]  = lamL[a] C[a,s,t,b]                                       (mixed-canonical wave function)
        theta_{(as),(tb)} = sum_k U_{(as),k} S_k Vh_{k,(tb)}                       (SVD; S = new Schmidt values)
        keep k < chi:   lam' = S/||S_kept||,  B2' = Vh,  B1' = C . conj(B2') / ||S_kept||   (Hastings: no 1/lam)
        eps = sum_{k>=chi} S_k^2 / sum_k S_k^2                                     (discarded weight)
    ARGUMENTS   lamL (chi,) Schmidt values LEFT of the first site; B1, B2 (chi,2,chi); G (2,2,2,2) = U4.reshape.
    RETURNS     B1', B2', lam' (Schmidt values of the bond between the two sites), eps.
    COST        SVD of a (2chi x 2chi) matrix: O(chi^3); contractions O(chi^3).
    JAX         static shapes (chi is the array size, never data-dependent) -> jit / vmap / scan friendly.
    """
    chi = B1.shape[0]
    C = jnp.einsum("asm,mtb->astb", B1, B2)                           # (a) merge the two sites
    C = jnp.einsum("stuv,auvb->astb", G, C)                           # (b) apply the gate
    theta = lamL[:, None, None, None] * C                             #     weight with the left Schmidt values
    U, S, Vh = jnp.linalg.svd(theta.reshape(2 * chi, 2 * chi), full_matrices=False)   # (c) split
    weight = S ** 2
    kept = jnp.sum(weight[:chi])
    eps = jnp.sum(weight[chi:]) / jnp.sum(weight)                     # discarded Schmidt weight (never negative)
    nrm = jnp.sqrt(kept)
    lam_new = S[:chi] / nrm                                           # truncate + renormalise to norm 1
    mask = lam_new > LAM_CUT                                          # True on used bond indices, False on padding
    lam_new = jnp.where(mask, lam_new, 0.0)
    B2_new = Vh[:chi].reshape(chi, 2, chi) * mask[:, None, None]      # right-canonical by construction
    B1_new = jnp.einsum("astb,ktb->ask", C, jnp.conj(B2_new)) / nrm   # Hastings: C.V instead of (1/lam) U S
    return B1_new, B2_new, lam_new.astype(lamL.dtype), eps

def bond_hamiltonians(N, Jxx=0.0, Jyy=0.0, Jzz=0.0, hx=0.0, hz=0.0):
    """Two-site terms h_j (array (N-1,4,4)) with  sum_j h_j = H,
        H = sum_j (Jxx XX + Jyy YY + Jzz ZZ)_{j,j+1} + sum_j (hx X_j + hz Z_j)      (Pauli convention, open chain).
    The single-site fields are split equally between the two bonds that contain the site."""
    one = hx * X + hz * Z
    hs = []
    for j in range(N - 1):
        cL = 1.0 if j == 0 else 0.5
        cR = 1.0 if j == N - 2 else 0.5
        hs.append(Jxx * XX + Jyy * YY + Jzz * ZZ + cL * jnp.kron(one, I2) + cR * jnp.kron(I2, one))
    return jnp.stack(hs)

def bond_gates(h_bonds, tau):
    """exp(-i tau h_j) for every bond, as rank-4 gate tensors G[j][s,t,s',t'] (shape (N-1,2,2,2,2)).
    JAX: vmap turns the single-matrix function expm_herm into a batched one."""
    return jax.vmap(lambda h: expm_herm(h, tau))(h_bonds).reshape(-1, 2, 2, 2, 2)

def tebd_layer(B, lam, G, idx):
    """Apply the gates G[k] to the bonds (idx[k], idx[k]+1) -- all bonds of one parity, in parallel.
    JAX   gather with integer-array indexing, ONE vmapped two_site_update (batched einsums + batched SVD),
          scatter with .at[].set().
    RETURNS  the new (B, lam), the summed discarded weight sum_b eps_b of the layer, and sum_b sqrt(eps_b),
             the quantity that enters the triangle-inequality error estimate of Sec. 9."""
    B1, B2, lam_new, eps = jax.vmap(two_site_update)(lam[idx], B[idx], B[idx + 1], G)
    B = B.at[idx].set(B1).at[idx + 1].set(B2)
    lam = lam.at[idx + 1].set(lam_new)
    return B, lam, jnp.sum(eps), jnp.sum(jnp.sqrt(eps))

def mps_rdm2(B, lam, i, j):
    """Two-site reduced density matrix of sites i < j (static ints) from a right-canonical MPS.

    MATH   rho[(s_i s_j), (s_i' s_j')] = sum over everything else of psi psi^*.  The right part of the
           chain traces to the identity by right-canonicity.  The left part is the density matrix of the
           bond to the left of site i, built by a transfer-matrix sweep from the left boundary:
               L_0 = e_0 e_0^T ,   L_{k+1} = sum_s B[k]^s^dag L_k B[k]^s .
    WHY NOT lam  In the canonical gauge that left environment IS diag(lam[i]^2), and using `lam` directly
           is cheaper -- but only exact if lam belongs to B.  The values `dmrg` records during its sweeps
           do not (later local steps change the left block of a recorded cut); `dmrg` now ends with
           `mps_recanonicalise`, which fixes its output, but without that pass, at N=12, chi=16, XXZ
           Delta=1.5, hx=0.5 after ONE sweep, the recorded lam is off by 2.2e-1 and the lam^2-based rho by
           2.2e-1 -- while still being Hermitian, unit-trace and positive, so no physicality check catches
           it.  The sweep below removes the dependence on lam: the result is exact for ANY right-canonical
           B, however its lam was obtained.  `lam` is kept in the signature for call compatibility and is
           not used.
    CAVEAT After TRUNCATING TEBD steps, B is right-canonical only to O(eps / lam_min^2), so the right part of the
           chain traces to the identity only approximately and rho is approximate to the same order.  Measured
           against the partial trace of the state B encodes (critical TFIM, N = 12, 80 steps of dt = 0.05):
           1.2e-5 to 8.1e-5 for chi = 4..16, 3.8e-7 for chi = 32 -- below the truncation error of the state itself
           (1.7e-1 to 5.3e-4 against the exact evolution).  dmrg and state_to_mps return an exactly canonical B.
    EINSUM open the physical legs at site i against the left environment, walk the transfer matrix through
           the sites between (their physical legs traced), then close at site j with the right bond traced.
    COST   O(j chi^3).  Returns a (4, 4) matrix, row/column order (s_i s_j).
    """
    n = B.shape[0]
    i, j = int(i), int(j)
    if not 0 <= i < j < n:                       # j <= i silently returned a plausible, wrong matrix
        raise ValueError(f"mps_rdm2 needs 0 <= i < j < {n}, got i={i}, j={j}")
    L = jnp.zeros((B.shape[1], B.shape[1]), dtype=B.dtype).at[0, 0].set(1.0)
    for k in range(i):                                                    # left environment of the bond
        L = jnp.einsum("ac,asb,csd->bd", L, B[k], jnp.conj(B[k]))
    T = jnp.einsum("ax,asb,xtc->stbc", L, B[i], jnp.conj(B[i]))           # site i: legs s (ket), t (bra)
    for k in range(i + 1, j):
        T = jnp.einsum("stbc,bud,cue->stde", T, B[k], jnp.conj(B[k]))     # push through site k
    R = jnp.einsum("stbc,bud,cvd->sutv", T, B[j], jnp.conj(B[j]))         # site j open, right bond traced
    return R.reshape(4, 4)


@partial(jax.jit, static_argnames=("n_steps",))
def mps_tebd_evolve(B, lam, h_bonds, dt, n_steps):
    """Evolve a padded MPS for n_steps Strang steps  e^{-i H_even dt/2} e^{-i H_odd dt} e^{-i H_even dt/2}.

    RETURNS   (B, lam) at the final time and a dict of arrays with leading axis = time step:
        "z"    (n_steps, N)     <Z_j>                      "S"   (n_steps, N+1) entanglement entropy of every cut
        "E"    (n_steps,)       energy <H>                 "eps" (n_steps,)     discarded weight summed over the step
        "sqeps" (n_steps,)      sum of sqrt(eps) over the ~1.5 N truncations of the step  (Sec. 9)
    COST      per step ~ 1.5 N SVDs of (2chi x 2chi) matrices: O(N chi^3) time, O(N chi^2) memory.
    JAX       lax.scan over time (carry = MPS of fixed shape), vmap over the bonds of a layer, jit around everything.
    """
    N = B.shape[0]
    even, odd = jnp.arange(0, N - 1, 2), jnp.arange(1, N - 1, 2)
    G_half, G_full = bond_gates(h_bonds, dt / 2), bond_gates(h_bonds, dt)

    def step(carry, _):
        B, lam = carry
        B, lam, e1, q1 = tebd_layer(B, lam, G_half[even], even)
        B, lam, e2, q2 = tebd_layer(B, lam, G_full[odd], odd)
        B, lam, e3, q3 = tebd_layer(B, lam, G_half[even], even)
        out = {"z": mps_expect_sites(B, lam, Z), "S": mps_entropies(lam),
               "E": jnp.sum(mps_expect_bonds(B, lam, h_bonds)), "eps": e1 + e2 + e3, "sqeps": q1 + q2 + q3}
        return (B, lam), out

    return lax.scan(step, (B, lam), None, length=n_steps)
