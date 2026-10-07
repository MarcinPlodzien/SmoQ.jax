"""SmoQ.jax: a matrix-free JAX engine for quantum many-body systems.

A state of N spins-1/2 (qubits) is a rank-N tensor of shape (2,)*N, and every operator -- a gate, a Hamiltonian
term, a Kraus operator, a projector -- acts on it through a single einsum contraction, so no 2^N x 2^N matrix is
ever built.  All functions are jit-, vmap- and grad-compatible.

    import smoqjax as sq
    psi = sq.apply_gate(sq.zero_state(3), sq.H, (0,))

The whole engine lives in ``smoqjax.engine`` (one documented file); this package re-exports its public names.
Precision: double (float64/complex128) by default, which switches on ``jax_enable_x64`` at import; set the
environment variable ``QE_PRECISION=single`` before importing for float32/complex64.
"""
from . import engine as _engine

__all__ = sorted(
    name for name, obj in vars(_engine).items()
    if not name.startswith("_")
    and (getattr(obj, "__module__", None) == _engine.__name__ or (name.isupper() and not callable(obj)))
)
globals().update({name: getattr(_engine, name) for name in __all__})
engine = _engine
