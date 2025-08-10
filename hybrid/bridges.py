# hybrid/bridges.py
# -*- coding: utf-8 -*-
"""
Bridges between the classical (tensor) domain and a quantum-like statevector domain.

This module provides:
  - Robust, configurable mappings from classical R^n vectors to quantum statevectors C^{2^k}
    (amplitude encoding and angle/product-state encoding).
  - Inverse mappings that recover classical features from a statevector (approximate for angle encoding).
  - Power-of-two length management (pad/trim/error).
  - Normalization policies, dtype control, batch APIs.
  - Statevector validation and measurement sampling (for tests/simulations).
  - Fusion utilities to combine classical and quantum-derived features.

Design principles:
  - Pure NumPy implementation (no hard dependency on a quantum SDK).
  - Numerically stable, explicit eps thresholds.
  - Clear, dataclass-based configuration with safe defaults.
"""

from __future__ import annotations

from dataclasses import dataclass
from spec.spec_loader import get_project_config
from typing import Iterable, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, Sequence[float], List[float], Tuple[float, ...]]

EncodeMode = Literal["amplitude", "angle"]
Pow2Policy = Literal["pad", "trim", "error"]
NormMode = Literal["l2", "max", "none"]


# ================================
# Configuration
# ================================

@dataclass(frozen=True)
class BridgeConfig:
    """
    Configuration for classical↔quantum bridging.

    Parameters
    ----------
    encode_mode : {"amplitude","angle"}
        - "amplitude": map a real vector to a complex statevector by normalized amplitudes.
        - "angle":     build a product state ⨂_i [cos(θ_i/2), sin(θ_i/2)] from features θ_i.
    pow2_policy : {"pad","trim","error"}
        How to handle non-power-of-two feature lengths for "amplitude" mode.
        - "pad":  right-pad with zeros to next power of two.
        - "trim": truncate to previous power of two (if len is already pow2, no change).
        - "error": raise ValueError if not a power of two.
    normalize : {"l2","max","none"}
        Classical-vector normalization before encoding.
    dtype_real : np.dtype
        Floating dtype for reals.
    dtype_complex : np.dtype
        Complex dtype for statevectors.
    angle_clip : float
        Absolute upper bound for feature angles |θ| ≤ angle_clip in "angle" mode (radians).
    eps : float
        Small epsilon for numerical stability.
    max_qubits : Optional[int]
        If set, limit the number of qubits used (truncate extra features in "angle" mode,
        or limit padded/trimmed length in "amplitude" mode to 2**max_qubits).
    """

    encode_mode: EncodeMode = "amplitude"
    pow2_policy: Pow2Policy = "pad"
    normalize: NormMode = "l2"
    dtype_real: np.dtype = np.float64
    dtype_complex: np.dtype = np.complex128
    angle_clip: float = np.pi  # allow full [-π, π] by default
    eps: float = 1e-12
    max_qubits: Optional[int] = None


# ================================
# Utilities
# ================================
# --- add below existing imports ---
from spec.spec_loader import get_project_config

# --- add below BridgeConfig class ---
def bridge_config_from_spec(fallback: BridgeConfig = BridgeConfig()) -> BridgeConfig:
    cfg = get_project_config()  # spec/hybrid.toml
    q = (cfg.get("quantum") or {})
    t = (cfg.get("tensor") or {})

    max_q = q.get("max_qubits", fallback.max_qubits)
    norm  = (t.get("normalize_inputs", True))
    normalize_mode = "l2" if norm else "none"

    return BridgeConfig(
        encode_mode=fallback.encode_mode,
        pow2_policy=fallback.pow2_policy,
        normalize=normalize_mode,
        dtype_real=fallback.dtype_real,
        dtype_complex=fallback.dtype_complex,
        angle_clip=fallback.angle_clip,
        eps=fallback.eps,
        max_qubits=max_q
    )
def _to_1d_array(x: ArrayLike, dtype: np.dtype) -> np.ndarray:
    a = np.asarray(x, dtype=dtype)
    if a.ndim == 0:
        a = a.reshape(1)
    if a.ndim > 1:
        a = a.reshape(-1)
    return a


def _l2(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def _safe_normalize(x: np.ndarray, mode: NormMode, eps: float) -> np.ndarray:
    if mode == "none":
        return x
    if mode == "l2":
        n = max(_l2(x), eps)
        return x / n
    if mode == "max":
        m = float(np.max(np.abs(x))) if x.size else 1.0
        m = max(m, eps)
        return x / m
    raise ValueError(f"Unknown normalize mode: {mode}")


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1) == 0)


def _next_pow2(n: int) -> int:
    # next power of two ≥ n
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def _prev_pow2(n: int) -> int:
    # previous power of two ≤ n (for n>=1)
    return 1 if n <= 1 else 1 << (n.bit_length() - 1)


def _maybe_cap_qubits(length_pow2: int, cfg: BridgeConfig) -> int:
    if cfg.max_qubits is None:
        return length_pow2
    cap = 1 << cfg.max_qubits
    return min(length_pow2, cap)


def _ensure_pow2_length(vec: np.ndarray, cfg: BridgeConfig) -> np.ndarray:
    n = vec.size
    if n == 0:
        # Degenerate state |0>
        out = np.zeros(1, dtype=vec.dtype)
        out[0] = 1.0
        return out

    if _is_power_of_two(n):
        length = n
    else:
        if cfg.pow2_policy == "error":
            raise ValueError(f"Length {n} is not a power of two (policy=error)")
        if cfg.pow2_policy == "pad":
            length = _next_pow2(n)
        elif cfg.pow2_policy == "trim":
            length = _prev_pow2(n)
        else:
            raise ValueError(f"Unknown pow2_policy={cfg.pow2_policy}")

    length = _maybe_cap_qubits(length, cfg)

    if length == n:
        return vec

    if length > n:
        out = np.zeros(length, dtype=vec.dtype)
        out[:n] = vec
        return out
    else:
        # trim
        return vec[:length]


def _validate_statevector(psi: np.ndarray, eps: float) -> None:
    if psi.ndim != 1:
        raise ValueError("Statevector must be 1-D.")
    norm = float(np.vdot(psi, psi).real)
    if not np.isfinite(norm):
        raise ValueError("Statevector norm is not finite.")
    if abs(norm - 1.0) > 1e-6 and norm > eps:
        # Normalize in place if "almost normalized"; else raise.
        psi /= np.sqrt(norm)
        norm2 = float(np.vdot(psi, psi).real)
        if abs(norm2 - 1.0) > 1e-6:
            raise ValueError(f"Statevector not normalized (norm={norm}) and auto-fix failed.")


def _kronecker_all(vectors: List[np.ndarray]) -> np.ndarray:
    """Compute tensor product ⊗ vectors[i]."""
    out = np.array([1.0], dtype=np.complex128)
    for v in vectors:
        out = np.kron(out, v.astype(np.complex128, copy=False))
    return out


# ================================
# Encoders
# ================================

def _amplitude_encode(vec: np.ndarray, cfg: BridgeConfig) -> np.ndarray:
    """Map real vector -> complex statevector by normalized amplitudes."""
    vec = _safe_normalize(vec, cfg.normalize, cfg.eps).astype(cfg.dtype_real, copy=False)
    vec = _ensure_pow2_length(vec, cfg)
    # Normalize to unit length (amplitudes)
    nrm = max(_l2(vec), cfg.eps)
    amp = (vec / nrm).astype(cfg.dtype_real, copy=False)
    state = amp.astype(cfg.dtype_complex)  # purely real amplitudes → complex
    # ensure normalization (minor correction for fp error)
    _validate_statevector(state, cfg.eps)
    return state


def _angle_encode(angles: np.ndarray, cfg: BridgeConfig) -> np.ndarray:
    """
    Map angles θ_i to a product state: |ψ⟩ = ⊗_i [cos(θ_i/2), sin(θ_i/2)] (real amplitudes in C^2).
    If max_qubits is set, truncate the feature list to that many qubits.
    """
    if cfg.max_qubits is not None:
        angles = angles[: cfg.max_qubits]

    # Clip angles for numerical sanity
    a = np.clip(angles.astype(cfg.dtype_real, copy=False), -cfg.angle_clip, cfg.angle_clip)
    half = 0.5 * a
    # Local qubit states
    locals_ = [np.array([np.cos(h), np.sin(h)], dtype=cfg.dtype_real) for h in half]
    if not locals_:
        # zero-qubit system: define |0> as statevector of length 1
        return np.array([1.0 + 0.0j], dtype=cfg.dtype_complex)
    # Tensor product into 2^n vector
    psi = _kronecker_all(locals_)
    psi = psi.astype(cfg.dtype_complex, copy=False)
    _validate_statevector(psi, cfg.eps)
    return psi


# ================================
# Decoders (approximate)
# ================================

def _amplitude_decode(state: np.ndarray, cfg: BridgeConfig, original_len: Optional[int]) -> np.ndarray:
    """
    Recover a real vector from an amplitude-encoded state by taking real part of amplitudes.
    If original_len is provided and smaller than len(state), trim to it.
    """
    _validate_statevector(state, cfg.eps)
    real_amp = state.real.astype(cfg.dtype_real, copy=False)
    if original_len is not None and 0 < original_len <= real_amp.size:
        real_amp = real_amp[:original_len]
    # Undo normalization if desired: here we keep it normalized to keep pipeline stable.
    return real_amp


def _angle_decode_product_marginals(state: np.ndarray, cfg: BridgeConfig, out_dim: Optional[int]) -> np.ndarray:
    """
    Approximate inverse of product-state angle encoding by computing single-qubit marginals.

    If |ψ⟩ = ⊗_i [c_i, s_i] with c_i=cos(θ_i/2), s_i=sin(θ_i/2),
    then P_i(1) = s_i^2, P_i(0) = c_i^2 and θ_i ≈ 2 * arcsin( sqrt(P_i(1)) ).

    Returns an angle vector θ (length = number of qubits), optionally trimmed to out_dim.
    """
    _validate_statevector(state, cfg.eps)
    n = state.size
    if not _is_power_of_two(n):
        raise ValueError("Angle decode expects power-of-two statevector length.")

    qubits = int(np.log2(n))
    probs = (np.abs(state) ** 2).astype(cfg.dtype_real, copy=False)

    # Compute single-qubit marginals P_i(1)
    p1 = np.empty(qubits, dtype=cfg.dtype_real)
    for q in range(qubits):
        p = 0.0
        step = 1 << (q + 1)
        block = 1 << q  # size of 1-block for bit q
        # Sum over basis states where bit q is 1
        for start in range(0, n, step):
            sl = slice(start + block, start + step)
            p += float(probs[sl].sum())
        p1[q] = p

    # Recover angles θ_i = 2 * arcsin( sqrt(p1) )
    p1 = np.clip(p1, 0.0, 1.0)
    angles = 2.0 * np.arcsin(np.sqrt(p1))

    if out_dim is not None:
        angles = angles[:out_dim]

    # Clip back into valid range
    angles = np.clip(angles, -cfg.angle_clip, cfg.angle_clip)
    return angles.astype(cfg.dtype_real, copy=False)


# ================================
# Public API: single-vector bridge
# ================================

def classical_to_quantum(
    vec: ArrayLike,
    cfg: BridgeConfig = BridgeConfig(),
    original_len_hint: Optional[int] = None,
) -> np.ndarray:
    """
    Map a classical feature vector to a complex statevector per cfg.encode_mode.

    Parameters
    ----------
    vec : ArrayLike
        Classical features (1-D).
    cfg : BridgeConfig
        Bridge configuration.
    original_len_hint : Optional[int]
        Used only for book-keeping if you plan round-trips (e.g., to restore length after amplitude decode).

    Returns
    -------
    np.ndarray (complex)
        Statevector of length 2^k.
    """
    x = _to_1d_array(vec, cfg.dtype_real)
    if cfg.encode_mode == "amplitude":
        return _amplitude_encode(x, cfg)
    elif cfg.encode_mode == "angle":
        return _angle_encode(x, cfg)
    else:
        raise ValueError(f"Unsupported encode_mode: {cfg.encode_mode}")


def quantum_to_classical(
    state: ArrayLike,
    cfg: BridgeConfig = BridgeConfig(),
    out_dim: Optional[int] = None,
) -> np.ndarray:
    """
    Map a quantum statevector to a classical feature vector.

    For "amplitude": returns the real amplitudes (trimmed to out_dim if given).
    For "angle":     recovers approximate angles via single-qubit marginals.

    Parameters
    ----------
    state : ArrayLike (complex)
        Statevector length 2^k (will be normalized if not already).
    cfg : BridgeConfig
        Bridge configuration.
    out_dim : Optional[int]
        Desired output feature length (truncate if provided).

    Returns
    -------
    np.ndarray (real)
        Classical features.
    """
    psi = np.asarray(state, dtype=cfg.dtype_complex)
    if cfg.encode_mode == "amplitude":
        return _amplitude_decode(psi, cfg, out_dim)
    elif cfg.encode_mode == "angle":
        return _angle_decode_product_marginals(psi, cfg, out_dim)
    else:
        raise ValueError(f"Unsupported encode_mode: {cfg.encode_mode}")


# ================================
# Batch APIs
# ================================

def classical_batch_to_quantum(
    mat: ArrayLike,
    cfg: BridgeConfig = BridgeConfig(),
) -> List[np.ndarray]:
    """
    Encode a batch of classical vectors to statevectors.

    Returns a Python list of complex np.ndarray (statevectors),
    because individual outputs may differ in length (due to pow2 policy).
    """
    A = np.asarray(mat, dtype=cfg.dtype_real)
    if A.ndim == 1:
        A = A.reshape(1, -1)
    out: List[np.ndarray] = []
    for i in range(A.shape[0]):
        out.append(classical_to_quantum(A[i], cfg=cfg))
    return out


def quantum_batch_to_classical(
    states: Sequence[ArrayLike],
    cfg: BridgeConfig = BridgeConfig(),
    out_dim: Optional[int] = None,
) -> np.ndarray:
    """
    Decode a batch of statevectors to a stacked 2-D array (padding/truncation rules apply).

    If out_dim is None, uses the maximum recoverable dimension across the batch.
    Shorter vectors are padded with zeros to match the longest (for amplitude mode),
    or truncated to min(qubits across batch) for angle mode.
    """
    # First pass: decode individually to figure lengths
    decoded: List[np.ndarray] = []
    max_len = 0
    for s in states:
        x = quantum_to_classical(s, cfg=cfg, out_dim=out_dim)
        decoded.append(x)
        max_len = max(max_len, x.size)

    if out_dim is None:
        target = max_len
    else:
        target = out_dim

    # Stack with right-padding zeros if needed
    R = np.zeros((len(decoded), target), dtype=cfg.dtype_real)
    for i, x in enumerate(decoded):
        L = min(target, x.size)
        R[i, :L] = x[:L]
    return R


# ================================
# Measurement & Validation
# ================================

def sample_counts(
    state: ArrayLike,
    shots: int = 1024,
    seed: Optional[int] = None,
    cfg: BridgeConfig = BridgeConfig(),
) -> dict:
    """
    Simulate projective measurement in computational basis.

    Returns
    -------
    dict: bitstring -> count
    """
    rng = np.random.default_rng(seed)
    psi = np.asarray(state, dtype=cfg.dtype_complex)
    _validate_statevector(psi, cfg.eps)

    probs = np.abs(psi) ** 2
    n = psi.size
    if not _is_power_of_two(n):
        raise ValueError("Statevector length must be a power of two for measurement.")
    qubits = int(np.log2(n))

    outcomes = rng.choice(n, size=shots, p=probs)
    counts: dict = {}
    for idx in outcomes:
        bitstr = format(idx, f"0{qubits}b")
        counts[bitstr] = counts.get(bitstr, 0) + 1
    return counts


def validate_statevector(
    state: ArrayLike,
    cfg: BridgeConfig = BridgeConfig(),
) -> Tuple[bool, float]:
    """
    Validate 1-D shape and (near-)unit norm. Returns (ok, norm).

    If the norm is off but finite, this function normalizes a copy
    to compute norm, but does not mutate the original array.
    """
    psi = np.asarray(state, dtype=cfg.dtype_complex)
    if psi.ndim != 1:
        return False, float("nan")
    norm = float(np.vdot(psi, psi).real)
    if not np.isfinite(norm):
        return False, norm
    if abs(norm - 1.0) <= 1e-6:
        return True, norm
    # Check if normalizable
    if norm <= cfg.eps:
        return False, norm
    psi2 = psi / np.sqrt(norm)
    norm2 = float(np.vdot(psi2, psi2).real)
    return (abs(norm2 - 1.0) <= 1e-6), norm


# ================================
# Fusion (classical + quantum-derived)
# ================================

def weighted_fuse(
    classical_vec: ArrayLike,
    quantum_vec: ArrayLike,
    w_classical: float = 0.5,
    w_quantum: float = 0.5,
    normalize_out: bool = True,
    cfg: BridgeConfig = BridgeConfig(),
) -> np.ndarray:
    """
    Fuse classical and quantum-derived vectors by weighted average, with optional normalization.

    If lengths differ, right-pad the shorter with zeros.
    """
    a = _to_1d_array(classical_vec, cfg.dtype_real)
    b = _to_1d_array(quantum_vec, cfg.dtype_real)
    L = max(a.size, b.size)
    A = np.zeros(L, dtype=cfg.dtype_real)
    B = np.zeros(L, dtype=cfg.dtype_real)
    A[: a.size] = a
    B[: b.size] = b
    out = w_classical * A + w_quantum * B
    return _safe_normalize(out, "l2", cfg.eps) if normalize_out else out


# ================================
# Minimal self-test
# ================================

if __name__ == "__main__":
    cfg_amp = BridgeConfig(encode_mode="amplitude", pow2_policy="pad", normalize="l2")
    cfg_ang = BridgeConfig(encode_mode="angle", max_qubits=4)

    x = np.array([0.2, -1.0, 0.5], dtype=np.float64)
    psi_amp = classical_to_quantum(x, cfg_amp)
    ok, nrm = validate_statevector(psi_amp, cfg_amp)
    assert ok, f"Amplitude state invalid, norm={nrm}"

    x_rec_amp = quantum_to_classical(psi_amp, cfg_amp, out_dim=x.size)
    assert x_rec_amp.shape == x.shape

    # Angle encoding with 3 features → 3 qubits → length 8 statevector
    angles = np.array([0.1, 1.2, -0.7], dtype=np.float64)
    psi_ang = classical_to_quantum(angles, cfg_ang)
    ok, nrm = validate_statevector(psi_ang, cfg_ang)
    assert ok, f"Angle state invalid, norm={nrm}"
    # Recover approximate angles from single-qubit marginals
    angles_rec = quantum_to_classical(psi_ang, cfg_ang, out_dim=len(angles))
    assert angles_rec.shape == angles.shape

    # Batch encode-decode
    batch = np.stack([x, x * 2.0 + 0.1])
    states = classical_batch_to_quantum(batch, cfg_amp)
    back = quantum_batch_to_classical(states, cfg_amp, out_dim=x.size)
    assert back.shape == (2, x.size)

    # Measurement
    counts = sample_counts(psi_ang, shots=256, seed=42, cfg=cfg_ang)
    assert sum(counts.values()) == 256

    # Fusion
    fused = weighted_fuse(x_rec_amp, angles_rec, 0.6, 0.4, True, cfg_amp)
    assert fused.ndim == 1 and fused.size == max(x_rec_amp.size, angles_rec.size)

    print("bridges.py self-check passed.")
