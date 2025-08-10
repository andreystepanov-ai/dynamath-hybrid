# core_py/tensors.py
"""
Basic tensor operations for Dynamath Hybrid (NumPy baseline).

This module is intentionally framework-agnostic and relies only on NumPy,
so it runs everywhere. You can later swap the internals to JAX/PyTorch
with the same API surface.

Covered:
- Creation: zeros, ones, full, arange, randn, seed
- Casting & shape: as_array, to_list, reshape, flatten, expand_dims, squeeze, tile
- Math: l2, norm, normalize, dot, matmul, outer, hadamard, proj, cosine_similarity
- Reductions: sum_, mean_, var_, std_, logsumexp
- Nonlinearities: relu, sigmoid, tanh, softmax
- Utils: clamp, clip_norm, safe_div, concat, stack, pad
- Distances: euclidean, cosine_distance
"""

from __future__ import annotations
from typing import Iterable, Sequence, Tuple, Optional
import numpy as np


# ---------- Creation ----------

def seed(value: int) -> None:
    """Set NumPy RNG seed."""
    np.random.seed(value)

def as_array(x, dtype=np.float64) -> np.ndarray:
    """Convert to np.ndarray with given dtype (no copy if possible)."""
    return np.asarray(x, dtype=dtype)

def zeros(shape: Sequence[int], dtype=np.float64) -> np.ndarray:
    return np.zeros(tuple(shape), dtype=dtype)

def ones(shape: Sequence[int], dtype=np.float64) -> np.ndarray:
    return np.ones(tuple(shape), dtype=dtype)

def full(shape: Sequence[int], fill_value: float, dtype=np.float64) -> np.ndarray:
    return np.full(tuple(shape), fill_value, dtype=dtype)

def arange(start: float, stop: Optional[float] = None, step: float = 1.0, dtype=np.float64) -> np.ndarray:
    if stop is None:
        start, stop = 0, start
    return np.arange(start, stop, step, dtype=dtype)

def randn(*shape: int, dtype=np.float64) -> np.ndarray:
    return np.random.randn(*shape).astype(dtype)


# ---------- Casting & shape ----------

def to_list(x: np.ndarray) -> list:
    return x.tolist()

def reshape(x: np.ndarray, new_shape: Sequence[int]) -> np.ndarray:
    return np.reshape(x, new_shape)

def flatten(x: np.ndarray) -> np.ndarray:
    return x.reshape(-1)

def expand_dims(x: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.expand_dims(x, axis)

def squeeze(x: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    return np.squeeze(x, axis=axis)

def tile(x: np.ndarray, reps: Sequence[int]) -> np.ndarray:
    return np.tile(x, reps)


# ---------- Linear algebra & basic math ----------

def l2(x: Iterable[float]) -> float:
    x = as_array(x)
    return float(np.sqrt(np.sum(x * x)))

def norm(x: np.ndarray, ord: Optional[int | float] = 2, axis: Optional[int | Tuple[int, ...]] = None, keepdims=False) -> np.ndarray | float:
    return np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)

def normalize(x: Iterable[float] | np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize vector or last-dim if x is 2D+."""
    a = as_array(x)
    if a.ndim == 1:
        n = max(l2(a), eps)
        return a / n
    # normalize along last axis
    n = np.sqrt(np.sum(a * a, axis=-1, keepdims=True))
    n = np.maximum(n, eps)
    return a / n

def dot(a: Iterable[float] | np.ndarray, b: Iterable[float] | np.ndarray) -> float:
    a = as_array(a); b = as_array(b)
    return float(np.dot(a, b))

def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a @ b

def outer(a: Iterable[float], b: Iterable[float]) -> np.ndarray:
    a = as_array(a); b = as_array(b)
    return np.outer(a, b)

def hadamard(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise product."""
    return np.multiply(a, b)

def proj(u: Iterable[float], v: Iterable[float], eps: float = 1e-12) -> np.ndarray:
    """
    Projection of u onto v: proj_v(u) = (⟨u,v⟩ / (⟨v,v⟩+eps)) * v
    """
    u = as_array(u); v = as_array(v)
    denom = float(np.dot(v, v)) + eps
    return (float(np.dot(u, v)) / denom) * v

def cosine_similarity(a: Iterable[float] | np.ndarray, b: Iterable[float] | np.ndarray, eps: float = 1e-12) -> float:
    a = as_array(a); b = as_array(b)
    na = max(l2(a), eps)
    nb = max(l2(b), eps)
    return float(np.dot(a, b) / (na * nb))


# ---------- Reductions ----------

def sum_(x: np.ndarray, axis: Optional[int | Tuple[int, ...]] = None, keepdims: bool = False) -> np.ndarray | float:
    return np.sum(x, axis=axis, keepdims=keepdims)

def mean_(x: np.ndarray, axis: Optional[int | Tuple[int, ...]] = None, keepdims: bool = False) -> np.ndarray | float:
    return np.mean(x, axis=axis, keepdims=keepdims)

def var_(x: np.ndarray, axis: Optional[int | Tuple[int, ...]] = None, keepdims: bool = False) -> np.ndarray | float:
    return np.var(x, axis=axis, keepdims=keepdims)

def std_(x: np.ndarray, axis: Optional[int | Tuple[int, ...]] = None, keepdims: bool = False) -> np.ndarray | float:
    return np.std(x, axis=axis, keepdims=keepdims)

def logsumexp(x: np.ndarray, axis: Optional[int | Tuple[int, ...]] = None, keepdims: bool = False) -> np.ndarray | float:
    """Stable log-sum-exp."""
    m = np.max(x, axis=axis, keepdims=True)
    y = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m
    return y if keepdims else np.squeeze(y, axis=axis)


# ---------- Nonlinearities ----------

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)

def sigmoid(x: np.ndarray) -> np.ndarray:
    # Stable sigmoid
    x = as_array(x)
    pos = x >= 0
    neg = ~pos
    z = np.empty_like(x)
    z[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    z[neg] = ex / (1.0 + ex)
    return z

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    s = np.sum(e, axis=axis, keepdims=True)
    return e / s


# ---------- Utils ----------

def clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(x, lo, hi)

def clip_norm(x: np.ndarray, max_norm: float, eps: float = 1e-12) -> np.ndarray:
    n = norm(x)
    if n <= max_norm + eps:
        return x
    return x * (max_norm / (n + eps))

def safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return a / (b + eps)

def concat(arrays: Sequence[np.ndarray], axis: int = 0) -> np.ndarray:
    return np.concatenate(arrays, axis=axis)

def stack(arrays: Sequence[np.ndarray], axis: int = 0) -> np.ndarray:
    return np.stack(arrays, axis=axis)

def pad(x: np.ndarray, pad_width: Sequence[Tuple[int, int]], mode: str = "constant", constant_values: float = 0.0) -> np.ndarray:
    return np.pad(x, pad_width, mode=mode, constant_values=constant_values)


# ---------- Distances ----------

def euclidean(a: Iterable[float] | np.ndarray, b: Iterable[float] | np.ndarray) -> float:
    a = as_array(a); b = as_array(b)
    return float(np.linalg.norm(a - b))

def cosine_distance(a: Iterable[float] | np.ndarray, b: Iterable[float] | np.ndarray, eps: float = 1e-12) -> float:
    return 1.0 - cosine_similarity(a, b, eps=eps)


# ---------- Self-test (optional) ----------

if __name__ == "__main__":
    seed(42)
    v = randn(3)
    v_hat = normalize(v)
    assert abs(l2(v_hat) - 1.0) < 1e-9
    M = matmul(outer(v_hat, v_hat), np.eye(3))
    assert M.shape == (3, 3)
    cs = cosine_similarity([1, 0, 0], [0.5, 0, 0])
    assert abs(cs - 1.0) < 1e-12
    print("tensors.py self-check passed.")
