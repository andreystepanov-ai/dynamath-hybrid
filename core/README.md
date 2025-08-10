# Dynamath Hybrid – Tensor Operations Module

## Overview

**Dynamath Hybrid** is an experimental platform for **Tensor + Quantum Hybrid Computing**.  
It bridges classical tensor-based machine learning and quantum computing by providing a modular architecture:

- `core/` – tensor algebra and classical computation (CPU/GPU).
- `quantum/` – quantum circuit manipulation (Qiskit, Cirq).
- `hybrid/` – bridge layer that converts between tensor states and quantum amplitudes.
- `viz/` – visualization of hybrid computation flows.
- `spec/` – data format and API specifications.

This document describes the **Tensor Operations Module** (`tensors.py`), the foundation for all classical computation in Dynamath Hybrid.

---

## Principles

### 1. Tensor Mathematics as a Foundation
Tensors are multi-dimensional arrays that generalize scalars, vectors, and matrices.  
They are the language of modern **deep learning**, **physics simulations**, and **quantum state preparation**.

In **Dynamath Hybrid**, tensors are:
- **Normalized** for stable numerical computations.
- **Composable** to form complex structures.
- **Backend-agnostic** so they can run on NumPy, JAX, or PyTorch seamlessly.

---

### 2. Why Normalization?
In hybrid tensor-quantum workflows, vector length often represents **probability amplitude norms** in quantum mechanics.  
If the vector is not normalized:
- Machine learning models may become unstable.
- Quantum simulation may produce **non-physical states**.

Thus, normalization is the **first step** before any cross-domain operation.

---

### 3. Backend Flexibility
The module is designed to:
- Run **without dependencies** using pure Python lists.
- Automatically use **NumPy** for fast CPU operations.
- Optionally switch to **JAX** or **PyTorch** for GPU/TPU acceleration.

This makes it usable both in **bare-metal research environments** and **production-scale compute clusters**.

---

## File: `tensors.py`

### Features
- **Normalization** – Euclidean vector norm to unit length.
- **Element-wise addition & multiplication** – for linear algebra and machine learning operations.
- **Concatenation** – merging tensors into larger structures.
- **Backend support** – works with Python lists, NumPy arrays, JAX, and PyTorch tensors.

---

## Installation

```bash
pip install numpy
# Optional acceleration:
pip install jax jaxlib
pip install torch
```

---

## Example Usage

```python
from tensors import normalize, tensor_add, tensor_mul, tensor_concat

v1 = [1, 2, 3]
v2 = [4, 5, 6]

print("Normalized:", normalize(v1))
print("Sum:", tensor_add(v1, v2))
print("Product:", tensor_mul(v1, v2))
print("Concatenated:", tensor_concat(v1, v2))
```

Expected output:
```
Normalized: [0.26726, 0.53452, 0.80178]
Sum: [5, 7, 9]
Product: [4, 10, 18]
Concatenated: [1, 2, 3, 4, 5, 6]
```

---

## Function Reference

### `normalize(vec)`
Normalizes a vector to have unit Euclidean norm.

**Args:**
- `vec` – list, NumPy array, JAX array, or PyTorch tensor.

**Returns:**
- Same type as input, normalized.

---

### `tensor_add(a, b)`
Performs element-wise addition.

### `tensor_mul(a, b)`
Performs element-wise multiplication.

### `tensor_concat(a, b)`
Concatenates two tensors end-to-end.

---

## Future Work
- Support higher-order tensors (rank > 1).
- Tensor contraction and decomposition.
- Quantum state embedding functions.

---

## License
MIT License – free for use and modification.
