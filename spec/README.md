# Specification for Dynamath Hybrid

This folder contains formal specifications for data formats, API contracts, and operator definitions used in **Dynamath Hybrid**.

## Files

- **`hybrid.toml`** — global configuration file for the hybrid computation system.
- **`ontology.proto`** — Protocol Buffers schema defining core data structures.
- **`operators.json`** — list of supported tensor and quantum operators.

---

## Data Flow

1. **Tensor Layer (Core)**
   - Input: Numeric arrays or tensors
   - Normalization: Optional
   - Output: Encoded feature vectors

2. **Hybrid Bridge**
   - Converts tensor data into quantum state vectors.
   - Prepares them for execution on Qiskit/Cirq.

3. **Quantum Layer**
   - Executes the circuit on a backend (simulator or quantum hardware).
   - Returns measurement results.

---

## Encoding

- **Float32/Float64** for tensor values.
- **Complex amplitudes** for quantum states.
- All serialized data should be compatible with `ontology.proto`.

---

## Versioning

This specification is version-controlled alongside the source code.
Breaking changes must increment the `spec_version` in `hybrid.toml`.
