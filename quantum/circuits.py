# quantum/circuits.py
# -*- coding: utf-8 -*-
"""
Quantum circuits aligned with the Dynamath formalism.

This module provides parameterized, executable circuits that reflect key constructs
from the paper:

- (§6, §18) Embedding and Field Preparation:
    * Amplitude encoding (Initialize) and angle/product-state encoding (Ry) to map
      classical semantic vectors into quantum statevectors.

- (§12, §18) Semantic Field Evolution (idea-field / potential-flow analogue):
    * Layered Hamiltonian-inspired evolution with single-qubit rotations (local
      "potential" terms) and pairwise ZZ couplings (interaction / "curl" analogue).

- (§7) Meta-operators (conjugation):
    * Programmatic operator conjugation g⁻¹ f g to recontextualize a transformation.

- (§8) Phase Dynamics & Attractors (phase-kick):
    * Global phase/controlled phase operations as a proxy for semantic phase flows.

- (§14) Measurement / Information flux:
    * Computational-basis sampling and expectation estimation.

The circuits are compatible with both simulators (Aer) and real IBM Quantum backends.
If Aer is unavailable, fall back uses Statevector routines where feasible.

Usage sketch:

    from quantum.circuits import (
        build_embedding_circuit,
        build_semantic_flow_circuit,
        build_conjugated_circuit,
        measure_all,
        run_counts,
        get_backend
    )

    # (1) Embedding circuit (angle encoding)
    circ = build_embedding_circuit(angles=[0.1, 1.2, -0.7], mode="angle")
    circ = measure_all(circ)
    backend = get_backend()  # Aer or raise if none available
    result = run_counts(circ, backend, shots=2048)
    print(result.get_counts())

    # (2) Dynamic flow circuit with L layers
    flow = build_semantic_flow_circuit(
        n_qubits=4,
        layers=3,
        theta_local=[0.2, -0.3, 0.1, 0.05],   # local fields per qubit
        theta_coupling=0.4,                    # ZZ coupling strength
        phase_kick=0.25                        # global phase
    )
    flow = measure_all(flow)
    result2 = run_counts(flow, backend, shots=4096)

"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Union
import math
import warnings
from spec.spec_loader import get_operator_registry

try:
    # Qiskit >= 1.x
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Statevector, SparsePauliOp
except Exception as e:  # pragma: no cover
    raise ImportError(
        "qiskit is required for quantum circuits. Install with `pip install qiskit`."
    ) from e

# Optional Aer
try:
    from qiskit_aer import Aer
    _AER_AVAILABLE = True
except Exception:
    _AER_AVAILABLE = False


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1) == 0)

def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()

def required_qubits_for_amplitude(length: int) -> int:
    """
    Return the number of qubits needed to amplitude-encode a vector of given length.
    Pads to next power of two if necessary.
    """
    if length <= 0:
        return 1
    L = length if _is_power_of_two(length) else _next_pow2(length)
    return int(math.log2(L))

def _pad_to_pow2(vec: Sequence[float]) -> List[float]:
    n = len(vec)
    if n == 0:
        return [1.0]
    if _is_power_of_two(n):
        return list(vec)
    L = _next_pow2(n)
    out = list(vec) + [0.0] * (L - n)
    return out

def _normalize_l2_real(vec: Sequence[float], eps: float = 1e-12) -> List[float]:
    n2 = sum(x * x for x in vec)
    if n2 < eps:
        # default to |0...0> if vector is near-zero
        return [1.0] + [0.0] * (len(vec) - 1)
    n = math.sqrt(n2)
    return [x / n for x in vec]


# ---------------------------------------------------------------------
# Embedding circuits: amplitude & angle (§6, §18)
# ---------------------------------------------------------------------

def build_embedding_circuit(
    *,
    amplitudes: Optional[Sequence[float]] = None,
    angles: Optional[Sequence[float]] = None,
    mode: str = "amplitude",
    name: str = "embed",
) -> QuantumCircuit:
    """
    Build a state-preparation circuit.

    Parameters
    ----------
    amplitudes : Optional[Sequence[float]]
        Real amplitudes to be L2-normalized and padded to 2^k. Used if mode="amplitude".
    angles : Optional[Sequence[float]]
        Feature angles (radians) for product-state angle encoding with Ry(θ). Used if mode="angle".
    mode : {"amplitude","angle"}
        Choose encoding mode.
    name : str
        Circuit name.

    Returns
    -------
    QuantumCircuit
        Prepared state on n qubits, no measurements.
    """
    if mode not in {"amplitude", "angle"}:
        raise ValueError("mode must be 'amplitude' or 'angle'")

    if mode == "amplitude":
        if amplitudes is None:
            raise ValueError("amplitudes must be provided for amplitude mode")
        amps = _normalize_l2_real(_pad_to_pow2(list(amplitudes)))
        n_qubits = required_qubits_for_amplitude(len(amps))
        qc = QuantumCircuit(n_qubits, name=name)
        # Qiskit initialize expects length 2^n vector
        if len(amps) != (1 << n_qubits):
            # pad again if corner case
            target_len = 1 << n_qubits
            amps = amps[:target_len] + [0.0] * max(0, target_len - len(amps))
        qc.initialize(amps, list(range(n_qubits)))
        return qc

    else:  # angle
        if angles is None:
            raise ValueError("angles must be provided for angle mode")
        theta = list(angles)
        n_qubits = max(1, len(theta))
        qc = QuantumCircuit(n_qubits, name=f"{name}_angle")
        for i, th in enumerate(theta):
            qc.ry(th, i)
        return qc


# ---------------------------------------------------------------------
# Semantic phase & field evolution (§12, §18)
# ---------------------------------------------------------------------

def _layer_semantic_field(
    qc: QuantumCircuit,
    theta_local: Sequence[float],
    theta_coupling: float,
) -> None:
    """
    One "semantic field" layer:
      - local potential rotation RZ (can be RZ or RX; we use RZ for phase-like)
      - mixer RX
      - ZZ couplings across a ring topology
    """
    n = qc.num_qubits
    # Local "potential" terms
    for q, th in enumerate(theta_local):
        qc.rz(float(th), q)
    # Mixer (helps explore manifold)
    for q in range(n):
        qc.rx(0.5 * float(theta_coupling), q)
    # Pairwise "curl-like" ZZ couplings on a ring
    for q in range(n):
        qn = (q + 1) % n
        qc.rzz(float(theta_coupling), q, qn)

def build_semantic_flow_circuit(
    *,
    n_qubits: int,
    layers: int = 2,
    theta_local: Optional[Sequence[float]] = None,
    theta_coupling: float = 0.3,
    phase_kick: float = 0.0,
    name: str = "semantic_flow",
) -> QuantumCircuit:
    """
    Build a layered, Hamiltonian-inspired ansatz capturing Dynamath's semantic flow:
    local "potential" + mixing + ZZ couplings, optionally with a global phase kick.

    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    layers : int
        Number of stacked field-evolution layers.
    theta_local : Optional[Sequence[float]]
        Local rotation parameters per qubit; if None, zeros.
    theta_coupling : float
        ZZ coupling strength shared across pairs per layer.
    phase_kick : float
        Apply a global RZ on all qubits at the end (proxy for §8 phase dynamics).
    name : str
        Circuit name.

    Returns
    -------
    QuantumCircuit
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")

    if theta_local is None:
        theta_local = [0.0] * n_qubits
    if len(theta_local) != n_qubits:
        raise ValueError("theta_local length must equal n_qubits")

    qc = QuantumCircuit(n_qubits, name=name)
    # Initialize to uniform superposition to "feel" couplings
    for q in range(n_qubits):
        qc.h(q)

    for _ in range(max(1, layers)):
        _layer_semantic_field(qc, theta_local, theta_coupling)

    if abs(phase_kick) > 1e-12:
        for q in range(n_qubits):
            qc.rz(float(phase_kick), q)

    return qc


# ---------------------------------------------------------------------
# Meta-operator: conjugation g⁻¹ f g (§7)
# ---------------------------------------------------------------------

def build_conjugated_circuit(
    g_builder,
    f_builder,
    *,
    n_qubits: int,
    name: str = "conjugated"
) -> QuantumCircuit:
    """
    Compose g⁻¹ · f · g on n_qubits, where g_builder and f_builder are callables:
        g = g_builder(n_qubits) -> QuantumCircuit
        f = f_builder(n_qubits) -> QuantumCircuit

    Returns
    -------
    QuantumCircuit
        The conjugated operator as a single circuit.

    Note: For real hardware, the transpiler will inline and optimize the concatenation.
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    g = g_builder(n_qubits)
    f = f_builder(n_qubits)
    # Inverse of g
    g_inv = g.inverse()
    # Stitch
    qc = QuantumCircuit(n_qubits, name=name)
    qc.compose(g, inplace=True)
    qc.compose(f, inplace=True)
    qc.compose(g_inv, inplace=True)
    return qc


# ---------------------------------------------------------------------
# Phase kick / global or controlled phase (§8)
# ---------------------------------------------------------------------

def apply_phase_kick(qc: QuantumCircuit, theta: float, controlled: bool = False) -> QuantumCircuit:
    """
    Apply a phase kick on the last qubit (global or controlled by the first qubit).

    Parameters
    ----------
    qc : QuantumCircuit
    theta : float
        Phase angle (radians).
    controlled : bool
        If True, apply a controlled-phase from qubit 0 to last qubit.

    Returns
    -------
    QuantumCircuit
        Modified circuit (in-place and also returned).
    """
    n = qc.num_qubits
    if n == 0:
        return qc
    target = n - 1
    if controlled and n >= 2:
        qc.cp(float(theta), 0, target)
    else:
        qc.p(float(theta), target)
    return qc

# -- spec-driven operator application --

def _apply_op(qc, op_name: str, params: dict):
    # Минимальный белый список безопасных операторов
    if op_name == "h_gate":
        qc.h(int(params["target_qubit"]))
    elif op_name == "cx_gate":
        qc.cx(int(params["control_qubit"]), int(params["target_qubit"]))
    elif op_name == "rz":
        qc.rz(float(params["theta"]), int(params["target_qubit"]))
    elif op_name == "ry":
        qc.ry(float(params["theta"]), int(params["target_qubit"]))
    elif op_name == "rzz":
        qc.rzz(float(params["theta"]), int(params["control_qubit"]), int(params["target_qubit"]))
    elif op_name == "phase":
        qc.p(float(params["theta"]), int(params["target_qubit"]))
    elif op_name == "measure":
        # Позже можно оптимизировать, но для простоты измерим все
        qc.measure_all()
    else:
        raise ValueError(f"Unsupported operator in spec: {op_name}")

def build_circuit_from_spec(n_qubits: int, sequence: list[dict], name: str = "spec_circuit"):
    """
    sequence: список шагов вида:
      {"name": "h_gate", "params": {"target_qubit": 0}}
      {"name": "cx_gate", "params": {"control_qubit": 0, "target_qubit": 1}}
      {"name": "rz", "params": {"target_qubit": 0, "theta": 0.3}}
      {"name": "measure", "params": {}}
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    qc = QuantumCircuit(n_qubits, name=name)
    for step in sequence:
        _apply_op(qc, step["name"], step.get("params", {}))
    return qc

def build_circuit_from_registry(n_qubits: int, name: str = "spec_registry_example"):
    """
    Пример: читает operators.json (как реестр доступных операторов),
    но последовательность шагов ты передаёшь сам (см. build_circuit_from_spec).
    Эта функция полезна, если хочешь сперва проверить, что нужные операторы поддерживаются.
    """
    registry = get_operator_registry()
    supported = {op["name"] for op in registry.get("quantum", [])}
    # просто возвращаем список/набор, чтобы можно было валидировать снаружи
    return supported

# ---------------------------------------------------------------------
# Measurement & Execution (§14)
# ---------------------------------------------------------------------

def measure_all(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Add measurements on all qubits into a classical register of the same size.
    """
    n = qc.num_qubits
    measured = qc.copy()
    measured.measure_all()
    # (Qiskit auto-creates classical bits on measure_all)
    return measured

def get_backend(backend_name: Optional[str] = None):
    """
    Return a Qiskit backend. Defaults to Aer simulator if available.

    Parameters
    ----------
    backend_name : Optional[str]
        - None or "aer_simulator": return Aer simulator (if available).
        - Any other string: try to load Aer backend by that name, else raise.
          For IBM Quantum backends, pass an already-initialized backend to run_counts() directly.

    Returns
    -------
    Backend-like object
    """
    if backend_name is None or backend_name == "aer_simulator":
        if not _AER_AVAILABLE:
            raise RuntimeError(
                "Aer not available. Install with `pip install qiskit-aer` "
                "or supply a real IBM backend to run_counts()."
            )
        return Aer.get_backend("aer_simulator")
    else:
        if not _AER_AVAILABLE:
            raise RuntimeError("Requested Aer backend but qiskit-aer is not installed.")
        return Aer.get_backend(backend_name)

def run_counts(
    qc: QuantumCircuit,
    backend,
    *,
    shots: int = 4096,
    seed_simulator: Optional[int] = None,
    optimization_level: int = 3,
) -> "qiskit.result.Result":
    """
    Transpile and execute a circuit, returning a Result with counts for measured circuits.

    For IBM backends (non-Aer), pass a backend obtained from IBM Runtime/Provider.
    """
    # For Aer, set seed if provided
    run_args = {}
    if _AER_AVAILABLE and hasattr(backend, "set_options"):
        if seed_simulator is not None:
            backend.set_options(seed_simulator=seed_simulator)
        run_args["shots"] = shots

    tqc = transpile(qc, backend=backend, optimization_level=optimization_level)
    job = backend.run(tqc, shots=shots, **run_args)
    return job.result()

def expectation_z(
    qc: QuantumCircuit,
    backend,
    *,
    shots: int = 4096,
    qubits: Optional[Sequence[int]] = None,
    seed_simulator: Optional[int] = None,
) -> Dict[int, float]:
    """
    Estimate ⟨Z⟩ for each specified qubit via sampling.

    Returns a dict: qubit_index -> expectation in [-1, 1].
    """
    n = qc.num_qubits
    if qubits is None:
        qubits = list(range(n))
    # Build measure-all version
    measured = measure_all(qc)
    res = run_counts(measured, backend, shots=shots, seed_simulator=seed_simulator)
    counts = res.get_counts()

    # Compute expectation: ⟨Z⟩ = P(0) - P(1) per qubit, derived from bitstrings.
    # Qiskit bitstring order: by default, classical bits map little-endian; we adhere to string indexing.
    totals = {q: 0 for q in qubits}
    for bitstr, c in counts.items():
        # bitstr like '0101' with leftmost the highest classical bit
        # Map to per-qubit contributions:
        # We assume measure_all used default mapping 1-1 qubit->clbit with reversed order in string.
        for q in qubits:
            # Position in string: Qiskit uses big-endian string (leftmost c_(n-1))
            bit = bitstr[n - 1 - q]
            totals[q] += (1 if bit == "0" else -1) * c

    expectations = {q: totals[q] / float(shots) for q in qubits}
    return expectations


# ---------------------------------------------------------------------
# Pauli expectation via SparsePauliOp (statevector path)
# ---------------------------------------------------------------------

def pauli_expectation_statevector(
    qc: QuantumCircuit,
    pauli_terms: Sequence[Tuple[str, float]],
) -> float:
    """
    Compute ⟨ψ| H |ψ⟩ for a Pauli-sum H using Statevector simulation.

    Parameters
    ----------
    qc : QuantumCircuit
        Unmeasured circuit.
    pauli_terms : list of (label, coeff)
        Example: [("ZZII", 0.3), ("IXIX", -0.2)]

    Returns
    -------
    float
        Expectation value.

    Notes
    -----
    - This path does not require Aer; uses qiskit.quantum_info.Statevector.
    - For hardware execution, use primitives (Estimator) or measurement-based techniques.
    """
    psi = Statevector.from_instruction(qc)
    H = SparsePauliOp.from_list([(lbl, coef) for (lbl, coef) in pauli_terms])
    val = psi.expectation_value(H)
    # Older Qiskit versions may return complex ~0j; take real part safely.
    return float(val.real)


# ---------------------------------------------------------------------
# Example builders (for unit tests / demos)
# ---------------------------------------------------------------------

def _g_hadamards(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="g_H")
    for q in range(n_qubits):
        qc.h(q)
    return qc

def _f_ring_zz(n_qubits: int, lam: float = 0.4) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="f_ZZ")
    for q in range(n_qubits):
        qc.rzz(lam, q, (q + 1) % n_qubits)
    return qc


# ---------------------------------------------------------------------
# Self-check (basic smoke tests; safe to run on simulators)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # (1) Angle embedding + measurement
    circ1 = build_embedding_circuit(angles=[0.1, 1.2, -0.7], mode="angle")
    circ1 = measure_all(circ1)
    if _AER_AVAILABLE:
        backend = get_backend()
        r1 = run_counts(circ1, backend, shots=1024, seed_simulator=7)
        print("[SMOKE] counts (angle embed):", r1.get_counts())

    # (2) Amplitude embedding + expectation via statevector
    amps = [0.2, -1.0, 0.5]
    circ2 = build_embedding_circuit(amplitudes=amps, mode="amplitude")
    exp = pauli_expectation_statevector(circ2, [("Z", 1.0)])
    print("[SMOKE] <Z> on |ψ>: ", exp)

    # (3) Semantic flow circuit
    flow = build_semantic_flow_circuit(
        n_qubits=3, layers=2,
        theta_local=[0.2, -0.3, 0.1],
        theta_coupling=0.35,
        phase_kick=0.12
    )
    flow_m = measure_all(flow)
    if _AER_AVAILABLE:
        r2 = run_counts(flow_m, get_backend(), shots=2048, seed_simulator=42)
        print("[SMOKE] counts (semantic flow):", r2.get_counts())

    # (4) Conjugation test g⁻¹ f g
    conj = build_conjugated_circuit(_g_hadamards, _f_ring_zz, n_qubits=3)
    conj_m = measure_all(conj)
    if _AER_AVAILABLE:
        r3 = run_counts(conj_m, get_backend(), shots=1024, seed_simulator=123)
        print("[SMOKE] counts (conjugated):", r3.get_counts())

    print("circuits.py self-check finished.")
