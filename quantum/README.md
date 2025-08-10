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
