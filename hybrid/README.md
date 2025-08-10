## bridges.py

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
