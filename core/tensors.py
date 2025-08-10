"""
tensors.py — базовые тензорные операции для Dynamath Hybrid.
Здесь будут функции для создания, нормализации и комбинирования тензоров.

TODO: реализовать операции на JAX/PyTorch
"""

def normalize(vec):
    """Нормализует вектор."""
    norm = sum(x*x for x in vec) ** 0.5
    return [x / norm for x in vec] if norm != 0 else vec
