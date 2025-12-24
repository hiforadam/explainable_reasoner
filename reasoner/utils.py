"""Utility functions."""
import numpy as np
from typing import List


def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Normalize rows of matrix.
    
    Args:
        x: Input matrix (N, D)
        eps: Epsilon for numerical stability
    
    Returns:
        Normalized matrix
    """
    if x.size == 0:
        return x
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def softmax(logits: np.ndarray, temp: float = 1.0) -> np.ndarray:
    """
    Softmax with temperature.
    
    Args:
        logits: Input logits array
        temp: Temperature parameter
    
    Returns:
        Probability distribution
    """
    if logits.size == 0:
        return logits
    if temp <= 0:
        raise ValueError("Temperature must be positive")
    z = logits / max(temp, 1e-8)
    z = z - np.max(z)
    e = np.exp(np.clip(z, -500, 500))  # Clip to prevent overflow
    s = np.sum(e)
    if s == 0 or np.isnan(s) or np.isinf(s):
        # Fallback to uniform
        return np.ones_like(e) / len(e)
    return e / s

