import numpy as np
from typing import Dict, List, Tuple

def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)

def cosine(a: np.ndarray, b: np.ndarray, eps: float=1e-8) -> float:
    return float(a.dot(b) / ((np.linalg.norm(a)+eps)*(np.linalg.norm(b)+eps)))

def softmax(logits: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = logits / max(temp, 1e-8)
    z = z - np.max(z)
    e = np.exp(z)
    return e / np.sum(e)

def topk_indices(x: np.ndarray, k: int) -> List[int]:
    if k <= 0:
        return []
    k = min(k, x.shape[0])
    return list(np.argpartition(-x, k-1)[:k])
