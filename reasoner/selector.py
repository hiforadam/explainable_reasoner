import json
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np


@dataclass
class SelectorArtifacts:
    # Uses the same vocab/token_to_id as ReasonerArtifacts (shared ID space).
    vocab: List[str]
    token_to_id: Dict[str, int]

    # Sparse transition tables:
    # trigram_logp: key "a,b" (ids) -> list[(next_id, logp)] sorted desc
    trigram_logp: Dict[str, List[Tuple[int, float]]]
    # bigram_logp: key "a" (id) -> list[(next_id, logp)] sorted desc
    bigram_logp: Dict[str, List[Tuple[int, float]]]
    # unigram log-prob for fallback
    unigram_logp: List[float]

    meta: Dict[str, Any]

    def save(self, folder: str) -> None:
        import os
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, "selector.json")
        def conv(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, tuple):
                return [conv(x) for x in obj]
            if isinstance(obj, list):
                return [conv(x) for x in obj]
            if isinstance(obj, dict):
                return {str(k): conv(v) for k, v in obj.items()}
            return obj

        payload = {
            "vocab": self.vocab,
            "trigram_logp": {k: [[int(i), float(lp)] for i, lp in v] for k, v in self.trigram_logp.items()},
            "bigram_logp": {k: [[int(i), float(lp)] for i, lp in v] for k, v in self.bigram_logp.items()},
            "unigram_logp": [float(x) for x in self.unigram_logp],
            "meta": conv(self.meta or {}),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, folder: str) -> "SelectorArtifacts":
        import os
        path = os.path.join(folder, "selector.json")
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        vocab = payload["vocab"]
        token_to_id = {t: i for i, t in enumerate(vocab)}
        tri = {k: [(int(i), float(lp)) for i, lp in v] for k, v in payload.get("trigram_logp", {}).items()}
        bi = {k: [(int(i), float(lp)) for i, lp in v] for k, v in payload.get("bigram_logp", {}).items()}
        uni = [float(x) for x in payload.get("unigram_logp", [])]
        meta = payload.get("meta", {}) or {}
        return cls(vocab=vocab, token_to_id=token_to_id, trigram_logp=tri, bigram_logp=bi, unigram_logp=uni, meta=meta)

    def candidates(self, prev2: Optional[int], prev1: Optional[int], top_k: int = 24) -> List[Tuple[int, float]]:
        # Returns list of (token_id, logp) best-first.
        if prev2 is not None and prev1 is not None:
            key = f"{int(prev2)},{int(prev1)}"
            if key in self.trigram_logp and self.trigram_logp[key]:
                return self.trigram_logp[key][:max(1, int(top_k))]
        if prev1 is not None:
            key = str(int(prev1))
            if key in self.bigram_logp and self.bigram_logp[key]:
                return self.bigram_logp[key][:max(1, int(top_k))]
        # unigram fallback: return top-k tokens
        if self.unigram_logp:
            arr = np.array(self.unigram_logp, dtype=np.float64)
            k = min(max(1, int(top_k)), arr.shape[0])
            idx = np.argpartition(-arr, k-1)[:k]
            idx = idx[np.argsort(-arr[idx])]
            return [(int(i), float(arr[i])) for i in idx]
        return []


def _log_normalize(counts: Dict[int, float], alpha: float = 0.5) -> List[Tuple[int, float]]:
    # add-alpha smoothing over observed items only (keeps sparse)
    items = list(counts.items())
    if not items:
        return []
    total = sum(v for _, v in items) + alpha * len(items)
    out = []
    for k, v in items:
        p = (v + alpha) / max(1e-12, total)
        out.append((int(k), float(math.log(p + 1e-12))))
    out.sort(key=lambda x: x[1], reverse=True)
    return out


def train_selector_from_seqs(
    vocab: List[str],
    seqs: List[List[int]],
    smooth: float = 0.5,
    max_per_context: int = 256,
) -> SelectorArtifacts:
    V = len(vocab)
    token_to_id = {t: i for i, t in enumerate(vocab)}

    uni = np.zeros((V,), dtype=np.float64)
    bi_counts: Dict[int, Dict[int, float]] = {}
    tri_counts: Dict[Tuple[int, int], Dict[int, float]] = {}

    for s in seqs:
        for i, tid in enumerate(s):
            uni[tid] += 1.0
            if i >= 1:
                p1 = s[i-1]
                bi_counts.setdefault(p1, {})
                bi_counts[p1][tid] = bi_counts[p1].get(tid, 0.0) + 1.0
            if i >= 2:
                p2 = s[i-2]
                p1 = s[i-1]
                tri_counts.setdefault((p2, p1), {})
                tri_counts[(p2, p1)][tid] = tri_counts[(p2, p1)].get(tid, 0.0) + 1.0

    # unigram logp
    uni = uni + smooth
    uni = uni / np.sum(uni)
    unigram_logp = np.log(uni + 1e-12).tolist()

    bigram_logp: Dict[str, List[Tuple[int, float]]] = {}
    for p1, counts in bi_counts.items():
        arr = _log_normalize(counts, alpha=smooth)[:max_per_context]
        bigram_logp[str(int(p1))] = arr

    trigram_logp: Dict[str, List[Tuple[int, float]]] = {}
    for (p2, p1), counts in tri_counts.items():
        arr = _log_normalize(counts, alpha=smooth)[:max_per_context]
        trigram_logp[f"{int(p2)},{int(p1)}"] = arr

    meta = {"smooth": float(smooth), "max_per_context": int(max_per_context)}
    return SelectorArtifacts(vocab=vocab, token_to_id=token_to_id, trigram_logp=trigram_logp, bigram_logp=bigram_logp, unigram_logp=unigram_logp, meta=meta)
