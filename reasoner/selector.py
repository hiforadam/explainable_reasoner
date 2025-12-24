"""Selector model for fluent token selection."""
import json
import pickle
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any


@dataclass
class SelectorArtifacts:
    """Selector model artifacts."""
    vocab: List[str]
    token_to_id: Dict[str, int]
    trigram_logp: Dict[str, List[Tuple[int, float]]]
    bigram_logp: Dict[str, List[Tuple[int, float]]]
    unigram_logp: List[float]
    meta: Dict[str, Any]

    def save(self, folder: str) -> None:
        """Save selector - now handled by ReasonerArtifacts.save() with selector parameter."""
        # This method is kept for backward compatibility but does nothing
        # Selector is saved together with reasoner in model.npz
        pass

    @classmethod
    def load(cls, folder: str) -> "SelectorArtifacts":
        """Load selector from folder - reads from model.npz."""
        import os
        
        # Load vocab.json (shared with reasoner)
        vocab_path = os.path.join(folder, "vocab.json")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"vocab.json not found in {folder}")
        
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)["vocab"]
        token_to_id = {t: i for i, t in enumerate(vocab)}
        
        # Try different formats
        model_path = os.path.join(folder, "model.npz")
        selector_json_path = os.path.join(folder, "selector.json")
        selector_npz_path = os.path.join(folder, "selector.npz")
        
        if os.path.exists(model_path):
            # New unified format: model.npz (contains everything)
            m = np.load(model_path, allow_pickle=True)
            if "selector_data" in m:
                selector_data = pickle.loads(m["selector_data"].item())
                tri = selector_data.get("trigram_logp", {})
                bi = selector_data.get("bigram_logp", {})
                uni = selector_data.get("unigram_logp", [])
                meta = selector_data.get("meta", {}) or {}
            else:
                raise FileNotFoundError(f"Selector data not found in model.npz")
        elif os.path.exists(selector_json_path):
            # Old format: selector.json
            with open(selector_json_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            tri = {k: [(int(i), float(lp)) for i, lp in v] for k, v in payload.get("trigram_logp", {}).items()}
            bi = {k: [(int(i), float(lp)) for i, lp in v] for k, v in payload.get("bigram_logp", {}).items()}
            uni = [float(x) for x in payload.get("unigram_logp", [])]
            meta = payload.get("meta", {}) or {}
        elif os.path.exists(selector_npz_path):
            # Old format: selector.npz (with pickle)
            s = np.load(selector_npz_path, allow_pickle=True)
            selector_data = pickle.loads(s["data"].item())
            tri = selector_data.get("trigram_logp", {})
            bi = selector_data.get("bigram_logp", {})
            uni = selector_data.get("unigram_logp", [])
            meta = selector_data.get("meta", {}) or {}
        else:
            raise FileNotFoundError(f"Selector files not found in {folder}")
        
        return cls(vocab=vocab, token_to_id=token_to_id, trigram_logp=tri, bigram_logp=bi, unigram_logp=uni, meta=meta)

    def candidates(self, prev2: Optional[int], prev1: Optional[int], top_k: int = 24) -> List[Tuple[int, float]]:
        """Get candidate tokens with log probabilities."""
        if prev2 is not None and prev1 is not None:
            key = f"{int(prev2)},{int(prev1)}"
            if key in self.trigram_logp and self.trigram_logp[key]:
                return self.trigram_logp[key][:max(1, int(top_k))]
        if prev1 is not None:
            key = str(int(prev1))
            if key in self.bigram_logp and self.bigram_logp[key]:
                return self.bigram_logp[key][:max(1, int(top_k))]
        if self.unigram_logp:
            arr = np.array(self.unigram_logp, dtype=np.float64)
            k = min(max(1, int(top_k)), arr.shape[0])
            idx = np.argpartition(-arr, k-1)[:k]
            idx = idx[np.argsort(-arr[idx])]
            return [(int(i), float(arr[i])) for i in idx]
        return []


def _log_normalize(counts: Dict[int, float], alpha: float = 0.5) -> List[Tuple[int, float]]:
    """Normalize counts to log probabilities."""
    import math
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


def train_selector_from_seqs(vocab: List[str], seqs: List[List[int]], smooth: float = 0.5,
                             max_per_context: int = 256) -> SelectorArtifacts:
    """Train selector model from sequences."""
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
    
    # Unigram logp
    uni = uni + smooth
    uni = uni / np.sum(uni)
    unigram_logp = np.log(uni + 1e-12).tolist()
    
    # Bigram logp
    bigram_logp: Dict[str, List[Tuple[int, float]]] = {}
    for p1, counts in bi_counts.items():
        arr = _log_normalize(counts, alpha=smooth)[:max_per_context]
        bigram_logp[str(int(p1))] = arr
    
    # Trigram logp
    trigram_logp: Dict[str, List[Tuple[int, float]]] = {}
    for (p2, p1), counts in tri_counts.items():
        arr = _log_normalize(counts, alpha=smooth)[:max_per_context]
        trigram_logp[f"{int(p2)},{int(p1)}"] = arr
    
    meta = {"smooth": float(smooth), "max_per_context": int(max_per_context)}
    return SelectorArtifacts(
        vocab=vocab,
        token_to_id=token_to_id,
        trigram_logp=trigram_logp,
        bigram_logp=bigram_logp,
        unigram_logp=unigram_logp,
        meta=meta
    )
