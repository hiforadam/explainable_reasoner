import json
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any

@dataclass
class ReasonerArtifacts:
    vocab: List[str]
    token_to_id: Dict[str, int]
    vectors: np.ndarray              # (V, D) normalized
    bigram_logp: np.ndarray          # (V, V) log-prob next given prev
    roles: Dict[str, str]            # token -> role
    neighbors: Dict[str, Dict[str, List[str]]]  # token -> {before:[...], after:[...]}
    contradiction_pairs: List[Dict[str, Any]]    # list of {a,b,penalty,reason}

    # Optional, purely derived artifacts (no hardcoded token lists)
    clusters: Dict[str, int] = field(default_factory=dict)           # token -> cluster_id
    meta_cluster_ids: List[int] = field(default_factory=list)        # cluster ids considered "dataset/meta"
    sentence_end_tokens: List[str] = field(default_factory=list)     # punctuation tokens that often end sentences

    # Discourse role model (token-by-token explicit hidden state)
    discourse_role_names: List[str] = field(default_factory=list)    # Learned from data (no hardcoded names)
    discourse_level_names: List[str] = field(default_factory=list)   # Learned from data (no hardcoded names)
    token_role_to_level: Dict[str, str] = field(default_factory=dict)  # Learned mapping: token_role -> level
    discourse_trans: Any = field(default_factory=list)               # (R,R) transition probs (list or ndarray)
    discourse_emit_logp: Any = field(default_factory=list)           # (R,L) log-prob token-level given role

    # NEW: position-conditioned discourse stats
    discourse_pos_thresholds: List[int] = field(default_factory=list)  # [q33_idx, q66_idx] token index thresholds within sentence
    discourse_trans_pos: Any = field(default_factory=list)             # (B,R,R) transition probs per position bucket
    discourse_emit_logp_pos: Any = field(default_factory=list)         # (B,R,L) log-prob token-level given role per position bucket
    discourse_role_min_run: List[int] = field(default_factory=list)    # (R,) minimum run length per role

    def save(self, folder: str) -> None:
        import os
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab}, f, ensure_ascii=False, indent=2)
        np.savez_compressed(os.path.join(folder, "weights.npz"), vectors=self.vectors.astype(np.float32), bigram_logp=self.bigram_logp.astype(np.float32))
        def _tolist(x: Any) -> Any:
            try:
                if isinstance(x, np.ndarray):
                    return x.tolist()
                if isinstance(x, (np.integer, np.int32, np.int64, np.int_, np.uint32, np.uint64)):
                    return int(x)
                if isinstance(x, (np.floating, np.float32, np.float64, np.float_)):
                    return float(x)
                if isinstance(x, np.generic):
                    return x.item()
                if isinstance(x, (list, tuple)):
                    return [_tolist(item) for item in x]
                if isinstance(x, dict):
                    return {k: _tolist(v) for k, v in x.items()}
                return x
            except Exception:
                try:
                    if hasattr(x, 'item'):
                        return x.item()
                    if hasattr(x, '__float__'):
                        return float(x)
                    if hasattr(x, '__int__'):
                        return int(x)
                except Exception:
                    pass
                return x
        with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
            json.dump({"roles": self.roles, "neighbors": self.neighbors, "contradiction_pairs": self.contradiction_pairs,
                      "clusters": self.clusters, "meta_cluster_ids": self.meta_cluster_ids, "sentence_end_tokens": self.sentence_end_tokens,
                      "discourse_role_names": self.discourse_role_names, "discourse_level_names": self.discourse_level_names,
                      "token_role_to_level": self.token_role_to_level, "discourse_trans": _tolist(self.discourse_trans),
                      "discourse_emit_logp": _tolist(self.discourse_emit_logp), "discourse_pos_thresholds": _tolist(self.discourse_pos_thresholds),
                      "discourse_trans_pos": _tolist(self.discourse_trans_pos), "discourse_emit_logp_pos": _tolist(self.discourse_emit_logp_pos),
                      "discourse_role_min_run": _tolist(self.discourse_role_min_run)}, f, ensure_ascii=False, indent=2, default=lambda o: _tolist(o))

    @classmethod
    def load(cls, folder: str) -> "ReasonerArtifacts":
        import os

        with open(os.path.join(folder, "vocab.json"), "r", encoding="utf-8") as f:
            vocab = json.load(f)["vocab"]
        token_to_id = {t: i for i, t in enumerate(vocab)}
        w = np.load(os.path.join(folder, "weights.npz"))
        with open(os.path.join(folder, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)

        return cls(
            vocab=vocab,
            token_to_id=token_to_id,
            vectors=w["vectors"],
            bigram_logp=w["bigram_logp"],
            roles=meta.get("roles", {}),
            neighbors=meta.get("neighbors", {}),
            contradiction_pairs=meta.get("contradiction_pairs", []),
            clusters=meta.get("clusters", {}),
            meta_cluster_ids=meta.get("meta_cluster_ids", []),
            sentence_end_tokens=meta.get("sentence_end_tokens", []),
            discourse_role_names=meta.get("discourse_role_names", []),
            discourse_level_names=meta.get("discourse_level_names", []),
            token_role_to_level=meta.get("token_role_to_level", {}),
            discourse_trans=meta.get("discourse_trans", []),
            discourse_emit_logp=meta.get("discourse_emit_logp", []),
            discourse_pos_thresholds=meta.get("discourse_pos_thresholds", []),
            discourse_trans_pos=meta.get("discourse_trans_pos", []),
            discourse_emit_logp_pos=meta.get("discourse_emit_logp_pos", []),
            discourse_role_min_run=meta.get("discourse_role_min_run", []),
        )

# Removed: CriticArtifacts and ControllerArtifacts - no longer needed
# Using rule-based quality scoring instead of trained models

