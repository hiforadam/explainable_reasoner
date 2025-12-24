"""Model artifacts storage."""
import json
import pickle
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple


@dataclass
class ReasonerArtifacts:
    """Reasoner model artifacts."""
    vocab: List[str]
    token_to_id: Dict[str, int]
    vectors: np.ndarray  # (V, D) normalized semantic vectors
    bigram_logp: Dict[str, List[Tuple[int, float]]]  # sparse: token_id -> [(next_id, logp), ...]
    roles: Dict[str, str]  # token -> role
    neighbors: Dict[str, Dict[str, List[str]]]  # token -> {before:[...], after:[...]}
    contradiction_pairs: List[Dict[str, Any]]  # list of {a,b,penalty,reason}
    
    # Optional artifacts
    clusters: Dict[str, int] = field(default_factory=dict)
    meta_cluster_ids: List[int] = field(default_factory=list)
    sentence_end_tokens: List[str] = field(default_factory=list)
    discourse_role_names: List[str] = field(default_factory=list)
    discourse_level_names: List[str] = field(default_factory=list)
    token_role_to_level: Dict[str, str] = field(default_factory=dict)
    discourse_trans: Any = field(default_factory=list)
    discourse_emit_logp: Any = field(default_factory=list)
    discourse_pos_thresholds: List[int] = field(default_factory=list)
    discourse_trans_pos: Any = field(default_factory=list)
    discourse_emit_logp_pos: Any = field(default_factory=list)
    discourse_role_min_run: List[int] = field(default_factory=list)
    learning_signals: Dict[str, Any] = field(default_factory=dict)

    def save(self, folder: str, selector: Optional[Any] = None) -> None:
        """Save all artifacts to folder - vocab.json and model.npz (includes selector if provided)."""
        import os
        os.makedirs(folder, exist_ok=True)
        
        # vocab.json (tokenizer)
        with open(os.path.join(folder, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump({"vocab": self.vocab}, f, ensure_ascii=False, indent=2)
        
        # Helper function to convert numpy types to Python types
        def _tolist(x: Any) -> Any:
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (np.integer, np.int32, np.int64, np.int_, np.uint32, np.uint64)):
                return int(x)
            if isinstance(x, (np.floating, np.float32, np.float64)):
                return float(x)
            if isinstance(x, (list, tuple)):
                return [_tolist(item) for item in x]
            if isinstance(x, dict):
                return {k: _tolist(v) for k, v in x.items()}
            return x
        
        # Prepare reasoner metadata
        reasoner_meta = {
            "roles": self.roles,
            "neighbors": self.neighbors,
            "contradiction_pairs": self.contradiction_pairs,
            "clusters": self.clusters,
            "meta_cluster_ids": self.meta_cluster_ids,
            "sentence_end_tokens": self.sentence_end_tokens,
            "discourse_role_names": self.discourse_role_names,
            "discourse_level_names": self.discourse_level_names,
            "token_role_to_level": self.token_role_to_level,
            "discourse_trans": _tolist(self.discourse_trans),
            "discourse_emit_logp": _tolist(self.discourse_emit_logp),
            "discourse_pos_thresholds": _tolist(self.discourse_pos_thresholds),
            "discourse_trans_pos": _tolist(self.discourse_trans_pos),
            "discourse_emit_logp_pos": _tolist(self.discourse_emit_logp_pos),
            "discourse_role_min_run": _tolist(self.discourse_role_min_run),
            "learning_signals": self.learning_signals,
        }
        
        # Prepare selector data if provided
        selector_data = None
        if selector is not None:
            selector_data = {
                "trigram_logp": selector.trigram_logp,
                "bigram_logp": selector.bigram_logp,
                "unigram_logp": selector.unigram_logp,
                "meta": selector.meta or {},
            }
        
        # Prepare bigram_logp for saving (sparse format)
        bigram_data = {
            "bigram_logp": self.bigram_logp,
        }
        
        # Save everything in model.npz
        save_dict = {
            "vectors": self.vectors.astype(np.float32),
            "bigram_data": pickle.dumps(bigram_data, protocol=pickle.HIGHEST_PROTOCOL),
            "reasoner_meta": pickle.dumps(reasoner_meta, protocol=pickle.HIGHEST_PROTOCOL),
        }
        
        if selector_data is not None:
            save_dict["selector_data"] = pickle.dumps(selector_data, protocol=pickle.HIGHEST_PROTOCOL)
        
        np.savez_compressed(os.path.join(folder, "model.npz"), **save_dict)

    @classmethod
    def load(cls, folder: str) -> "ReasonerArtifacts":
        """Load artifacts from folder - supports multiple formats."""
        import os
        
        # Load vocab.json
        vocab_path = os.path.join(folder, "vocab.json")
        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"vocab.json not found in {folder}")
        
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab = json.load(f)["vocab"]
        token_to_id = {t: i for i, t in enumerate(vocab)}
        
        # Try different formats
        model_path = os.path.join(folder, "model.npz")
        weights_path = os.path.join(folder, "weights.npz")
        meta_path = os.path.join(folder, "meta.json")
        
        if os.path.exists(model_path):
            # New unified format: model.npz (contains everything)
            w = np.load(model_path, allow_pickle=True)
            if "reasoner_meta" in w:
                reasoner_meta = pickle.loads(w["reasoner_meta"].item())
            else:
                # Fallback: try old pickle format
                reasoner_meta = pickle.loads(w["meta"].item()) if "meta" in w else {}
            
            # Load bigram_logp (sparse or dense)
            if "bigram_data" in w:
                bigram_data = pickle.loads(w["bigram_data"].item())
                bigram_logp = bigram_data.get("bigram_logp", {})
            elif "bigram_logp" in w:
                # Old format: dense matrix - convert to sparse (top-150 per row)
                bigram_dense = w["bigram_logp"]
                bigram_logp = {}
                for i in range(bigram_dense.shape[0]):
                    row = bigram_dense[i]
                    top_indices = np.argsort(-row)[:150]
                    bigram_logp[str(i)] = [(int(j), float(row[j])) for j in top_indices if row[j] > -50.0]
            else:
                bigram_logp = {}
        elif os.path.exists(weights_path) and os.path.exists(meta_path):
            # Old format: weights.npz + meta.json
            w = np.load(weights_path)
            with open(meta_path, "r", encoding="utf-8") as f:
                reasoner_meta = json.load(f)
            # Convert dense to sparse
            bigram_dense = w["bigram_logp"]
            bigram_logp = {}
            for i in range(bigram_dense.shape[0]):
                row = bigram_dense[i]
                top_indices = np.argsort(-row)[:150]
                bigram_logp[str(i)] = [(int(j), float(row[j])) for j in top_indices if row[j] > -50.0]
        else:
            raise FileNotFoundError(f"Model files not found in {folder}")

        return cls(
            vocab=vocab,
            token_to_id=token_to_id,
            vectors=w["vectors"],
            bigram_logp=bigram_logp,
            roles=reasoner_meta.get("roles", {}),
            neighbors=reasoner_meta.get("neighbors", {}),
            contradiction_pairs=reasoner_meta.get("contradiction_pairs", []),
            clusters=reasoner_meta.get("clusters", {}),
            meta_cluster_ids=reasoner_meta.get("meta_cluster_ids", []),
            sentence_end_tokens=reasoner_meta.get("sentence_end_tokens", []),
            discourse_role_names=reasoner_meta.get("discourse_role_names", []),
            discourse_level_names=reasoner_meta.get("discourse_level_names", []),
            token_role_to_level=reasoner_meta.get("token_role_to_level", {}),
            discourse_trans=reasoner_meta.get("discourse_trans", []),
            discourse_emit_logp=reasoner_meta.get("discourse_emit_logp", []),
            discourse_pos_thresholds=reasoner_meta.get("discourse_pos_thresholds", []),
            discourse_trans_pos=reasoner_meta.get("discourse_trans_pos", []),
            discourse_emit_logp_pos=reasoner_meta.get("discourse_emit_logp_pos", []),
            discourse_role_min_run=reasoner_meta.get("discourse_role_min_run", []),
            learning_signals=reasoner_meta.get("learning_signals", {}),
        )
