import numpy as np
import re
from typing import List, Dict, Any, Tuple, Optional
from .tokenizer import ClosedVocabTokenizer
from .model import ReasonerArtifacts
from .selector import SelectorArtifacts
from .utils import softmax

def _build_contradiction_lookup(pairs: List[Dict[str, Any]], token_to_id: Dict[str, int]) -> Dict[Tuple[int, int], float]:
    """Build contradiction lookup dictionary from pairs."""
    out = {}
    for p in pairs:
        a, b = token_to_id.get(p.get("a")), token_to_id.get(p.get("b"))
        if a is not None and b is not None:
            pen = float(p.get("penalty", 0.0))
            if pen > 0:
                out[(a, b)] = out[(b, a)] = max(out.get((a, b), 0.0), pen)
    return out

def build_contra_lookup(art: ReasonerArtifacts) -> Dict[Tuple[int, int], float]:
    """Build contradiction lookup from ReasonerArtifacts."""
    return _build_contradiction_lookup(art.contradiction_pairs or [], art.token_to_id)

def _repetition_penalty(candidate_id: int, recent: List[int], base_penalty: float) -> float:
    return float(base_penalty * sum(1 for r in recent if r == candidate_id)) if base_penalty > 0 else 0.0

def _semantic_repeat_penalty(candidate_id: int, recent: List[int], vectors: np.ndarray,
                            threshold: float, base_penalty: float) -> float:
    if base_penalty <= 0 or not recent:
        return 0.0
    smax = max([float(vectors[r] @ vectors[candidate_id]) for r in recent], default=0.0)
    return float(base_penalty * (smax - threshold) / max(1e-6, 1.0 - threshold)) if smax > threshold else 0.0

def _cluster_switch_penalty(candidate_id: int, recent: List[int], clusters_by_id: Optional[List[int]],
                           window: int, base_penalty: float, roles_by_id: Optional[List[str]] = None) -> float:
    if base_penalty <= 0 or clusters_by_id is None or window <= 0 or (roles_by_id and roles_by_id[candidate_id] == "punct"):
        return 0.0
    matches = [r for r in recent[-window:] if (not roles_by_id or roles_by_id[r] != "punct") and clusters_by_id[r] == clusters_by_id[candidate_id]]
    return float(base_penalty * len(matches) / max(1, len([r for r in recent[-window:] if not roles_by_id or roles_by_id[r] != "punct"])))



def _entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1.0)
    p = p / np.sum(p)
    return float(-np.sum(p * np.log(p)))

def _level_id_from_token_role(token_role: str, level_names: List[str], token_role_to_level: Optional[Dict[str, str]] = None) -> int:
    """Map token role to level using learned mapping or pattern-based fallback."""
    if token_role_to_level:
        level = token_role_to_level.get(token_role, "content")
    else:
        # Pattern-based level inference (no hardcoded role lists)
        if token_role == "punct":
            level = "punct"
        elif token_role and token_role.startswith("meta"):
            level = "meta"
        elif token_role and ("function" in token_role or "adverb" in token_role):
            level = "function"
        else:
            level = "content"
    return level_names.index(level) if level in level_names else 0

def _init_role_state(prompt_ids: List[int], roles_by_id: List[str], end_token_ids: List[int],
                    role_names: List[str], trans: Optional[np.ndarray] = None) -> np.ndarray:
    R, p = len(role_names), np.ones((len(role_names),), dtype=np.float64) / max(1, len(role_names))
    if prompt_ids and trans is not None and trans.ndim == 2 and trans.shape[0] == R and trans.shape[1] == R:
        prev_role_probs = np.ones((R,), dtype=np.float64) / R
        next_role_probs = (prev_role_probs @ trans.astype(np.float64))
        next_role_probs = next_role_probs / np.sum(np.maximum(next_role_probs, 1e-12))
        p = 0.15 * p + 0.85 * next_role_probs
    return (p / np.sum(p)).astype(np.float32)

def _role_compat_log(role_state: np.ndarray, emit_logp: np.ndarray, level_id: int) -> float:
    return float(np.log(max(1e-12, float(np.dot(role_state.astype(np.float64), np.exp(emit_logp[:, level_id]).astype(np.float64))))))

def _get_position_bucket(tokens_since_end: int, pos_thresholds: List[int], max_sentence_tokens: int) -> int:
    if pos_thresholds and len(pos_thresholds) >= 2:
        return 0 if tokens_since_end <= pos_thresholds[0] else (1 if tokens_since_end <= pos_thresholds[1] else 2)
    t1, t2 = max(1, int(max_sentence_tokens * 0.33)), max(max(1, int(max_sentence_tokens * 0.33)) + 1, int(max_sentence_tokens * 0.66))
    return 0 if tokens_since_end <= t1 else (1 if tokens_since_end <= t2 else 2)
def _closure_boost(candidate_id: int, tokens_since_end: int, min_tokens: int, max_tokens: int,
                  end_token_ids: List[int], base_boost: float, roles_by_id: Optional[List[str]] = None) -> float:
    if base_boost <= 0 or not end_token_ids or candidate_id not in end_token_ids or (roles_by_id and roles_by_id[candidate_id] != "punct"):
        return 0.0
    if tokens_since_end < max(0, int(min_tokens)):
        return 0.0
    if max_tokens > 0 and tokens_since_end >= max_tokens:
        return float(base_boost * 2.0)
    if max_tokens > min_tokens and max_tokens > 0:
        t = float(np.clip((tokens_since_end - min_tokens) / max(1e-6, max_tokens - min_tokens), 0.0, 1.0))
        return float(base_boost * (0.4 + 0.6 * t))
    return float(base_boost)

def generate(
    art: ReasonerArtifacts,
    prompt: str,
    max_new_tokens: int = 30,
    temperature: float = 0.8,
    top_k: int = 8,
    explain: bool = False,
    block_dataset_meta: bool = True,
    repeat_window: int = 12,
    repetition_penalty: float = 2.25,
    semantic_repeat_window: int = 6,
    semantic_repeat_threshold: float = 0.68,
    semantic_repeat_penalty: float = 1.15,
    # New: style + sentence closure + cluster switching (no hardcoded tokens)
    style: str = "descriptive",  # descriptive|explanatory
    min_sentence_tokens: int = 12,
    max_sentence_tokens: int = 35,
    closure_strength: float = 1.15,
    cluster_switch_window: int = 2,
    cluster_switch_penalty: float = 0.28,
    # New: role-aware neighbor gating (no hardcoded token lists)
    meta_frame_penalty: float = 1.05,
    role_loop_penalty: float = 0.85,
    # Configurable weights and thresholds (previously hardcoded)
    weight_bigram: float = 1.35,
    weight_semantic: float = 1.15,
    weight_contradiction: float = 1.45,
    context_window: int = 30,  # Increased from 20 for better coherence
    contradiction_window: int = 10,
    role_weight: float = 0.45,
    role_weight_explanatory_mult: float = 1.10,
    role_weight_descriptive_mult: float = 0.95,
    style_closure_mult_descriptive: float = 0.6,
    style_cluster_mult_descriptive: float = 0.35,
    position_bucket_early_ratio: float = 0.33,
    position_bucket_mid_ratio: float = 0.66,
    entropy_threshold_low: float = 0.55,
    entropy_threshold_role_state: float = 0.85,
    role_state_threshold: float = 0.60,
    loop_boost_base: float = 1.35,
    loop_boost_meta: float = 1.25,
    role_inertia_keep: float = 0.70,
    role_inertia_new: float = 0.30,
    closure_max_boost_multiplier: float = 2.0,
    closure_ramp_start: float = 0.4,
    closure_ramp_end: float = 0.6,
    role_init_uniform_weight: float = 0.15,
    role_init_target_weight: float = 0.85,
) -> Dict[str, Any]:
    tok = ClosedVocabTokenizer(vocab=art.vocab, token_to_id=art.token_to_id)
    ids = tok.encode(prompt)
    V = len(art.vocab)
    vec = art.vectors
    bigram_logp = art.bigram_logp

    contra = _build_contradiction_lookup(art.contradiction_pairs, art.token_to_id)

    # role lookup arrays for speed
    roles_by_id = [art.roles.get(t, "unknown") for t in art.vocab]

    # optional cluster ids array
    clusters_by_id: Optional[List[int]] = None
    meta_cluster_centroid = None
    if art.clusters:
        clusters_by_id = [int(art.clusters.get(t, -1)) for t in art.vocab]
        if all(c == -1 for c in clusters_by_id):
            clusters_by_id = None
        # Build meta cluster centroid for dynamic meta-text detection
        meta_cluster_ids = getattr(art, "meta_cluster_ids", []) or []
        if meta_cluster_ids and clusters_by_id:
            meta_tokens = [i for i, cid in enumerate(clusters_by_id) if cid in meta_cluster_ids]
            if len(meta_tokens) >= 3:
                meta_cluster_centroid = vec[meta_tokens].mean(axis=0)
                meta_norm = np.linalg.norm(meta_cluster_centroid)
                if meta_norm > 1e-8:
                    meta_cluster_centroid = meta_cluster_centroid / meta_norm
                else:
                    meta_cluster_centroid = None

    # inferred sentence-end tokens (punctuation-like)
    end_token_ids: List[int] = []
    for t in getattr(art, "sentence_end_tokens", []) or []:
        if t in art.token_to_id:
            end_token_ids.append(int(art.token_to_id[t]))

    

    # ---- Discourse role state (optional) ----
    role_names = getattr(art, "discourse_role_names", []) or []
    emit_logp: Optional[np.ndarray] = None
    trans: Optional[np.ndarray] = None
    emit_logp_pos: Optional[np.ndarray] = None
    trans_pos: Optional[np.ndarray] = None
    pos_thresholds: List[int] = list(getattr(art, "discourse_pos_thresholds", []) or [])
    role_min_run: List[int] = list(getattr(art, "discourse_role_min_run", []) or [])
    role_state: Optional[np.ndarray] = None
    committed_role: Optional[int] = None
    committed_age: int = 0

    if role_names:
        try:
            base_emit = getattr(art, "discourse_emit_logp", None)
            base_trans = getattr(art, "discourse_trans", None)
            if base_emit is not None:
                emit_logp = np.array(base_emit, dtype=np.float32)
            if base_trans is not None:
                trans = np.array(base_trans, dtype=np.float32)

            pos_emit = getattr(art, "discourse_emit_logp_pos", None)
            pos_trans = getattr(art, "discourse_trans_pos", None)
            if pos_emit is not None:
                emit_logp_pos = np.array(pos_emit, dtype=np.float32)
            if pos_trans is not None:
                trans_pos = np.array(pos_trans, dtype=np.float32)

            # validate position-conditioned shapes
            ok_pos = False
            if emit_logp_pos is not None and trans_pos is not None:
                if emit_logp_pos.ndim == 3 and trans_pos.ndim == 3:
                    if emit_logp_pos.shape[0] == trans_pos.shape[0] and emit_logp_pos.shape[1] == trans_pos.shape[1] == trans_pos.shape[2]:
                        ok_pos = True
            if not ok_pos:
                emit_logp_pos = None
                trans_pos = None

            if (emit_logp is not None and trans is not None and emit_logp.ndim == 2 and trans.ndim == 2 and emit_logp.shape[0] == trans.shape[0]) or ok_pos:
                role_state = _init_role_state(ids, roles_by_id, end_token_ids, role_names, trans)
                committed_role = int(np.argmax(role_state)) if role_state is not None else None
                committed_age = 1 if committed_role is not None else 0
        except Exception:
            emit_logp = None
            trans = None
            emit_logp_pos = None
            trans_pos = None
            role_state = None
            committed_role = None
            committed_age = 0

    st = (style or "descriptive").strip().lower()
    if st not in {"descriptive", "explanatory"}:
        st = "descriptive"
    if st == "descriptive":
        # lighter structure constraints
        closure_k = float(closure_strength) * style_closure_mult_descriptive
        cluster_pen = float(cluster_switch_penalty) * style_cluster_mult_descriptive
    else:
        closure_k = float(closure_strength)
        cluster_pen = float(cluster_switch_penalty)

    log = []
    tokens_since_end = 0

    for step in range(max_new_tokens):
        prev = ids[-1]

        # update tokens_since_end based on last emitted token
        last_role = roles_by_id[prev] if prev < len(roles_by_id) else "unknown"
        if last_role == "punct" and prev in end_token_ids:
            tokens_since_end = 0
        else:
            tokens_since_end += 1

        # context vector = weighted mean of last tokens (recent tokens weighted more)
        # Improved: use positional encoding awareness for better context understanding
        k = min(context_window, len(ids))
        recent_ids = ids[-k:]
        # Exponential decay weighting: more recent tokens have higher weight
        # Also add positional bias: tokens closer to current position matter more
        if len(recent_ids) > 1:
            # Decay factor: recent tokens get exponentially more weight
            decay_factor = 0.85
            weights = np.array([decay_factor ** (len(recent_ids) - 1 - i) for i in range(len(recent_ids))], dtype=np.float32)
            weights = weights / weights.sum()
            ctx = (vec[recent_ids] * weights[:, np.newaxis]).sum(axis=0)
        else:
            ctx = vec[recent_ids].mean(axis=0) if recent_ids else np.zeros(vec.shape[1], dtype=np.float32)

        comp_bigram = bigram_logp[prev].astype(np.float64)
        # Optimized semantic similarity: vectorized dot product with normalization
        # Normalize context vector for better cosine similarity (with numerical stability)
        ctx_norm_val = np.linalg.norm(ctx)
        if ctx_norm_val > 1e-8:
            ctx_norm = ctx / ctx_norm_val
        else:
            ctx_norm = np.zeros_like(ctx)
        
        vec_norms = np.linalg.norm(vec, axis=1, keepdims=True)
        vec_norm = np.where(vec_norms > 1e-8, vec / vec_norms, np.zeros_like(vec))
        
        # Safe matmul with numerical stability checks
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            comp_sem = (vec_norm @ ctx_norm).astype(np.float64)
            comp_sem = np.nan_to_num(comp_sem, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Apply adaptive scaling: boost semantic component when it's informative
        sem_std = float(np.std(comp_sem))
        if sem_std > 0.15:  # Good semantic diversity
            comp_sem = comp_sem * 1.15
        else:  # Low diversity, boost more
            comp_sem = comp_sem * 1.25
        
        # Combine bigram and semantic scores
        logits = (weight_bigram * comp_bigram + weight_semantic * comp_sem).astype(np.float64)
        
        # Topic drift detection: check alignment with prompt anchor (if available)
        # Build anchor from prompt tokens
        if len(ids) > 0:
            # Use initial prompt tokens as anchor
            k0 = min(24, len(ids))
            anchor_tokens = ids[:k0]
            if len(anchor_tokens) > 0:
                if len(anchor_tokens) > 1:
                    weights = np.exp(np.linspace(0, -0.25, len(anchor_tokens), dtype=np.float32))
                    weights = weights / (weights.sum() + 1e-12)
                    anchor = (vec[anchor_tokens] * weights[:, np.newaxis]).sum(axis=0)
                else:
                    anchor = vec[anchor_tokens[0]]
                anchor_norm_val = np.linalg.norm(anchor)
                if anchor_norm_val > 1e-8:
                    anchor_norm = anchor / anchor_norm_val
                    # Safe topic alignment calculation
                    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                        topic_alignment = (vec_norm @ anchor_norm).astype(np.float64)
                        topic_alignment = np.nan_to_num(topic_alignment, nan=0.0, posinf=1.0, neginf=-1.0)
                    # Penalty for tokens that drift from topic (strengthened)
                    topic_drift_penalty = np.zeros((V,), dtype=np.float64)
                    for j in range(V):
                        if topic_alignment[j] < 0.15:  # Low alignment with topic
                            topic_drift_penalty[j] = 0.5 * (0.15 - topic_alignment[j])  # Increased from 0.25
                    logits = logits - topic_drift_penalty

        # contradiction penalty based on recent tokens - OPTIMIZED: sparse lookup
        recent_contra = ids[-min(contradiction_window, len(ids)) :]
        contra_pen = np.zeros((V,), dtype=np.float64)
        # Optimized: only iterate over tokens that have contradictions
        recent_set = set(recent_contra)
        # Build reverse lookup: for each recent token, get all its contradictions
        for r in recent_set:
            # Only check tokens that actually contradict with r
            for j in range(V):
                pen = contra.get((r, j), 0.0)
                if pen > 0:
                    contra_pen[j] += pen * (1.0 + 0.1 * len(recent_set))  # Scale by context size

        # Enhanced: better balance with stronger bigram component
        # Bigram is critical for fluency, so give it more weight
        logits = weight_bigram * comp_bigram + weight_semantic * comp_sem - weight_contradiction * contra_pen
        
        # Additional coherence check: boost tokens that follow well from previous
        if len(ids) >= 2:
            prev_prev = ids[-2]
            if 0 <= prev_prev < V:
                # Check bigram probability for prev_prev -> candidate
                for j in range(V):
                    if prev_prev < len(bigram_logp) and j < len(bigram_logp[prev_prev]):
                        bigram_score = float(bigram_logp[prev_prev, j])
                        if bigram_score > -2.0:  # Good transition
                            logits[j] += 0.15
                        elif bigram_score < -6.0:  # Poor transition
                            logits[j] -= 0.10

                # --- Discourse role compatibility (LLM-like token-by-token role_state) ---
        # If discourse artifacts exist, adjust logits by how compatible a candidate token's LEVEL is
        # with the current inferred discourse role_state. If position-conditioned artifacts exist,
        # select the appropriate bucket (early/mid/late within the current sentence).
        emit_use = emit_logp
        if role_state is not None:
            b = _get_position_bucket(tokens_since_end, pos_thresholds, max_sentence_tokens)
            if emit_logp_pos is not None and emit_logp_pos.ndim == 3 and 0 <= b < emit_logp_pos.shape[0]:
                emit_use = emit_logp_pos[b]

        if role_state is not None and emit_use is not None:
            # small style-dependent scaling
            style_mult = role_weight_explanatory_mult if st == "explanatory" else role_weight_descriptive_mult
            rw = float(role_weight) * style_mult
            level_names = getattr(art, "discourse_level_names", []) or []
            # Use learned token_role_to_level mapping from art
            token_role_to_level: Optional[Dict[str, str]] = getattr(art, "token_role_to_level", None)
            for j in range(V):
                lvl = _level_id_from_token_role(roles_by_id[j], level_names, token_role_to_level)
                logits[j] += rw * _role_compat_log(role_state, emit_use, lvl)

# --- Output gating: dataset/meta tokens (derived during training, no hardcoded token lists) ---
        if block_dataset_meta:
            for j in range(V):
                if roles_by_id[j] == "dataset_meta":
                    logits[j] = -1e18
        
        # Dynamic meta-text detection: penalize tokens that form meta-text patterns
        # Check if candidate token would create meta-text sequences
        if meta_cluster_centroid is not None and len(ids) >= 1:
            for j in range(V):
                # Check semantic similarity to meta cluster
                cand_vec = vec[j]
                cand_norm_val = np.linalg.norm(cand_vec)
                if cand_norm_val > 1e-8:
                    cand_norm = cand_vec / cand_norm_val
                    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                        meta_sim = float(np.dot(cand_norm, meta_cluster_centroid))
                        meta_sim = np.nan_to_num(meta_sim, nan=0.0, posinf=1.0, neginf=-1.0)
                    # If token is semantically similar to meta cluster, check context
                    if meta_sim > 0.35:  # Lowered threshold from 0.4
                        # Check if recent tokens also have meta roles (forming meta-text phrase)
                        recent_meta_count = 0
                        for rid in ids[-3:]:
                            if 0 <= rid < V:
                                rrole = roles_by_id[rid]
                                if rrole and (rrole.startswith("meta") or rrole == "dataset_meta"):
                                    recent_meta_count += 1
                        # If recent tokens are also meta, this forms a meta-text phrase
                        if recent_meta_count >= 1:
                            logits[j] -= 1.5 * meta_sim  # Increased from 0.8 - stronger penalty
                        # Also penalize if token itself has meta role
                        if roles_by_id[j] and (roles_by_id[j].startswith("meta") or roles_by_id[j] == "dataset_meta"):
                            logits[j] -= 0.6 * meta_sim  # Additional penalty for meta role + semantic similarity

        
        # --- Role-aware neighbor gating: penalize editorial/meta framing loops (no hardcoded token lists) ---
        # Tokens labeled as 'meta_frame' are derived statistically during training. In explanatory mode,
        # we discourage these tokens unless the distribution strongly supports them.
        if style == "explanatory":
            # detect low-entropy role loops in the recent context
            recent_roles = []
            for rid in ids[-6:]:
                if 0 <= rid < V:
                    recent_roles.append(roles_by_id[rid])
            # compute simple role entropy (over roles, excluding punct)
            rr = [r for r in recent_roles if r != "punct"]
            entropy = 0.0
            if rr:
                uniq = sorted(set(rr))
                p = np.array([rr.count(u) / len(rr) for u in uniq], dtype=np.float64)
                entropy = float(-(p * np.log(np.maximum(p, 1e-12))).sum())
            # if we are stuck in low-entropy loops, apply stronger penalty
            loop_boost = 1.0
            if rr and entropy < entropy_threshold_low:
                loop_boost = loop_boost_base
            # additional boost: low-entropy high-probability role discourse in explanatory mode
            # Use role_state probabilities directly (no hardcoded role name lookups)
            if role_state is not None and emit_logp is not None and st == "explanatory" and role_names:
                # Find role with highest probability (no hardcoded "meta" name)
                max_role_idx = int(np.argmax(role_state))
                if float(role_state[max_role_idx]) > role_state_threshold and _entropy(role_state) < entropy_threshold_role_state:
                    loop_boost *= loop_boost_meta

            for j in range(V):
                if roles_by_id[j] == "meta_frame":
                    logits[j] -= float(meta_frame_penalty) * loop_boost

                # discourage sequences that remain in meta_frame role repeatedly
                if rr and rr[-1] == "meta_frame" and roles_by_id[j] == "meta_frame":
                    logits[j] -= float(role_loop_penalty) * loop_boost

# --- Anti-repetition controls (exact + semantic + n-gram) - ENHANCED ---
        rw = max(0, int(repeat_window))
        if rw > 0 and repetition_penalty > 0:
            exact_recent = ids[-rw:]
            # Enhanced: count frequency and apply MUCH stronger penalty for frequent repeats
            recent_counts = {}
            for r in exact_recent:
                recent_counts[r] = recent_counts.get(r, 0) + 1
            for j in range(V):
                count = recent_counts.get(j, 0)
                if count > 0:
                    # Very strong exponential penalty: more repeats = MUCH stronger penalty
                    # Base penalty scales with count, then exponential multiplier
                    base_penalty = float(repetition_penalty) * count
                    exponential_mult = 1.0 + (count - 1) * 1.2  # Exponential growth
                    logits[j] -= base_penalty * exponential_mult
                    # Additional penalty if token appeared very recently (last 3 positions)
                    if len(ids) >= 3 and j in ids[-3:]:
                        logits[j] -= float(repetition_penalty) * 0.8
            
            # N-gram repetition detection (phrases/bigrams)
            if len(ids) >= 2:
                # Check for bigram repetition
                recent_bigrams = {}
                for i in range(max(0, len(ids) - rw), len(ids) - 1):
                    bigram = (ids[i], ids[i+1])
                    recent_bigrams[bigram] = recent_bigrams.get(bigram, 0) + 1
                # Penalize tokens that would create repeated bigrams
                if len(ids) >= 1:
                    prev_token = ids[-1]
                    for j in range(V):
                        bigram = (prev_token, j)
                        bigram_count = recent_bigrams.get(bigram, 0)
                        if bigram_count > 0:
                            # Penalty for repeating bigrams
                            logits[j] -= float(repetition_penalty) * 0.6 * bigram_count
            
            # Trigram repetition detection
            if len(ids) >= 3:
                recent_trigrams = {}
                for i in range(max(0, len(ids) - rw), len(ids) - 2):
                    trigram = (ids[i], ids[i+1], ids[i+2])
                    recent_trigrams[trigram] = recent_trigrams.get(trigram, 0) + 1
                # Penalize tokens that would create repeated trigrams
                if len(ids) >= 2:
                    prev2_token = ids[-2]
                    prev1_token = ids[-1]
                    for j in range(V):
                        trigram = (prev2_token, prev1_token, j)
                        trigram_count = recent_trigrams.get(trigram, 0)
                        if trigram_count > 0:
                            # Strong penalty for repeating trigrams (phrases)
                            logits[j] -= float(repetition_penalty) * 1.2 * trigram_count

        sw = max(0, int(semantic_repeat_window))
        if sw > 0 and semantic_repeat_penalty > 0:
            sem_recent = ids[-sw:]
            # Enhanced: check semantic similarity more aggressively
            for j in range(V):
                if roles_by_id[j] == "punct":
                    continue
                # Check against all recent tokens, not just one
                max_sim = 0.0
                for r in sem_recent:
                    sim = float(vec[r] @ vec[j])
                    max_sim = max(max_sim, sim)
                if max_sim > float(semantic_repeat_threshold):
                    # Stronger penalty for high similarity
                    penalty = float(semantic_repeat_penalty) * (1.0 + 2.0 * (max_sim - float(semantic_repeat_threshold)))
                    logits[j] -= penalty

        # --- Cluster switch (concept packing) ---
        if clusters_by_id is not None and cluster_pen > 0:
            cw = max(0, int(cluster_switch_window))
            if cw > 0:
                for j in range(V):
                    logits[j] -= _cluster_switch_penalty(j, ids, clusters_by_id, cw, cluster_pen, roles_by_id=roles_by_id)

        # Enhanced sentence length control
        if closure_k > 0 and end_token_ids:
            for j in end_token_ids:
                closure_boost_val = _closure_boost(j, tokens_since_end, int(min_sentence_tokens), int(max_sentence_tokens), end_token_ids, closure_k, roles_by_id)
                logits[j] += closure_boost_val
                # Penalty for ending sentences too early (before min_sentence_tokens)
                if tokens_since_end < int(min_sentence_tokens):
                    logits[j] -= 1.5 * closure_k  # Strong penalty for premature sentence end
                # Bonus for continuing longer sentences (encourage longer sentences)
                elif tokens_since_end >= int(min_sentence_tokens) and tokens_since_end < int(max_sentence_tokens):
                    # Encourage continuation: penalize sentence end tokens when sentence is still growing
                    logits[j] -= 0.4 * closure_k
        
        # Sentence completeness check: penalize incomplete sentences
        # Check if we're ending a sentence that seems incomplete
        if len(ids) >= 3:
            # Check if last few tokens suggest incomplete sentence
            last_tokens = ids[-3:]
            last_roles = [roles_by_id[t] if 0 <= t < V else "unknown" for t in last_tokens]
            # If ending with function words or incomplete structure, penalize sentence end
            if any(r == "function_word" for r in last_roles[-2:]):
                for j in end_token_ids:
                    logits[j] -= 0.3 * closure_k

        # Improved sampling: nucleus (top-p) + top-k hybrid for better quality
        # Sort all logits
        sorted_idx = np.argsort(-logits)
        sorted_logits = logits[sorted_idx]
        
        # Apply temperature first
        temp_adj = max(float(temperature), 0.1)
        temp_logits = sorted_logits / temp_adj
        
        # Compute probabilities for nucleus sampling
        temp_logits = temp_logits - np.max(temp_logits)  # Numerical stability
        exp_logits = np.exp(np.clip(temp_logits, -50, 50))
        probs = exp_logits / exp_logits.sum()
        
        # Nucleus (top-p) sampling: take tokens until cumulative prob >= 0.9
        cumsum_probs = np.cumsum(probs)
        nucleus_idx = np.searchsorted(cumsum_probs, 0.9, side='right') + 1
        nucleus_idx = min(nucleus_idx, len(probs), max(1, top_k))
        
        # Also respect top-k limit
        nucleus_idx = min(nucleus_idx, max(1, top_k))
        
        # Select from nucleus
        top_idx = sorted_idx[:nucleus_idx]
        top_probs = probs[:nucleus_idx]
        top_probs = top_probs / top_probs.sum()  # Renormalize
        
        choice_local = int(np.random.choice(len(top_idx), p=top_probs))
        nxt = int(top_idx[choice_local])
        ids.append(nxt)

        if role_state is not None:
            b = _get_position_bucket(tokens_since_end, pos_thresholds, max_sentence_tokens)
            trans_use = trans_pos[b] if trans_pos is not None and trans_pos.ndim == 3 and 0 <= b < trans_pos.shape[0] else trans
            emit_use = emit_logp_pos[b] if emit_logp_pos is not None and emit_logp_pos.ndim == 3 and 0 <= b < emit_logp_pos.shape[0] else emit_logp
            if emit_use is not None and trans_use is not None:
                level_names = getattr(art, "discourse_level_names", []) or []
                token_role_to_level = getattr(art, "token_role_to_level", None)
                lvl = _level_id_from_token_role(roles_by_id[nxt], level_names, token_role_to_level)
                rs = (role_state.astype(np.float64) @ np.asarray(trans_use, dtype=np.float64)) * np.exp(np.asarray(emit_use, dtype=np.float64)[:, lvl])
                role_state = (rs / max(1e-12, float(np.sum(rs)))).astype(np.float32) if np.sum(rs) > 1e-12 else (np.ones_like(role_state) / max(1, role_state.shape[0])).astype(np.float32)
                if role_min_run and committed_role is not None:
                    prev_dom, new_dom = int(committed_role), int(np.argmax(role_state))
                    mr = int(role_min_run[prev_dom]) if prev_dom < len(role_min_run) else 2
                    if committed_age < mr and new_dom != prev_dom:
                        one = np.zeros_like(role_state, dtype=np.float32)
                        one[prev_dom] = 1.0
                        role_state = ((0.70 * role_state + 0.30 * one) / max(1e-12, float(np.sum(0.70 * role_state + 0.30 * one)))).astype(np.float32)
                        committed_age += 1
                    else:
                        committed_role, committed_age = (new_dom, 1) if new_dom != prev_dom else (committed_role, committed_age + 1)
                else:
                    committed_role, committed_age = int(np.argmax(role_state)), 1

        if explain:
            kshow = min(5, len(top_idx))
            cand = []
            for j in top_idx[:kshow]:
                cand.append(
                    {
                        "token": art.vocab[int(j)],
                        "score": float(logits[int(j)]),
                        "bigram_logp": float(comp_bigram[int(j)]),
                        "semantic_sim": float(comp_sem[int(j)]),
                        "contradiction_penalty": float(contra_pen[int(j)]),
                        "role": roles_by_id[int(j)],
                        "cluster": int(clusters_by_id[int(j)]) if clusters_by_id is not None else -1,
                    }
                )
            log.append(
                {
                    "step": step + 1,
                    "prev_token": art.vocab[int(prev)],
                    "chosen_token": art.vocab[int(nxt)],
                    "top_candidates": cand,
                }
            )

    return {"text": tok.decode(ids), "token_ids": ids, "explain": log}

# ---------------- Dual-model generation (Reasoner + Selector debate) ----------------
def _generate_dual_once(
    reasoner: ReasonerArtifacts,
    selector: SelectorArtifacts,
    prompt: str,
    initial_ids: Optional[List[int]] = None,
    return_ids: bool = False,
    max_new_tokens: int = 60,
    temperature: float = 0.85,
    selector_top_k: int = 32,
    # Debate mix weights: favor reasoner slightly more for better coherence
    alpha_selector: float = 0.45,
    beta_reasoner: float = 0.55,
    # Online strengthening/weakening during generation:
    eta_trust: float = 0.10,
    eta_bias: float = 0.08,
    trust_clip: float = 2.0,
    # Reasoner scoring weights:
    w_semantic: float = 1.25,
    w_anchor: float = 1.15,  # Increased from 0.85 to strengthen prompt relevance
    w_contra: float = 1.45,
    w_repeat: float = 1.35,
    repeat_window: int = 12,
    block_dataset_meta: bool = True,
    explain: bool = False,
    context_window: int = 30,  # Increased from 20 for better coherence
    # NEW: micro-continuation debate (sequence-level critique)
    lookahead_len: int = 10,
    num_continuations: int = 18,
    commit_len: int = 1,
    # NEW: penalize "LLM-like" meta framing (derived token roles, no hardcoded token lists)
    w_meta_frame: float = 0.45,
) -> Dict[str, Any]:
    """Generate text via a micro-continuation 'debate' between two models.

    Previous version debated the *next token* only. This patch debates *short continuations* (3-8 tokens),
    scores them at the sequence level, and then commits the first token (or a small number of tokens).

    Rationale: the Reasoner is strongest at recognizing global/structural patterns. Scoring only a single
    token makes it difficult to suppress "LLM-corpus style" continuations. Sequence-level critique lets the
    Reasoner veto entire continuation patterns while the Selector still provides fluency.

    - Selector: proposes short continuations via sparse trigram/bigram LM.
    - Reasoner: scores each continuation for semantic stability, anchor alignment, contradiction avoidance,
      repetition avoidance, and (optionally) meta-framing suppression.
    - Online adaptation: trust[token] and per-context selector bias are updated during generation, based on
      the chosen continuation's advantage relative to alternatives.
    """
    tok = ClosedVocabTokenizer(vocab=reasoner.vocab, token_to_id=reasoner.token_to_id)
    prompt_ids: List[int] = tok.encode(prompt)
    ids: List[int] = list(initial_ids) if initial_ids is not None else list(prompt_ids)
    V = len(reasoner.vocab)
    vec = reasoner.vectors

    # prompt anchor (directional intent) - enhanced with better weighting and topic tracking
    if prompt_ids:
        k0 = min(24, len(prompt_ids))  # Use more tokens for better anchor
        # Weight earlier tokens more (they define the topic)
        anchor_tokens = prompt_ids[:k0]
        if len(anchor_tokens) > 1:
            weights = np.exp(np.linspace(0, -0.25, len(anchor_tokens), dtype=np.float32))
            weights = weights / weights.sum()
            anchor = (vec[anchor_tokens] * weights[:, np.newaxis]).sum(axis=0)
        else:
            anchor = vec[anchor_tokens].mean(axis=0)
        # Normalize anchor
        anchor = anchor / (np.linalg.norm(anchor) + 1e-8)
    else:
        anchor = np.zeros((vec.shape[1],), dtype=np.float32)
    
    # Topic tracking: maintain running average of recent tokens to detect topic drift
    # Strengthened: higher decay to preserve original topic better
    topic_tracker = anchor.copy() if len(prompt_ids) > 0 else np.zeros((vec.shape[1],), dtype=np.float32)
    topic_tracker_decay = 0.95  # Increased from 0.92 to preserve topic better
    # Initialize topic tracker with recent tokens from ids if available
    if len(ids) > 0:
        recent_for_topic = ids[-min(8, len(ids)):]
        if recent_for_topic:
            topic_tracker = vec[recent_for_topic].mean(axis=0)
            topic_norm = np.linalg.norm(topic_tracker)
            if topic_norm > 1e-8:
                topic_tracker = topic_tracker / topic_norm
            else:
                topic_tracker = anchor.copy() if len(prompt_ids) > 0 else np.zeros((vec.shape[1],), dtype=np.float32)

    contra = _build_contradiction_lookup(reasoner.contradiction_pairs, reasoner.token_to_id)
    contra_lookup = build_contra_lookup(reasoner)
    roles_by_id = [reasoner.roles.get(t, "unknown") for t in reasoner.vocab]
    
    # Build meta cluster centroid for dynamic meta-text detection (no hardcoded lists)
    meta_cluster_centroid = None
    meta_cluster_ids = getattr(reasoner, "meta_cluster_ids", []) or []
    if meta_cluster_ids and reasoner.clusters:
        clusters_by_id = [int(reasoner.clusters.get(t, -1)) for t in reasoner.vocab]
        meta_tokens = [i for i, cid in enumerate(clusters_by_id) if cid in meta_cluster_ids]
        if len(meta_tokens) >= 3:
            meta_cluster_centroid = vec[meta_tokens].mean(axis=0)
            meta_norm = np.linalg.norm(meta_cluster_centroid)
            if meta_norm > 1e-8:
                meta_cluster_centroid = meta_cluster_centroid / meta_norm
            else:
                meta_cluster_centroid = None

    # Online state
    trust = np.zeros((V,), dtype=np.float64)
    # sparse bias: ctx_key -> {token_id: bias}
    bias: Dict[str, Dict[int, float]] = {}

    explain_rows: List[Dict[str, Any]] = []

    def ctx_key(p2: Optional[int], p1: Optional[int]) -> str:
        if p2 is not None and p1 is not None:
            return f"{int(p2)},{int(p1)}"
        if p1 is not None:
            return str(int(p1))
        return "<start>"

    def get_bias(k: str, tid: int) -> float:
        d = bias.get(k)
        if not d:
            return 0.0
        return float(d.get(int(tid), 0.0))

    def add_bias(k: str, tid: int, delta: float) -> None:
        d = bias.get(k)
        if d is None:
            d = {}
            bias[k] = d
        d[int(tid)] = float(d.get(int(tid), 0.0) + float(delta))

    def contradiction_penalty(candidate: int, recent: List[int]) -> float:
        pen = 0.0
        for r in recent:
            pen = max(pen, float(contra.get((int(candidate), int(r)), 0.0)))
        return pen

    def sample_next(prev2: Optional[int], prev1: Optional[int]) -> Optional[Tuple[int, float]]:
        cands = selector.candidates(prev2, prev1, top_k=int(selector_top_k))
        if not cands:
            return None
        cand_ids = np.array([int(c) for c, _ in cands], dtype=np.int64)
        cand_lps = np.array([float(lp) for _, lp in cands], dtype=np.float64)
        probs = softmax(cand_lps, temp=max(1e-6, float(temperature)))
        j = int(np.random.choice(len(cand_ids), p=probs))
        return int(cand_ids[j]), float(cand_lps[j])

    def score_continuation(cont: List[int], prev2: Optional[int], prev1: Optional[int]) -> Tuple[float, float, float]:
        """Return (selector_bias_avg, reasoner_avg, meta_frac).

        Note: this function does NOT include the selector's sampled logp (handled by the caller).
        Quality scoring is handled separately in the caller.
        """
        if not cont:
            return -1e9, -1e9, 0.0

        # selector: include online bias along the path
        sel_sum = 0.0
        p2, p1 = prev2, prev1
        for tid in cont:
            kctx = ctx_key(p2, p1)
            # we don't have per-step logp stored for every token here (only sampled token's lp when building cont)
            # so sel_sum is accumulated when building continuations; bias is added here to reflect online shaping
            sel_sum += get_bias(kctx, int(tid))
            p2, p1 = (p1, int(tid)) if p1 is not None else (None, int(tid))

        # reasoner: score sequence token-by-token with moving context
        # Optimized: batch process where possible
        tmp_ids = list(ids)
        reason_sum = 0.0
        meta_count = 0.0
        
        # Initialize local topic tracker for this continuation (use anchor as base)
        local_topic_tracker = anchor.copy() if len(prompt_ids) > 0 else np.zeros((vec.shape[1],), dtype=np.float32)
        if len(tmp_ids) > 0:
            recent_for_topic = tmp_ids[-min(6, len(tmp_ids)):]
            if recent_for_topic:
                local_topic_tracker = vec[recent_for_topic].mean(axis=0)
                local_topic_norm = np.linalg.norm(local_topic_tracker)
                if local_topic_norm > 1e-8:
                    local_topic_tracker = local_topic_tracker / local_topic_norm
                else:
                    local_topic_tracker = anchor.copy() if len(prompt_ids) > 0 else np.zeros((vec.shape[1],), dtype=np.float32)
        topic_tracker_decay_local = 0.95  # Increased from 0.92 to preserve topic better
        
        # Prompt relevance: compute overall continuation relevance to prompt
        cont_vec_sum = np.zeros((vec.shape[1],), dtype=np.float32)
        cont_token_count = 0
        
        # Dynamic meta-text detection: track meta patterns in continuation
        meta_sequence_penalty = 0.0
        meta_role_sequence = []
        
        for idx, tid in enumerate(cont):
            if not (0 <= int(tid) < V):
                continue
            role = roles_by_id[int(tid)]
            # block dataset/meta tokens completely if requested
            if block_dataset_meta and role == "dataset_meta":
                reason_sum -= 3.5  # Increased from 2.5
            # Count meta roles using pattern matching (no hardcoded set)
            if role and (role.startswith("meta") or role == "dataset_meta"):
                meta_count += 1.0
                # Additional penalty for meta_frame roles (stronger than other meta roles)
                if role == "meta_frame":
                    reason_sum -= 1.5  # Strong penalty for meta_frame tokens
                meta_role_sequence.append(int(tid))
            else:
                meta_role_sequence.append(None)

            k = min(int(context_window), len(tmp_ids)) if tmp_ids else 0
            if k > 1:
                # Weighted context: recent tokens more important with positional decay
                recent = tmp_ids[-k:]
                decay_factor = 0.88
                weights = np.array([decay_factor ** (len(recent) - 1 - i) for i in range(len(recent))], dtype=np.float32)
                weights = weights / weights.sum()
                ctx = (vec[recent] * weights[:, np.newaxis]).sum(axis=0)
            else:
                ctx = vec[tmp_ids[-k:]].mean(axis=0) if k > 0 else anchor

            # Use pre-computed vector for this token with normalization (numerically stable)
            tid_vec = vec[int(tid)]
            # Normalize for better cosine similarity with safety checks
            ctx_norm_val = np.linalg.norm(ctx)
            ctx_norm = ctx / ctx_norm_val if ctx_norm_val > 1e-8 else np.zeros_like(ctx)
            
            anchor_norm_val = np.linalg.norm(anchor)
            anchor_norm = anchor / anchor_norm_val if anchor_norm_val > 1e-8 else np.zeros_like(anchor)
            
            tid_vec_norm_val = np.linalg.norm(tid_vec)
            tid_vec_norm = tid_vec / tid_vec_norm_val if tid_vec_norm_val > 1e-8 else np.zeros_like(tid_vec)
            
            # Safe dot products
            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                sem = float(np.dot(tid_vec_norm, ctx_norm))
                anc = float(np.dot(tid_vec_norm, anchor_norm))
                sem = np.nan_to_num(sem, nan=0.0, posinf=1.0, neginf=-1.0)
                anc = np.nan_to_num(anc, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Topic drift detection: check if token aligns with topic tracker (strengthened)
            topic_tracker_norm_val = np.linalg.norm(local_topic_tracker)
            if topic_tracker_norm_val > 1e-8:
                topic_tracker_norm = local_topic_tracker / topic_tracker_norm_val
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    topic_alignment = float(np.dot(tid_vec_norm, topic_tracker_norm))
                    topic_alignment = np.nan_to_num(topic_alignment, nan=0.0, posinf=1.0, neginf=-1.0)
                # Strengthened penalty for topic drift
                if topic_alignment < 0.2:
                    reason_sum -= 0.5  # Increased from 0.35
                # Bonus for staying on topic
                elif topic_alignment > 0.5:
                    reason_sum += 0.25  # Increased from 0.20
            
            # Accumulate continuation vector for prompt relevance check
            cont_vec_sum += tid_vec
            cont_token_count += 1
            
            # Update local topic tracker (running average) - numerically stable
            local_topic_tracker = topic_tracker_decay_local * local_topic_tracker + (1.0 - topic_tracker_decay_local) * tid_vec
            local_topic_norm = np.linalg.norm(local_topic_tracker)
            if local_topic_norm > 1e-8:
                local_topic_tracker = local_topic_tracker / local_topic_norm
            else:
                local_topic_tracker = anchor.copy() if len(prompt_ids) > 0 else np.zeros((vec.shape[1],), dtype=np.float32)

            recent = tmp_ids[-max(1, int(repeat_window)):] if tmp_ids else []
            # Enhanced repetition check: count frequency with stronger penalty
            rep_count = sum(1 for r in recent if r == int(tid))
            rep = float(rep_count) * (1.0 + 0.8 * rep_count)  # Quadratic scaling
            # Additional penalty if this token appeared in last 2 positions
            if len(tmp_ids) >= 2 and int(tid) in tmp_ids[-2:]:
                rep += 1.5
            
            # N-gram repetition check within continuation
            if len(tmp_ids) >= 1:
                prev_tid = tmp_ids[-1]
                # Check if (prev_tid, tid) bigram appeared recently
                bigram_rep_count = 0
                for i in range(max(0, len(tmp_ids) - int(repeat_window)), len(tmp_ids) - 1):
                    if tmp_ids[i] == prev_tid and i + 1 < len(tmp_ids) and tmp_ids[i+1] == int(tid):
                        bigram_rep_count += 1
                if bigram_rep_count > 0:
                    rep += 0.8 * bigram_rep_count
            
            # Trigram repetition check
            if len(tmp_ids) >= 2:
                prev2_tid = tmp_ids[-2]
                prev1_tid = tmp_ids[-1]
                trigram_rep_count = 0
                for i in range(max(0, len(tmp_ids) - int(repeat_window)), len(tmp_ids) - 2):
                    if (tmp_ids[i] == prev2_tid and 
                        i + 1 < len(tmp_ids) and tmp_ids[i+1] == prev1_tid and
                        i + 2 < len(tmp_ids) and tmp_ids[i+2] == int(tid)):
                        trigram_rep_count += 1
                if trigram_rep_count > 0:
                    rep += 1.2 * trigram_rep_count  # Strong penalty for phrase repetition
            
            cp = contradiction_penalty(int(tid), recent)

            # Enhanced scoring: strong coherence bonus for smooth transitions
            coherence_bonus = 0.0
            coherence_penalty = 0.0
            if idx > 0 and tmp_ids:
                prev_tid = tmp_ids[-1]
                if 0 <= prev_tid < V:
                    # Semantic coherence between consecutive tokens (normalized)
                    prev_vec_norm = vec[prev_tid] / (np.linalg.norm(vec[prev_tid]) + 1e-8)
                    tid_vec_norm = tid_vec / (np.linalg.norm(tid_vec) + 1e-8)
                    coherence = float(np.dot(prev_vec_norm, tid_vec_norm))
                    # Strong bonus for good coherence, penalty for poor coherence
                    if coherence > 0.4:
                        coherence_bonus = 0.35 * (coherence - 0.4)  # Strong reward
                    elif coherence < 0.1:
                        coherence_penalty = 0.25 * (0.1 - coherence)  # Penalty for poor coherence
                    
                    # Also check bigram probability for this transition
                    if prev_tid < len(reasoner.bigram_logp) and int(tid) < len(reasoner.bigram_logp[prev_tid]):
                        bigram_lp = float(reasoner.bigram_logp[prev_tid, int(tid)])
                        if bigram_lp > -3.0:  # Good transition
                            coherence_bonus += 0.20
                        elif bigram_lp < -8.0:  # Poor transition
                            coherence_penalty += 0.15

            # Enhanced: MUCH stronger repetition penalty
            if rep_count > 0:
                rep_penalty = float(w_repeat) * rep * (1.0 + 1.0 * rep_count)  # Stronger scaling
            else:
                rep_penalty = 0.0
            reason_sum += (float(w_semantic) * sem) + (float(w_anchor) * anc) - (float(w_contra) * cp) - rep_penalty + coherence_bonus - coherence_penalty
            tmp_ids.append(int(tid))
            
            # Update local topic tracker for next iteration in this continuation (numerically stable)
            local_topic_tracker = topic_tracker_decay_local * local_topic_tracker + (1.0 - topic_tracker_decay_local) * tid_vec
            local_topic_norm = np.linalg.norm(local_topic_tracker)
            if local_topic_norm > 1e-8:
                local_topic_tracker = local_topic_tracker / local_topic_norm
            else:
                local_topic_tracker = anchor.copy() if len(prompt_ids) > 0 else np.zeros((vec.shape[1],), dtype=np.float32)

        L = max(1, len(cont))
        selector_avg = float(sel_sum) / float(L)
        reasoner_avg = float(reason_sum) / float(L)
        meta_frac = float(meta_count) / float(L)
        
        # Prompt relevance: check if continuation is relevant to prompt
        prompt_relevance_penalty = 0.0
        if cont_token_count > 0 and len(prompt_ids) > 0:
            cont_vec_avg = cont_vec_sum / cont_token_count
            cont_vec_norm_val = np.linalg.norm(cont_vec_avg)
            if cont_vec_norm_val > 1e-8 and anchor_norm_val > 1e-8:
                cont_vec_norm = cont_vec_avg / cont_vec_norm_val
                with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                    prompt_similarity = float(np.dot(cont_vec_norm, anchor_norm))
                    prompt_similarity = np.nan_to_num(prompt_similarity, nan=0.0, posinf=1.0, neginf=-1.0)
                # Penalty if continuation is not relevant to prompt
                if prompt_similarity < 0.2:
                    prompt_relevance_penalty = 0.4 * (0.2 - prompt_similarity)
                # Bonus if continuation is highly relevant
                elif prompt_similarity > 0.5:
                    reasoner_avg += 0.15 * (prompt_similarity - 0.5)
        
        # Apply prompt relevance penalty
        reasoner_avg -= prompt_relevance_penalty
        
        # Dynamic meta-text detection: semantic + pattern-based (no hardcoded lists)
        if cont_token_count > 0 and len(cont) >= 2:
            # Semantic detection: check if continuation vector is similar to meta cluster
            if meta_cluster_centroid is not None:
                cont_vec_avg = cont_vec_sum / cont_token_count
                cont_vec_norm_val = np.linalg.norm(cont_vec_avg)
                if cont_vec_norm_val > 1e-8:
                    cont_vec_norm = cont_vec_avg / cont_vec_norm_val
                    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                        meta_similarity = float(np.dot(cont_vec_norm, meta_cluster_centroid))
                        meta_similarity = np.nan_to_num(meta_similarity, nan=0.0, posinf=1.0, neginf=-1.0)
                    # Stronger penalty if continuation is semantically similar to meta-text
                    if meta_similarity > 0.35:  # Lowered threshold from 0.4
                        meta_sequence_penalty += 1.2 * (meta_similarity - 0.35)  # Increased from 0.6
            
            # Pattern-based detection: check for meta role sequences and bigram/trigram patterns
            # Check for sequences of meta_frame roles (indicating meta-text phrases)
            meta_role_count = sum(1 for r in meta_role_sequence if r is not None)
            if meta_role_count >= 2:
                # Check if meta roles appear in sequence (not scattered)
                consecutive_meta = 0
                max_consecutive = 0
                for r in meta_role_sequence:
                    if r is not None:
                        consecutive_meta += 1
                        max_consecutive = max(max_consecutive, consecutive_meta)
                    else:
                        consecutive_meta = 0
                # Strong penalty for consecutive meta roles (meta-text phrases)
                if max_consecutive >= 2:
                    meta_sequence_penalty += 1.5 * max_consecutive  # Increased from 0.8
                # Also penalize if meta roles are frequent (even if not consecutive)
                if meta_role_count >= len(cont) * 0.3:  # 30% or more meta roles
                    meta_sequence_penalty += 1.0
            
            # Check for bigram/trigram patterns that indicate meta-text
            # Look for patterns like "entry" + "composed", "pretraining" + "prose", etc.
            if len(cont) >= 2:
                for i in range(len(cont) - 1):
                    if 0 <= cont[i] < V and 0 <= cont[i+1] < V:
                        role1 = roles_by_id[cont[i]]
                        role2 = roles_by_id[cont[i+1]]
                        # Check if tokens form meta-text patterns (semantic similarity to meta cluster)
                        tok1_vec = vec[cont[i]]
                        tok2_vec = vec[cont[i+1]]
                        if meta_cluster_centroid is not None:
                            tok1_norm = tok1_vec / (np.linalg.norm(tok1_vec) + 1e-8)
                            tok2_norm = tok2_vec / (np.linalg.norm(tok2_vec) + 1e-8)
                            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                                sim1 = float(np.dot(tok1_norm, meta_cluster_centroid))
                                sim2 = float(np.dot(tok2_norm, meta_cluster_centroid))
                                sim1 = np.nan_to_num(sim1, nan=0.0, posinf=1.0, neginf=-1.0)
                                sim2 = np.nan_to_num(sim2, nan=0.0, posinf=1.0, neginf=-1.0)
                            # If both tokens are semantically similar to meta cluster, strong penalty
                            if sim1 > 0.35 and sim2 > 0.35:
                                meta_sequence_penalty += 1.5 * (sim1 + sim2) / 2.0
                        # If both tokens have meta roles, check bigram probability
                        if (role1 and (role1.startswith("meta") or role1 == "dataset_meta") and
                            role2 and (role2.startswith("meta") or role2 == "dataset_meta")):
                            # Check if this bigram is common (high probability = common pattern)
                            if cont[i] < len(reasoner.bigram_logp) and cont[i+1] < len(reasoner.bigram_logp[cont[i]]):
                                bigram_lp = float(reasoner.bigram_logp[cont[i], cont[i+1]])
                                # If high probability, it's a common meta-text pattern
                                if bigram_lp > -2.0:
                                    meta_sequence_penalty += 1.5  # Increased from 1.0
            
            # Check for trigram meta patterns
            if len(cont) >= 3:
                for i in range(len(cont) - 2):
                    if (0 <= cont[i] < V and 0 <= cont[i+1] < V and 0 <= cont[i+2] < V):
                        role1 = roles_by_id[cont[i]]
                        role2 = roles_by_id[cont[i+1]]
                        role3 = roles_by_id[cont[i+2]]
                        # Check semantic similarity to meta cluster for all three tokens
                        if meta_cluster_centroid is not None:
                            tok1_vec = vec[cont[i]]
                            tok2_vec = vec[cont[i+1]]
                            tok3_vec = vec[cont[i+2]]
                            tok1_norm = tok1_vec / (np.linalg.norm(tok1_vec) + 1e-8)
                            tok2_norm = tok2_vec / (np.linalg.norm(tok2_vec) + 1e-8)
                            tok3_norm = tok3_vec / (np.linalg.norm(tok3_vec) + 1e-8)
                            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                                sim1 = float(np.dot(tok1_norm, meta_cluster_centroid))
                                sim2 = float(np.dot(tok2_norm, meta_cluster_centroid))
                                sim3 = float(np.dot(tok3_norm, meta_cluster_centroid))
                                sim1 = np.nan_to_num(sim1, nan=0.0, posinf=1.0, neginf=-1.0)
                                sim2 = np.nan_to_num(sim2, nan=0.0, posinf=1.0, neginf=-1.0)
                                sim3 = np.nan_to_num(sim3, nan=0.0, posinf=1.0, neginf=-1.0)
                            # If all three are semantically similar to meta cluster, very strong penalty
                            if sim1 > 0.3 and sim2 > 0.3 and sim3 > 0.3:
                                meta_sequence_penalty += 2.5 * (sim1 + sim2 + sim3) / 3.0
                        # If all three have meta roles, strong penalty
                        if (role1 and (role1.startswith("meta") or role1 == "dataset_meta") and
                            role2 and (role2.startswith("meta") or role2 == "dataset_meta") and
                            role3 and (role3.startswith("meta") or role3 == "dataset_meta")):
                            meta_sequence_penalty += 2.5  # Increased from 2.0 - very strong penalty for meta-text phrases
        
        # Apply meta sequence penalty
        reasoner_avg -= meta_sequence_penalty
        
        # Enhanced: add sequence-level coherence score and repetition check
        sequence_coherence = 0.0
        sequence_repetition_penalty = 0.0
        if len(cont) > 1:
            # Check coherence between all token pairs in continuation (not just consecutive)
            coherence_scores = []
            unique_tokens = set()
            # Check consecutive pairs
            for i in range(len(cont) - 1):
                if 0 <= cont[i] < V and 0 <= cont[i+1] < V:
                    # Coherence check with numerical stability
                    vec1_norm_val = np.linalg.norm(vec[cont[i]])
                    vec2_norm_val = np.linalg.norm(vec[cont[i+1]])
                    if vec1_norm_val > 1e-8 and vec2_norm_val > 1e-8:
                        vec1_norm = vec[cont[i]] / vec1_norm_val
                        vec2_norm = vec[cont[i+1]] / vec2_norm_val
                        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                            coherence = float(np.dot(vec1_norm, vec2_norm))
                            coherence = np.nan_to_num(coherence, nan=0.0, posinf=1.0, neginf=-1.0)
                        coherence_scores.append(coherence)
                    # Also check bigram
                    if cont[i] < len(reasoner.bigram_logp) and cont[i+1] < len(reasoner.bigram_logp[cont[i]]):
                        bigram_lp = float(reasoner.bigram_logp[cont[i], cont[i+1]])
                        if bigram_lp > -3.0:
                            coherence_scores[-1] = coherence_scores[-1] + 0.2 if coherence_scores else 0.2
                        elif bigram_lp < -8.0:
                            coherence_scores[-1] = coherence_scores[-1] - 0.15 if coherence_scores else -0.15
                    
                    # Repetition check within continuation
                    if cont[i] in unique_tokens:
                        sequence_repetition_penalty += 0.5
                    unique_tokens.add(cont[i])
                if cont[i+1] in unique_tokens:
                    sequence_repetition_penalty += 0.5
                unique_tokens.add(cont[i+1])
            
            # Check non-consecutive pairs for broader coherence (every other token)
            if len(cont) > 2:
                for i in range(0, len(cont) - 2, 2):
                    if 0 <= cont[i] < V and 0 <= cont[i+2] < V:
                        vec1_norm_val = np.linalg.norm(vec[cont[i]])
                        vec2_norm_val = np.linalg.norm(vec[cont[i+2]])
                        if vec1_norm_val > 1e-8 and vec2_norm_val > 1e-8:
                            vec1_norm = vec[cont[i]] / vec1_norm_val
                            vec2_norm = vec[cont[i+2]] / vec2_norm_val
                            with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
                                coherence = float(np.dot(vec1_norm, vec2_norm))
                                coherence = np.nan_to_num(coherence, nan=0.0, posinf=1.0, neginf=-1.0)
                            coherence_scores.append(coherence * 0.6)  # Lower weight for non-consecutive
            
            if coherence_scores:
                sequence_coherence = float(np.mean(coherence_scores))
                # Stronger bonus for good overall coherence
                if sequence_coherence > 0.35:
                    reasoner_avg += 0.40 * (sequence_coherence - 0.35)  # Increased from 0.30
                # Stronger penalty for poor coherence
                elif sequence_coherence < 0.15:
                    reasoner_avg -= 0.35 * (0.15 - sequence_coherence)  # Increased from 0.25
        
        # Apply repetition penalty to sequence score
        reasoner_avg -= sequence_repetition_penalty

        # meta framing penalty is applied at the sequence level (strengthened)
        reasoner_avg = float(reasoner_avg) - float(w_meta_frame) * meta_frac * 1.5  # Increased penalty

        return selector_avg, reasoner_avg, meta_frac

    # normalize/clip lookahead settings
    la = int(max(1, min(16, int(lookahead_len))))
    n_cont = int(max(2, min(64, int(num_continuations))))
    commit = int(max(1, min(la, int(commit_len))))

    for step in range(int(max_new_tokens)):
        prev1 = ids[-1] if len(ids) >= 1 else None
        prev2 = ids[-2] if len(ids) >= 2 else None

        # Build candidate continuations by sampling the selector
        cont_rows: List[Dict[str, Any]] = []
        for _ in range(n_cont):
            p2, p1 = prev2, prev1
            cont: List[int] = []
            sel_lp_sum = 0.0
            for _t in range(la):
                nxt = sample_next(p2, p1)
                if nxt is None:
                    break
                tid, lp = nxt
                # hard block dataset/meta tokens during proposal if requested
                if block_dataset_meta and 0 <= tid < V and roles_by_id[tid] == "dataset_meta":
                    # try a few resamples quickly
                    tried = 0
                    ok = False
                    while tried < 3:
                        nxt2 = sample_next(p2, p1)
                        if nxt2 is None:
                            break
                        tid2, lp2 = nxt2
                        if not (block_dataset_meta and 0 <= tid2 < V and roles_by_id[tid2] == "dataset_meta"):
                            tid, lp = tid2, lp2
                            ok = True
                            break
                        tried += 1
                    if not ok and block_dataset_meta:
                        break
                cont.append(int(tid))
                sel_lp_sum += float(lp)
                p2, p1 = (p1, int(tid)) if p1 is not None else (None, int(tid))

            if not cont:
                continue

            # store sampled selector score separately (avg per token)
            cont_rows.append({"cont": cont, "sel_lp_avg": float(sel_lp_sum) / float(len(cont))})

        if not cont_rows:
            break

        # Score continuations and choose one
        scored = []
        for r in cont_rows:
            cont = r["cont"]
            s_avg = float(r["sel_lp_avg"])
            # inject selector sampled logp avg into score_continuation via sel_sum baseline
            # (score_continuation adds only online bias; we add logp here)
            sel_bias_avg, rs_avg, meta_frac = score_continuation(cont, prev2, prev1)
            selector_avg = s_avg + sel_bias_avg
            trust_avg = float(np.mean([trust[int(t)] for t in cont if 0 <= int(t) < V])) if cont else 0.0
            total = (float(alpha_selector) * selector_avg) + (float(beta_reasoner) * rs_avg) + trust_avg

            # Enhanced scoring: add quality score based on features (replaces critic/controller)
            # Compute quality features for this continuation
            ctx_for_quality = ids[-max(1, int(context_window)) :]
            if ctx_for_quality and cont:
                # Cosine similarity between context and continuation
                ctx_vec = vec[ctx_for_quality].mean(axis=0)
                cont_vec = vec[cont].mean(axis=0)
                ctx_norm = ctx_vec / (np.linalg.norm(ctx_vec) + 1e-8)
                cont_norm = cont_vec / (np.linalg.norm(cont_vec) + 1e-8)
                quality_cos = float(np.dot(ctx_norm, cont_norm))
                
                # Mean bigram logp for continuation
                quality_bigram = 0.0
                bigram_count = 0
                if len(ctx_for_quality) > 0 and len(cont) > 0:
                    prev_tok = ctx_for_quality[-1]
                    if 0 <= prev_tok < len(reasoner.bigram_logp) and 0 <= cont[0] < len(reasoner.bigram_logp[prev_tok]):
                        quality_bigram += float(reasoner.bigram_logp[prev_tok, cont[0]])
                        bigram_count += 1
                for i in range(len(cont) - 1):
                    if 0 <= cont[i] < len(reasoner.bigram_logp) and 0 <= cont[i+1] < len(reasoner.bigram_logp[cont[i]]):
                        quality_bigram += float(reasoner.bigram_logp[cont[i], cont[i+1]])
                        bigram_count += 1
                quality_bigram = quality_bigram / max(1, bigram_count) if bigram_count > 0 else -5.0
                
                # Quality score: weighted combination of features (no training needed)
                # Positive: good cosine, good bigram, low meta, low repetition
                quality_score = (
                    0.35 * quality_cos +           # Semantic coherence
                    0.25 * max(0.0, quality_bigram + 3.0) / 3.0 +  # Bigram fluency (normalized)
                    0.15 * (1.0 - meta_frac) +     # Low meta fraction
                    -0.15 * meta_frac -            # Penalty for meta
                    -0.10 * min(1.0, len(set(cont)) / max(1, len(cont)))  # Penalty for repetition
                )
                total = float(total) + 0.20 * quality_score  # Add quality boost
            else:
                quality_score = 0.0

            scored.append((cont, selector_avg, rs_avg, total, meta_frac, quality_score, 0.0))

        if not scored:
            break

        totals = np.array([x[3] for x in scored], dtype=np.float64)
        probs = softmax(totals, temp=max(1e-6, float(temperature)))
        pick = int(np.random.choice(len(scored), p=probs))
        chosen_cont, chosen_sel, chosen_rs, chosen_total, chosen_meta, chosen_quality, _ = scored[pick]

        # advantage signal: chosen continuation total vs mean alternative total
        adv = float(chosen_total - float(np.mean(totals)))

        # Commit tokens (typically 1) from the chosen continuation
        for j in range(min(commit, len(chosen_cont))):
            chosen = int(chosen_cont[j])
            prev1 = ids[-1] if len(ids) >= 1 else None
            prev2 = ids[-2] if len(ids) >= 2 else None
            kctx = ctx_key(prev2, prev1)

            # Online strengthening/weakening
            trust[chosen] = float(np.clip(trust[chosen] + float(eta_trust) * adv, -float(trust_clip), float(trust_clip)))

            # Bias selector toward the chosen token in this context
            add_bias(kctx, chosen, float(eta_bias) * adv)

            # Push down a few alternatives (from direct candidates) to keep bias sparse
            alt = selector.candidates(prev2, prev1, top_k=min(8, int(selector_top_k)))
            for alt_id, _ in alt[:4]:
                alt_id = int(alt_id)
                if alt_id == chosen:
                    continue
                add_bias(kctx, alt_id, -0.25 * float(eta_bias) * adv)

            ids.append(chosen)
            # Update topic tracker after committing token
            if 0 <= chosen < V:
                chosen_vec = vec[chosen]
                topic_tracker = topic_tracker_decay * topic_tracker + (1.0 - topic_tracker_decay) * chosen_vec
                topic_tracker = topic_tracker / (np.linalg.norm(topic_tracker) + 1e-8)

        if explain:
            # show top continuations with decoded preview
            top_show = sorted(scored, key=lambda x: x[3], reverse=True)[:min(6, len(scored))]
            explain_rows.append({
                "step": int(step),
                "adv": float(adv),
                "chosen_first": tok.vocab[int(chosen_cont[0])] if chosen_cont else "",
                "chosen_cont_preview": tok.decode(ids[:-min(commit, len(chosen_cont))] + chosen_cont[:min(len(chosen_cont), 12)]),
                "candidates": [
                    {
                        "preview": tok.decode(ids + cont[:min(len(cont), 10)]),
                        "len": int(len(cont)),
                        "selector_avg": float(sel),
                        "reasoner_avg": float(rs),
                        "total": float(tt),
                        "meta_frac": float(mf),
                        "quality_score": float(qs),
                    }
                    for cont, sel, rs, tt, mf, qs, _ in top_show
                ],
            })

    text = tok.decode(ids)
    out: Dict[str, Any] = {"text": text}
    if return_ids:
        out["_ids"] = list(ids)
        out["_prompt_len"] = int(len(prompt_ids))
    if explain:
        out["explain"] = explain_rows
    out["online"] = {
        "num_steps": int(len(ids)),
        "bias_contexts": int(len(bias)),
        "trust_nonzero": int(np.sum(np.abs(trust) > 1e-9)),
        "trust_top": [
            {"tok": tok.vocab[i], "id": int(i), "trust": float(trust[i])}
            for i in list(np.argsort(-np.abs(trust))[:10])
            if abs(float(trust[i])) > 1e-9
        ]
    }
    return out


# Removed: Reasoning Gate functions - not used in simplified generation
# These were only used for expensive veto-based filtering which is disabled


def generate_dual(
    reasoner: ReasonerArtifacts,
    selector: SelectorArtifacts,
    prompt: str,
    max_new_tokens: int = 60,
    temperature: float = 0.70,
    selector_top_k: int = 32,
    alpha_selector: float = 0.45,
    beta_reasoner: float = 0.55,
    eta_trust: float = 0.10,
    eta_bias: float = 0.08,
    trust_clip: float = 2.0,
    w_semantic: float = 1.0,
    w_anchor: float = 0.85,  # Increased from 0.55 to strengthen prompt relevance
    w_contra: float = 1.25,
    w_repeat: float = 0.85,
    repeat_window: int = 6,
    block_dataset_meta: bool = True,
    explain: bool = True,
    context_window: int = 30,  # Increased from 20 for better coherence
    lookahead_len: int = 8,
    num_continuations: int = 12,
    commit_len: int = 1,
    w_meta_frame: float = 0.45,
) -> Dict[str, Any]:
    """Generate text via dual-model debate (Selector + Reasoner)."""
    return _generate_dual_once(
        reasoner=reasoner,
        selector=selector,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        selector_top_k=selector_top_k,
        alpha_selector=alpha_selector,
        beta_reasoner=beta_reasoner,
        eta_trust=eta_trust,
        eta_bias=eta_bias,
        trust_clip=trust_clip,
        w_semantic=w_semantic,
        w_anchor=w_anchor,
        w_contra=w_contra,
        w_repeat=w_repeat,
        repeat_window=repeat_window,
        block_dataset_meta=block_dataset_meta,
        explain=explain,
        context_window=context_window,
        lookahead_len=lookahead_len,
        num_continuations=num_continuations,
        commit_len=commit_len,
        w_meta_frame=w_meta_frame,
    )
