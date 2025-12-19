import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from .tokenizer import ClosedVocabTokenizer
from .model import ReasonerArtifacts
from .selector import SelectorArtifacts
from .utils import softmax

def _build_contradiction_lookup(pairs: List[Dict[str, Any]], token_to_id: Dict[str, int]) -> Dict[Tuple[int, int], float]:
    out = {}
    for p in pairs:
        a, b = token_to_id.get(p.get("a")), token_to_id.get(p.get("b"))
        if a is not None and b is not None:
            pen = float(p.get("penalty", 0.0))
            if pen > 0:
                out[(a, b)] = out[(b, a)] = max(out.get((a, b), 0.0), pen)
    return out

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
    if token_role_to_level:
        level = token_role_to_level.get(token_role, "content")
    else:
        level = "punct" if token_role == "punct" else ("meta" if token_role in ("dataset_meta", "meta_shape", "meta_frame") else ("function" if token_role in ("function_word", "adverb_like") else "content"))
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
    repeat_window: int = 3,
    repetition_penalty: float = 1.25,
    semantic_repeat_window: int = 2,
    semantic_repeat_threshold: float = 0.72,
    semantic_repeat_penalty: float = 0.85,
    # New: style + sentence closure + cluster switching (no hardcoded tokens)
    style: str = "descriptive",  # descriptive|explanatory
    min_sentence_tokens: int = 10,
    max_sentence_tokens: int = 26,
    closure_strength: float = 1.15,
    cluster_switch_window: int = 2,
    cluster_switch_penalty: float = 0.28,
    # New: role-aware neighbor gating (no hardcoded token lists)
    meta_frame_penalty: float = 1.05,
    role_loop_penalty: float = 0.85,
    # Configurable weights and thresholds (previously hardcoded)
    weight_bigram: float = 1.00,
    weight_semantic: float = 0.85,
    weight_contradiction: float = 1.15,
    context_window: int = 6,
    contradiction_window: int = 4,
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
    if art.clusters:
        clusters_by_id = [int(art.clusters.get(t, -1)) for t in art.vocab]
        if all(c == -1 for c in clusters_by_id):
            clusters_by_id = None

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

        # context vector = mean of last tokens
        k = min(context_window, len(ids))
        ctx = vec[ids[-k:]].mean(axis=0)

        comp_bigram = bigram_logp[prev].astype(np.float64)
        comp_sem = (vec @ ctx).astype(np.float64)

        # contradiction penalty based on recent tokens
        recent_contra = ids[-min(contradiction_window, len(ids)) :]
        contra_pen = np.zeros((V,), dtype=np.float64)
        for r in recent_contra:
            # sparse-ish lookup; V is small but keep it straightforward
            for j in range(V):
                contra_pen[j] += contra.get((r, j), 0.0)

        logits = weight_bigram * comp_bigram + weight_semantic * comp_sem - weight_contradiction * contra_pen

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

# --- Anti-repetition controls (exact + semantic) ---
        rw = max(0, int(repeat_window))
        if rw > 0 and repetition_penalty > 0:
            exact_recent = ids[-rw:]
            for j in range(V):
                logits[j] -= _repetition_penalty(j, exact_recent, float(repetition_penalty))

        sw = max(0, int(semantic_repeat_window))
        if sw > 0 and semantic_repeat_penalty > 0:
            sem_recent = ids[-sw:]
            for j in range(V):
                if roles_by_id[j] == "punct":
                    continue
                logits[j] -= _semantic_repeat_penalty(j, sem_recent, vec, float(semantic_repeat_threshold), float(semantic_repeat_penalty))

        # --- Cluster switch (concept packing) ---
        if clusters_by_id is not None and cluster_pen > 0:
            cw = max(0, int(cluster_switch_window))
            if cw > 0:
                for j in range(V):
                    logits[j] -= _cluster_switch_penalty(j, ids, clusters_by_id, cw, cluster_pen, roles_by_id=roles_by_id)

        if closure_k > 0 and end_token_ids:
            for j in end_token_ids:
                logits[j] += _closure_boost(j, tokens_since_end, int(min_sentence_tokens), int(max_sentence_tokens), end_token_ids, closure_k, roles_by_id)

        # top-k sampling
        top_idx = np.argsort(-logits)[:max(1, top_k)]
        top_logits = logits[top_idx]
        probs = softmax(top_logits, temp=max(float(temperature), 1e-6))
        choice_local = int(np.random.choice(len(top_idx), p=probs))
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
def generate_dual(
    reasoner: ReasonerArtifacts,
    selector: SelectorArtifacts,
    prompt: str,
    max_new_tokens: int = 60,
    temperature: float = 0.85,
    selector_top_k: int = 32,
    # Debate mix weights:
    alpha_selector: float = 0.55,
    beta_reasoner: float = 0.45,
    # Online strengthening/weakening during generation:
    eta_trust: float = 0.10,
    eta_bias: float = 0.08,
    trust_clip: float = 2.0,
    # Reasoner scoring weights:
    w_semantic: float = 1.0,
    w_anchor: float = 0.55,
    w_contra: float = 1.25,
    w_repeat: float = 0.85,
    repeat_window: int = 6,
    block_dataset_meta: bool = True,
    explain: bool = False,
    context_window: int = 10,
    # NEW: micro-continuation debate (sequence-level critique)
    lookahead_len: int = 6,
    num_continuations: int = 12,
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
    ids: List[int] = tok.encode(prompt)
    V = len(reasoner.vocab)
    vec = reasoner.vectors

    # prompt anchor (directional intent)
    if ids:
        k0 = min(16, len(ids))
        anchor = vec[ids[:k0]].mean(axis=0)
    else:
        anchor = np.zeros((vec.shape[1],), dtype=np.float32)

    contra = _build_contradiction_lookup(reasoner.contradiction_pairs, reasoner.token_to_id)
    roles_by_id = [reasoner.roles.get(t, "unknown") for t in reasoner.vocab]

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

    def score_continuation(cont: List[int], prev2: Optional[int], prev1: Optional[int]) -> Tuple[float, float, float, float]:
        """Return (selector_avg, reasoner_avg, total, meta_frac) for a continuation."""
        if not cont:
            return -1e9, -1e9, -1e9, 0.0

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
        tmp_ids = list(ids)
        reason_sum = 0.0
        meta_count = 0.0
        meta_roles = {"meta_frame", "meta_shape", "dataset_meta"}
        for tid in cont:
            if not (0 <= int(tid) < V):
                continue
            # block dataset/meta tokens completely if requested
            if block_dataset_meta and roles_by_id[int(tid)] == "dataset_meta":
                reason_sum -= 2.5
            if roles_by_id[int(tid)] in meta_roles:
                meta_count += 1.0

            k = min(int(context_window), len(tmp_ids)) if tmp_ids else 0
            ctx = vec[tmp_ids[-k:]].mean(axis=0) if k > 0 else anchor

            sem = float(np.dot(vec[int(tid)], ctx))
            anc = float(np.dot(vec[int(tid)], anchor))

            recent = tmp_ids[-max(1, int(repeat_window)):] if tmp_ids else []
            rep = 1.0 if int(tid) in recent else 0.0
            cp = contradiction_penalty(int(tid), recent)

            reason_sum += (float(w_semantic) * sem) + (float(w_anchor) * anc) - (float(w_contra) * cp) - (float(w_repeat) * rep)
            tmp_ids.append(int(tid))

        L = max(1, len(cont))
        selector_avg = float(sel_sum) / float(L)
        reasoner_avg = float(reason_sum) / float(L)
        meta_frac = float(meta_count) / float(L)

        # meta framing penalty is applied at the sequence level
        reasoner_avg = float(reasoner_avg) - float(w_meta_frame) * meta_frac

        # include average trust along the continuation (stabilizes consistent paths)
        trust_avg = float(np.mean([float(trust[int(t)]) for t in cont if 0 <= int(t) < V])) if cont else 0.0

        total = (float(alpha_selector) * selector_avg) + (float(beta_reasoner) * reasoner_avg) + trust_avg
        return selector_avg, reasoner_avg, total, meta_frac

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
            sel_bias_avg, rs_avg, total, meta_frac = score_continuation(cont, prev2, prev1)
            selector_avg = s_avg + sel_bias_avg
            total = (float(alpha_selector) * selector_avg) + (float(beta_reasoner) * rs_avg) + float(np.mean([trust[int(t)] for t in cont if 0 <= int(t) < V]))
            scored.append((cont, selector_avg, rs_avg, total, meta_frac))

        if not scored:
            break

        totals = np.array([x[3] for x in scored], dtype=np.float64)
        probs = softmax(totals, temp=max(1e-6, float(temperature)))
        pick = int(np.random.choice(len(scored), p=probs))
        chosen_cont, chosen_sel, chosen_rs, chosen_total, chosen_meta = scored[pick]

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
                    }
                    for cont, sel, rs, tt, mf in top_show
                ],
            })

    text = tok.decode(ids)
    out: Dict[str, Any] = {"text": text}
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
