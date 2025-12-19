import re
import numpy as np
import math
from typing import List, Dict, Any, Tuple, Optional

from .tokenizer import ClosedVocabTokenizer, simple_tokenize
from .selector import train_selector_from_seqs, SelectorArtifacts
from .utils import normalize_rows
from .model import ReasonerArtifacts, CriticArtifacts

_PUNCT_RE = re.compile(r"[^\w\s]+\Z", re.UNICODE)


def _is_punct_like(token: str) -> bool:
    return bool(_PUNCT_RE.fullmatch(token))


def _shape_features(token: str) -> Dict[str, float]:
    has_alpha = any(c.isalpha() for c in token)
    has_digit = any(c.isdigit() for c in token)
    has_underscore = "_" in token
    has_hyphen = "-" in token
    non_alnum = sum(1 for c in token if not c.isalnum())
    non_alnum_ratio = non_alnum / max(1, len(token))
    meta_shape = (float(has_underscore) + float(has_digit) + float(has_alpha and has_digit) + 
                  (0.5 if non_alnum_ratio >= 0.34 else 0.0)) / 3.5
    return {
        "has_alpha": float(has_alpha), "has_digit": float(has_digit),
        "has_underscore": float(has_underscore), "has_hyphen": float(has_hyphen),
        "non_alnum_ratio": float(non_alnum_ratio), "meta_shape": float(meta_shape),
        "len": float(len(token)), "is_punct": float(_is_punct_like(token)),
    }


def _kmeans(X: np.ndarray, k: int, iters: int = 25, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, k = X.shape[0], max(2, min(int(k), X.shape[0]))
    # Normalize and clean X to prevent numerical issues
    X = X.astype(np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
    X_norm = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    centroids = X_norm[rng.choice(n, size=k, replace=False)].copy()
    for _ in range(iters):
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            dot_product = X_norm @ centroids.T
            dot_product = np.nan_to_num(dot_product, nan=-np.inf, posinf=1.0, neginf=-np.inf)
        labels = np.argmax(dot_product, axis=1).astype(np.int32)
        new_centroids = np.zeros_like(centroids)
        for ci in range(k):
            mask = labels == ci
            if np.any(mask):
                c = X_norm[mask].mean(axis=0)
                norm_c = np.linalg.norm(c)
                if norm_c > 1e-12:
                    new_centroids[ci] = c / norm_c
                else:
                    new_centroids[ci] = X_norm[rng.integers(0, n)]
            else:
                new_centroids[ci] = X_norm[rng.integers(0, n)]
        if np.allclose(new_centroids, centroids, atol=1e-4):
            break
        centroids = new_centroids
    return labels


def build_ppmi_vectors(seqs: List[List[int]], V: int, window: int = 2, dim: int = 32) -> np.ndarray:
    C = np.zeros((V, V), dtype=np.float64)
    for s in seqs:
        for i, wi in enumerate(s):
            for j in range(max(0, i - window), min(len(s), i + window + 1)):
                if j != i:
                    C[wi, s[j]] += 1.0
    total = C.sum() + 1e-12
    P, Pr, Pc = C / total, C.sum(axis=1, keepdims=True) / total, C.sum(axis=0, keepdims=True) / total
    PPMI = np.maximum(np.log((P + 1e-12) / (Pr * Pc)), 0.0)
    U, S, _ = np.linalg.svd(PPMI, full_matrices=False)
    return (U[:, :min(dim, U.shape[1])] * np.sqrt(S[:min(dim, U.shape[1])])).astype(np.float32)


def build_bigram_logp(seqs: List[List[int]], V: int, alpha: float = 0.5) -> np.ndarray:
    counts = np.zeros((V, V), dtype=np.float64)
    for s in seqs:
        for a, b in zip(s[:-1], s[1:]):
            counts[a, b] += 1.0
    return np.log((counts + alpha) / (counts.sum(axis=1, keepdims=True) + alpha * V) + 1e-12).astype(np.float32)


def build_neighbors(seqs: List[List[int]], vocab: List[str], topn: int = 5) -> Dict[str, Dict[str, List[str]]]:
    V = len(vocab)
    before, after = np.zeros((V, V), dtype=np.float64), np.zeros((V, V), dtype=np.float64)
    for s in seqs:
        for i in range(1, len(s)):
            before[s[i], s[i-1]] += 1.0
        for i in range(len(s) - 1):
            after[s[i], s[i+1]] += 1.0
    out = {}
    for i, tok in enumerate(vocab):
        b_idx = np.argsort(-before[i])[:topn]
        a_idx = np.argsort(-after[i])[:topn]
        out[tok] = {
            "before": [vocab[j] for j in b_idx if before[i, j] > 0][:topn],
            "after": [vocab[j] for j in a_idx if after[i, j] > 0][:topn],
        }
    return out


def build_contradictions(vectors: np.ndarray, cooc_threshold: float, cooc: np.ndarray,
                        vocab: List[str], roles: Dict[str, str], sim_threshold: float = 0.55,
                        max_pairs: int = 300) -> List[Dict[str, Any]]:
    V, pairs = len(vocab), []
    # Normalize and clean vectors to prevent numerical issues
    vectors = vectors.astype(np.float64)
    vectors = np.nan_to_num(vectors, nan=0.0, posinf=1.0, neginf=-1.0)
    vectors_norm = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        sims = vectors_norm @ vectors_norm.T
        sims = np.nan_to_num(sims, nan=0.0, posinf=1.0, neginf=-1.0)
        # Clamp similarity values to valid range [-1, 1]
        sims = np.clip(sims, -1.0, 1.0)
    for i in range(V):
        for j in range(i + 1, V):
            if sims[i, j] >= sim_threshold and cooc[i, j] <= cooc_threshold and cooc[j, i] <= cooc_threshold:
                a, b = vocab[i], vocab[j]
                if roles.get(a) != "punct" and roles.get(b) != "punct":
                    penalty = round(min(1.0, (sims[i, j] - sim_threshold) * 1.5 + 0.2), 3)
                    pairs.append({"a": a, "b": b, "penalty": penalty,
                                "reason": "high semantic similarity with low co-occurrence"})
    return sorted(pairs, key=lambda x: -x["penalty"])[:max_pairs]


def _infer_sentence_end_tokens(seqs: List[List[int]], vocab: List[str]) -> List[str]:
    V = len(vocab)
    total, end = np.zeros((V,), dtype=np.float64), np.zeros((V,), dtype=np.float64)
    for s in seqs:
        for i in s:
            total[i] += 1.0
        if s:
            end[s[-1]] += 1.0
    candidates = [(end[i] / max(1, total[i]) + 0.05 * end[i], tok) for i, tok in enumerate(vocab)
                  if _is_punct_like(tok) and total[i] > 0 and end[i] >= 1.0 and end[i] / total[i] >= 0.25]
    return [t for _, t in sorted(candidates, key=lambda x: -x[0])[:6]]


def _infer_roles_and_clusters(vocab: List[str], vectors: np.ndarray, seqs: List[List[int]]) -> Tuple[Dict[str, str], Dict[str, int], List[int]]:
    V = len(vocab)
    df = np.zeros((V,), dtype=np.float64)
    for s in seqs:
        for i in set(s):
            df[i] += 1.0
    df = df / max(1.0, float(len(seqs)))
    meta_shape = np.array([_shape_features(t)["meta_shape"] for t in vocab], dtype=np.float64)
    k = int(max(4, min(12, np.sqrt(max(1, V)) + 3.0)))
    labels = _kmeans(vectors, k=k, iters=30, seed=7)
    clusters = {vocab[i]: int(labels[i]) for i in range(V)}
    seed_ids = [i for i in range(V) if meta_shape[i] >= 0.67 and not _is_punct_like(vocab[i])]
    meta_cluster_ids = []
    if len(seed_ids) >= 3:
        meta_centroid = vectors[seed_ids].mean(axis=0) / (np.linalg.norm(vectors[seed_ids].mean(axis=0)) + 1e-12)
        meta_sim = (vectors @ meta_centroid).astype(np.float64)
        cluster_scores = {ci: 0.75 * meta_sim[np.where(labels == ci)[0]].mean() + 0.25 * df[np.where(labels == ci)[0]].mean()
                         for ci in range(k) if np.any(labels == ci)}
        if cluster_scores:
            cutoff = float(np.quantile(list(cluster_scores.values()), 0.85))
            meta_cluster_ids = [int(ci) for ci, sc in sorted(cluster_scores.items(), key=lambda x: -x[1])
                              if sc >= cutoff][:2]
    roles = {}
    for i, tok in enumerate(vocab):
        if _is_punct_like(tok):
            roles[tok] = "punct"
        elif meta_cluster_ids and int(labels[i]) in meta_cluster_ids:
            roles[tok] = "dataset_meta"
        elif df[i] >= 0.55 and tok.isalpha() and len(tok) <= 4:
            roles[tok] = "function_word"
        elif tok.isalpha() and tok.endswith("ly") and len(tok) >= 4:
            roles[tok] = "adverb_like"
        else:
            roles[tok] = "content_word"
    if seqs:
        occ, fw_frame = np.zeros((V,), dtype=np.float64), np.zeros((V,), dtype=np.float64)
        is_fw = np.array([1.0 if roles.get(vocab[i]) == "function_word" else 0.0 for i in range(V)], dtype=np.float64)
        for s in seqs:
            for kpos, tid in enumerate(s):
                if 0 <= tid < V:
                    occ[tid] += 1.0
                    if 0 < kpos < len(s) - 1 and 0 <= s[kpos-1] < V and 0 <= s[kpos+1] < V:
                        if is_fw[s[kpos-1]] > 0.5 and is_fw[s[kpos+1]] > 0.5:
                            fw_frame[tid] += 1.0
        with np.errstate(divide="ignore", invalid="ignore"):
            frame_ratio = fw_frame / np.maximum(1.0, occ)
            score = frame_ratio * np.sqrt(np.maximum(0.0, df))
        # Filter valid tokens using pattern matching (no hardcoded role lists)
        valid = np.array([not _is_punct_like(vocab[i]) and 
                         roles.get(vocab[i], "") not in ("function_word", "dataset_meta")
                         for i in range(V)], dtype=bool)
        cand_scores = score[valid]
        cand_scores = cand_scores[np.isfinite(cand_scores)]
        if cand_scores.size >= 20:
            cutoff, fr_med = float(np.quantile(cand_scores, 0.95)), float(np.median(frame_ratio[valid]))
            for i in range(V):
                if valid[i] and score[i] >= cutoff and frame_ratio[i] >= fr_med:
                    # Use pattern matching instead of hardcoded list
                    role = roles.get(vocab[i], "")
                    if role and role not in ("punct", "function_word", "dataset_meta"):
                        roles[vocab[i]] = "meta_frame"
    return roles, clusters, meta_cluster_ids

# ---------------------------
# Discourse role modeling (token-by-token role state)
# ---------------------------

def _infer_discourse_structure(token_roles: Dict[str, str], vocab: List[str]) -> Tuple[List[str], List[str], Dict[str, str], Dict[str, int]]:
    """Infer discourse structure using pattern-based role-to-level mapping."""
    roles_by_id = [token_roles.get(t, "unknown") for t in vocab]
    token_role_to_level = {}
    for role in set(roles_by_id):
        if role == "punct":
            token_role_to_level[role] = "punct"
        elif role and (role.startswith("meta") or role == "dataset_meta"):
            # Pattern-based: any role starting with "meta" or exact "dataset_meta"
            token_role_to_level[role] = "meta"
        elif role and ("function" in role or "adverb" in role):
            # Pattern-based: roles containing "function" or "adverb"
            token_role_to_level[role] = "function"
        else:
            token_role_to_level[role] = "content"
    level_names = sorted(set(token_role_to_level.values()))
    role_names = [f"role_{i}" for i in range(5)]
    return role_names, level_names, token_role_to_level, {l: i for i, l in enumerate(level_names)}

def build_discourse_role_stats(seqs: List[List[int]], vocab: List[str], token_roles: Dict[str, str],
                               clusters: Dict[str, int], meta_cluster_ids: List[int],
                               sentence_end_tokens: List[str]) -> Tuple[List[str], List[str], Dict[str, str],
                               np.ndarray, np.ndarray, List[int], np.ndarray, np.ndarray, List[int]]:
    role_names, level_names, token_role_to_level, level_to_id = _infer_discourse_structure(token_roles, vocab)
    
    R = len(role_names)
    L = len(level_names)
    role_to_i = {r: i for i, r in enumerate(role_names)}
    level_to_i = level_to_id

    roles_by_id = [token_roles.get(t, "unknown") for t in vocab]
    clusters_by_id = [int(clusters.get(t, -1)) for t in vocab] if clusters else [-1] * len(vocab)

    token_to_id = {t: i for i, t in enumerate(vocab)}
    end_ids = set(int(token_to_id[t]) for t in (sentence_end_tokens or []) if t in token_to_id)

    meta_cluster_id_set = set(int(x) for x in (meta_cluster_ids or []))

    def is_meta_tok(tid: int) -> bool:
        """Check if token is meta using pattern matching."""
        tr = roles_by_id[tid]
        # Pattern-based: any role starting with "meta" or exact "dataset_meta"
        if tr and (tr.startswith("meta") or tr == "dataset_meta"):
            return True
        cid = clusters_by_id[tid]
        return (cid != -1) and (cid in meta_cluster_id_set)

    def is_punct_tok(tid: int) -> bool:
        return roles_by_id[tid] == "punct"

    pos_idx_samples = []
    for s in seqs:
        if not s:
            continue
        pos = 0
        for tid in s:
            if 0 <= tid < len(vocab):
                pos_idx_samples.append(int(pos))
                pos += 1
                if tid in end_ids:
                    pos = 0
    if pos_idx_samples:
        q1 = max(0, int(np.quantile(np.array(pos_idx_samples, dtype=np.float64), 0.33)))
        q2 = max(q1 + 1, int(np.quantile(np.array(pos_idx_samples, dtype=np.float64), 0.66)))
    else:
        q1, q2 = 3, 8
    pos_thresholds = [int(q1), int(q2)]
    bucket_from_pos = lambda p: 0 if p <= q1 else (1 if p <= q2 else 2)

    trans, emit = np.zeros((R, R), dtype=np.float64), np.zeros((R, L), dtype=np.float64)
    trans_pos, emit_pos = np.zeros((3, R, R), dtype=np.float64), np.zeros((3, R, L), dtype=np.float64)
    runlens = [[] for _ in range(R)]
    for s in seqs:
        if not s:
            continue
        prev_role, pos_in_sentence, cur_run_role, cur_run_len = 0, 0, None, 0
        for tid in s:
            if tid < 0 or tid >= len(vocab):
                continue
            b = bucket_from_pos(pos_in_sentence)
            tr, lvl = roles_by_id[tid], token_role_to_level.get(roles_by_id[tid], "content")
            if is_meta_tok(tid):
                lvl = "meta"
            if tid in end_ids or is_punct_tok(tid):
                lvl = "punct" if lvl != "meta" else lvl
            li, cur = level_to_i.get(lvl, 0), int(np.argmax(np.ones((R,), dtype=np.float64) / R))
            trans[prev_role, cur] += 1.0
            trans_pos[b, prev_role, cur] += 1.0
            emit[cur, li] += 1.0
            emit_pos[b, cur, li] += 1.0
            if cur_run_role == cur:
                cur_run_len += 1
            else:
                if cur_run_role is not None and cur_run_len > 0:
                    runlens[cur_run_role].append(int(cur_run_len))
                cur_run_role, cur_run_len = cur, 1
            prev_role = cur
            pos_in_sentence = 0 if tid in end_ids else pos_in_sentence + 1
        if cur_run_role is not None and cur_run_len > 0:
            runlens[cur_run_role].append(int(cur_run_len))
    smoothing = 1.0
    trans = (trans + smoothing) / np.maximum(1e-12, (trans + smoothing).sum(axis=1, keepdims=True))
    emit = (emit + smoothing) / np.maximum(1e-12, (emit + smoothing).sum(axis=1, keepdims=True))
    trans_pos = (trans_pos + smoothing) / np.maximum(1e-12, (trans_pos + smoothing).sum(axis=2, keepdims=True))
    emit_pos = (emit_pos + smoothing) / np.maximum(1e-12, (emit_pos + smoothing).sum(axis=2, keepdims=True))
    min_run = [int(max(2, min(18, round(float(np.quantile(np.array(runlens[r] if runlens[r] else [3.0], dtype=np.float64), 0.25)))))) for r in range(R)]

    return (
        role_names,
        level_names,
        token_role_to_level,  # Return learned mapping
        trans.astype(np.float32),
        np.log(emit.astype(np.float32) + 1e-12),
        pos_thresholds,
        trans_pos.astype(np.float32),
        np.log(emit_pos.astype(np.float32) + 1e-12),
        min_run,
    )



def train_reasoner(texts: List[str], token_desc: Dict[str, str], dim: int = 32,
                  window: int = 2, desc_alpha: float = 0.35) -> ReasonerArtifacts:
    tok = ClosedVocabTokenizer.from_texts(texts)
    seqs = [tok.encode(t) for t in texts]
    V = len(tok.vocab)
    cooc = np.zeros((V, V), dtype=np.float64)
    for s in seqs:
        for i, wi in enumerate(s):
            for j in range(max(0, i - window), min(len(s), i + window + 1)):
                if j != i:
                    cooc[wi, s[j]] += 1.0
    X = build_ppmi_vectors(seqs, V, window=window, dim=dim)
    X2 = X.astype(np.float32).copy()
    for token, desc in token_desc.items():
        if desc.strip() and token in tok.token_to_id:
            desc_toks = [t for t in simple_tokenize(desc) if t in tok.token_to_id]
            if desc_toks:
                ids = [tok.token_to_id[t] for t in desc_toks]
                X2[tok.token_to_id[token]] = (1.0 - desc_alpha) * X2[tok.token_to_id[token]] + desc_alpha * X[ids].mean(axis=0)
    Xn = normalize_rows(X2)
    roles, clusters, meta_cluster_ids = _infer_roles_and_clusters(tok.vocab, Xn, seqs)
    discourse_role_names, discourse_level_names, token_role_to_level, discourse_trans, discourse_emit_logp, discourse_pos_thresholds, discourse_trans_pos, discourse_emit_logp_pos, discourse_role_min_run = build_discourse_role_stats(seqs, tok.vocab, roles, clusters, meta_cluster_ids, _infer_sentence_end_tokens(seqs, tok.vocab))
    return ReasonerArtifacts(vocab=tok.vocab, token_to_id=tok.token_to_id, vectors=Xn.astype(np.float32),
                            bigram_logp=build_bigram_logp(seqs, V).astype(np.float32), roles=roles,
                            neighbors=build_neighbors(seqs, tok.vocab), contradiction_pairs=build_contradictions(Xn, 0.0, cooc, tok.vocab, roles),
                            clusters=clusters, meta_cluster_ids=meta_cluster_ids, sentence_end_tokens=_infer_sentence_end_tokens(seqs, tok.vocab),
                            discourse_role_names=discourse_role_names, discourse_level_names=discourse_level_names,
                            token_role_to_level=token_role_to_level, discourse_trans=discourse_trans, discourse_emit_logp=discourse_emit_logp,
                            discourse_pos_thresholds=discourse_pos_thresholds, discourse_trans_pos=discourse_trans_pos,
                            discourse_emit_logp_pos=discourse_emit_logp_pos, discourse_role_min_run=discourse_role_min_run)


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _critic_feature_names() -> List[str]:
    return [
        "cos_ctx_cont",
        "mean_bigram_logp",
        "repeat_frac",
        "meta_frac",
        "contra_mean",
        "len_ratio",
    ]


def _critic_features(
    art: ReasonerArtifacts,
    ctx_ids: List[int],
    cont_ids: List[int],
    roles_by_id: Optional[List[str]] = None,
    contra_lookup: Optional[Dict[Tuple[int, int], float]] = None,
    contra_window: int = 6,
) -> np.ndarray:
    V = len(art.vocab)
    if roles_by_id is None:
        roles_by_id = [art.roles.get(t, "unknown") for t in art.vocab]

    if contra_lookup is None:
        contra_lookup = {}
        for p in art.contradiction_pairs or []:
            a = art.token_to_id.get(p.get("a"))
            b = art.token_to_id.get(p.get("b"))
            if a is None or b is None:
                continue
            pen = float(p.get("penalty", 0.0) or 0.0)
            if pen > 0:
                contra_lookup[(a, b)] = max(contra_lookup.get((a, b), 0.0), pen)
                contra_lookup[(b, a)] = max(contra_lookup.get((b, a), 0.0), pen)

    ctx_ids = ctx_ids[-max(1, int(contra_window)):]
    cont_ids = cont_ids[:]
    if not ctx_ids or not cont_ids:
        return np.zeros((6,), dtype=np.float32)

    vec = art.vectors
    ctx = vec[ctx_ids].mean(axis=0)
    cont = vec[cont_ids].mean(axis=0)
    # cosine similarity
    cos = float(ctx @ cont)

    # mean bigram logp over internal transitions
    bl = 0.0
    cnt = 0
    prev = ctx_ids[-1]
    bl += float(art.bigram_logp[prev, cont_ids[0]])
    cnt += 1
    for a, b in zip(cont_ids[:-1], cont_ids[1:]):
        bl += float(art.bigram_logp[a, b])
        cnt += 1
    mean_bigram = bl / max(1, cnt)

    # repetition fraction within continuation
    rep = 0
    if len(cont_ids) > 1:
        rep = len(cont_ids) - len(set(cont_ids))
    repeat_frac = float(rep / max(1, len(cont_ids)))

    # meta fraction (pattern-based, no hardcoded list)
    meta_roles = 0
    for tid in cont_ids:
        r = roles_by_id[tid] if 0 <= tid < V else "unknown"
        # Pattern-based detection: starts with "meta" or exact "dataset_meta"
        if r and (r.startswith("meta") or r == "dataset_meta"):
            meta_roles += 1
    meta_frac = float(meta_roles / max(1, len(cont_ids)))

    # contradiction mean between cont tokens and recent ctx
    contra_sum = 0.0
    contra_cnt = 0
    recent = ctx_ids[-max(1, int(contra_window)):]
    for a in recent:
        for b in cont_ids:
            contra_sum += float(contra_lookup.get((a, b), 0.0))
            contra_cnt += 1
    contra_mean = float(contra_sum / max(1, contra_cnt))

    len_ratio = float(len(cont_ids) / max(1, len(ctx_ids)))

    return np.array([cos, mean_bigram, repeat_frac, meta_frac, contra_mean, len_ratio], dtype=np.float32)


def train_critic_from_seqs(
    art: ReasonerArtifacts,
    seqs: List[List[int]],
    steps: int = 3000,
    lr: float = 0.05,
    ctx_window: int = 10,
    cont_len: int = 4,
    num_negs: int = 3,
    seed: int = 7,
    l2: float = 1e-4,
) -> Tuple[CriticArtifacts, float]:
    """Self-supervised critic: distinguish real continuation vs negatives."""
    rng = np.random.default_rng(int(seed))
    feat_names = _critic_feature_names()
    w = np.zeros((len(feat_names),), dtype=np.float64)
    b = 0.0

    roles_by_id = [art.roles.get(t, "unknown") for t in art.vocab]

    # pre-build contradiction lookup
    contra_lookup = {}
    for p in art.contradiction_pairs or []:
        a = art.token_to_id.get(p.get("a"))
        b2 = art.token_to_id.get(p.get("b"))
        if a is None or b2 is None:
            continue
        pen = float(p.get("penalty", 0.0) or 0.0)
        if pen > 0:
            contra_lookup[(a, b2)] = max(contra_lookup.get((a, b2), 0.0), pen)
            contra_lookup[(b2, a)] = max(contra_lookup.get((b2, a), 0.0), pen)

    def step_loss(logit: float, y: int) -> float:
        # binary cross entropy with logits
        if y == 1:
            return float(np.log1p(np.exp(-logit)))
        return float(np.log1p(np.exp(logit)))

    losses: List[float] = []
    # flatten indices for fast random negatives
    all_positions = []
    for si, s in enumerate(seqs):
        if len(s) >= (ctx_window + cont_len + 1):
            all_positions.append(si)
    if not all_positions:
        # fallback: critic disabled
        critic = CriticArtifacts(vocab=art.vocab, token_to_id=art.token_to_id, weights=w.tolist(), bias=float(b),
                                 meta={"feature_names": feat_names, "note": "insufficient sequences for critic training"})
        return critic, 0.0

    for _ in range(max(1, int(steps))):
        si = int(rng.choice(all_positions))
        s = seqs[si]
        if len(s) < (ctx_window + cont_len + 1):
            continue
        i = int(rng.integers(low=ctx_window, high=len(s) - cont_len))
        ctx = s[i - ctx_window:i]
        pos = s[i:i + cont_len]

        f = _critic_features(art, ctx, pos, roles_by_id=roles_by_id, contra_lookup=contra_lookup)
        logit = float(w @ f + b)
        p = _sigmoid(logit)
        # gradient ascent on log-likelihood: (y - p) * x
        g = (1.0 - p)
        w += lr * (g * f - l2 * w)
        b += lr * g
        losses.append(step_loss(logit, 1))

        # negatives: random continuation slices from other sequences or random positions
        for _n in range(max(1, int(num_negs))):
            sj = int(rng.choice(all_positions))
            s2 = seqs[sj]
            if len(s2) < cont_len:
                continue
            j = int(rng.integers(low=0, high=max(1, len(s2) - cont_len)))
            neg = s2[j:j + cont_len]
            fn = _critic_features(art, ctx, neg, roles_by_id=roles_by_id, contra_lookup=contra_lookup)
            logitn = float(w @ fn + b)
            pn = _sigmoid(logitn)
            gn = (0.0 - pn)
            w += lr * (gn * fn - l2 * w)
            b += lr * gn
            losses.append(step_loss(logitn, 0))

    critic = CriticArtifacts(
        vocab=art.vocab,
        token_to_id=art.token_to_id,
        weights=w.astype(np.float32).tolist(),
        bias=float(b),
        meta={
            "feature_names": feat_names,
            "steps": int(steps),
            "lr": float(lr),
            "ctx_window": int(ctx_window),
            "cont_len": int(cont_len),
            "num_negs": int(num_negs),
            "seed": int(seed),
            "l2": float(l2),
        },
    )
    return critic, float(np.mean(losses)) if losses else 0.0


def train_dual(
    texts: List[str],
    token_desc: Dict[str, str],
    dim: int = 32,
    window: int = 2,
    desc_alpha: float = 0.35,
    selector_smooth: float = 0.5,
    selector_max_per_context: int = 256,
    critic_steps: int = 3000,
    critic_lr: float = 0.05,
    critic_ctx_window: int = 10,
    critic_cont_len: int = 4,
    critic_num_negs: int = 3,
    seed: int = 7,
) -> Tuple[ReasonerArtifacts, SelectorArtifacts, CriticArtifacts, Dict[str, float]]:
    """Train: reasoner + selector + outcome critic from the same data."""
    art = train_reasoner(texts=texts, token_desc=token_desc, dim=dim, window=window, desc_alpha=desc_alpha)
    tok = ClosedVocabTokenizer(vocab=art.vocab, token_to_id=art.token_to_id)
    seqs = [tok.encode(t) for t in texts if t and t.strip()]

    sel = train_selector_from_seqs(vocab=art.vocab, seqs=seqs, smooth=selector_smooth, max_per_context=selector_max_per_context)

    critic, critic_loss = train_critic_from_seqs(
        art=art,
        seqs=seqs,
        steps=critic_steps,
        lr=critic_lr,
        ctx_window=critic_ctx_window,
        cont_len=critic_cont_len,
        num_negs=critic_num_negs,
        seed=seed,
    )

    metrics = {
        "critic_loss": float(critic_loss),
        "selector_contexts_bigram": float(len(sel.bigram_logp)),
        "selector_contexts_trigram": float(len(sel.trigram_logp)),
    }
    return art, sel, critic, metrics
