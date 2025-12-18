import re
import numpy as np
from typing import List, Dict, Any, Tuple

from .tokenizer import ClosedVocabTokenizer, simple_tokenize
from .utils import normalize_rows
from .model import ReasonerArtifacts
from .selector import SelectorArtifacts, train_selector_from_seqs

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
    centroids = X[rng.choice(n, size=k, replace=False)].copy()
    for _ in range(iters):
        labels = np.argmax(X @ centroids.T, axis=1).astype(np.int32)
        new_centroids = np.zeros_like(centroids)
        for ci in range(k):
            mask = labels == ci
            if np.any(mask):
                c = X[mask].mean(axis=0)
                new_centroids[ci] = c / (np.linalg.norm(c) + 1e-12)
            else:
                new_centroids[ci] = X[rng.integers(0, n)]
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
    V, pairs, sims = len(vocab), [], vectors @ vectors.T
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
        valid = np.array([not _is_punct_like(vocab[i]) and roles.get(vocab[i]) not in ("function_word", "dataset_meta")
                         for i in range(V)], dtype=bool)
        cand_scores = score[valid]
        cand_scores = cand_scores[np.isfinite(cand_scores)]
        if cand_scores.size >= 20:
            cutoff, fr_med = float(np.quantile(cand_scores, 0.95)), float(np.median(frame_ratio[valid]))
            for i in range(V):
                if valid[i] and score[i] >= cutoff and frame_ratio[i] >= fr_med:
                    if roles.get(vocab[i]) not in ("punct", "function_word", "dataset_meta"):
                        roles[vocab[i]] = "meta_frame"
    return roles, clusters, meta_cluster_ids

# ---------------------------
# Discourse role modeling (token-by-token role state)
# ---------------------------

def _infer_discourse_structure(token_roles: Dict[str, str], vocab: List[str]) -> Tuple[List[str], List[str], Dict[str, str], Dict[str, int]]:
    roles_by_id = [token_roles.get(t, "unknown") for t in vocab]
    token_role_to_level = {}
    for role in set(roles_by_id):
        if role == "punct":
            token_role_to_level[role] = "punct"
        elif role in ("dataset_meta", "meta_shape", "meta_frame"):
            token_role_to_level[role] = "meta"
        elif role in ("function_word", "adverb_like"):
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
        tr = roles_by_id[tid]
        if tr in ("dataset_meta", "meta_shape", "meta_frame"):
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

# ---------------- Dual-model training (Reasoner + Selector) ----------------
def train_dual(
    texts: List[str],
    token_desc: Dict[str, str],
    dim: int = 32,
    window: int = 2,
    desc_alpha: float = 0.35,
    selector_smooth: float = 0.5,
    selector_max_per_context: int = 256,
) -> Tuple[ReasonerArtifacts, SelectorArtifacts]:
    """Train two models from the same data:
    - ReasonerArtifacts: semantic vectors + contradiction-aware metadata (no next-token CE).
    - SelectorArtifacts: sparse trigram/bigram transitions for fluent token selection.

    Both share the same vocab / token IDs (Reasoner vocab).
    """
    reasoner = train_reasoner(texts=texts, token_desc=token_desc, dim=dim, window=window, desc_alpha=desc_alpha)
    tok = ClosedVocabTokenizer(vocab=reasoner.vocab, token_to_id=reasoner.token_to_id)
    seqs = [tok.encode(t) for t in texts]
    selector = train_selector_from_seqs(
        vocab=reasoner.vocab,
        seqs=seqs,
        smooth=selector_smooth,
        max_per_context=selector_max_per_context,
    )
    return reasoner, selector
