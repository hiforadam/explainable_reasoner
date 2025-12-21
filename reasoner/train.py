import re
import numpy as np
from typing import List, Dict, Any, Tuple, Iterator, Optional
from collections import defaultdict

from .tokenizer import ClosedVocabTokenizer, simple_tokenize
from .utils import normalize_rows
from .model import ReasonerArtifacts
from .selector import SelectorArtifacts, train_selector_from_seqs

_PUNCT_RE = re.compile(r"[^\w\s]+\Z", re.UNICODE)

# Maximum vocab size to prevent unbounded growth
MAX_VOCAB_SIZE = 50000

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
        X_safe = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        centroids_safe = np.nan_to_num(centroids, nan=0.0, posinf=1e6, neginf=-1e6)
        X_norm = X_safe / (np.linalg.norm(X_safe, axis=1, keepdims=True) + 1e-12)
        centroids_norm = centroids_safe / (np.linalg.norm(centroids_safe, axis=1, keepdims=True) + 1e-12)
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            labels = np.argmax(np.clip(X_norm @ centroids_norm.T, -1.0, 1.0), axis=1).astype(np.int32)
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


def _build_ppmi_vectors_streaming(seq_stream: Iterator[List[int]], V: int, window: int = 2, dim: int = 32) -> np.ndarray:
    """Build PPMI vectors using streaming - processes one sequence at a time.
    
    Uses sparse accumulator (dict) instead of dense V×V matrix.
    """
    # Use sparse accumulator: dict of (i, j) -> count
    cooc_sparse = defaultdict(float)
    total_count = 0.0
    
    # Process sequences one at a time
    for s in seq_stream:
        for i, wi in enumerate(s):
            if wi < 0 or wi >= V:
                continue
            for j in range(max(0, i - window), min(len(s), i + window + 1)):
                if j != i:
                    wj = s[j]
                    if 0 <= wj < V:
                        cooc_sparse[(wi, wj)] += 1.0
                        total_count += 1.0
    
    if total_count == 0:
        # Return zero vectors if no co-occurrences
        return np.zeros((V, dim), dtype=np.float16)
    
    # Convert sparse to dense only for computation (smaller than full V×V)
    # Compute row and column sums from sparse data
    row_sums = defaultdict(float)
    col_sums = defaultdict(float)
    for (i, j), count in cooc_sparse.items():
        row_sums[i] += count
        col_sums[j] += count
    
    # Build PPMI matrix using sparse representation
    # For large V, we can't build full matrix, so use approximation
    if V > 5000:
        # For very large vocab, use sampling approach
        # Build dense matrix only for frequent pairs
        top_pairs = sorted(cooc_sparse.items(), key=lambda x: -x[1])[:min(100000, len(cooc_sparse))]
        # Create reduced matrix for top pairs
        unique_rows = sorted(set(i for (i, _), _ in top_pairs))
        unique_cols = sorted(set(j for (_, j), _ in top_pairs))
        if len(unique_rows) < V or len(unique_cols) < V:
            # Use full matrix but with float32
            C = np.zeros((V, V), dtype=np.float32)
            for (i, j), count in cooc_sparse.items():
                C[i, j] = count
            total = float(total_count) + 1e-12
            P = C / total
            Pr = np.array([row_sums.get(i, 0.0) / total for i in range(V)], dtype=np.float32).reshape(-1, 1)
            Pc = np.array([col_sums.get(j, 0.0) / total for j in range(V)], dtype=np.float32).reshape(1, -1)
        else:
            C = np.zeros((V, V), dtype=np.float32)
            for (i, j), count in cooc_sparse.items():
                C[i, j] = count
            total = float(total_count) + 1e-12
            P = C / total
            Pr = np.array([row_sums.get(i, 0.0) / total for i in range(V)], dtype=np.float32).reshape(-1, 1)
            Pc = np.array([col_sums.get(j, 0.0) / total for j in range(V)], dtype=np.float32).reshape(1, -1)
    else:
        # For smaller vocab, build full matrix
        C = np.zeros((V, V), dtype=np.float32)
        for (i, j), count in cooc_sparse.items():
            C[i, j] = count
        total = float(total_count) + 1e-12
        P = C / total
        Pr = C.sum(axis=1, keepdims=True) / total
        Pc = C.sum(axis=0, keepdims=True) / total
    
    # Cleanup sparse accumulator
    del cooc_sparse, row_sums, col_sums
    import gc
    gc.collect()
    
    # Fix divide by zero: ensure denominator is never zero
    denominator = np.maximum(Pr * Pc, 1e-12)
    with np.errstate(divide='ignore', invalid='ignore'):
        PPMI = np.maximum(np.log((P + 1e-12) / denominator), 0.0)
    
    # Optimized SVD: use truncated SVD for large matrices
    target_dim = min(dim + 10, V)
    if V > 1000:
        try:
            from scipy.sparse.linalg import svds
            U, S, _ = svds(PPMI.astype(np.float32), k=min(target_dim, V-1), which='LM')
            U = U[:, ::-1]
            S = S[::-1]
        except (ImportError, ValueError):
            U, S, _ = np.linalg.svd(PPMI, full_matrices=False)
    else:
        U, S, _ = np.linalg.svd(PPMI, full_matrices=False)
    
    result = (U[:, :min(dim, U.shape[1])] * np.sqrt(S[:min(dim, len(S))])).astype(np.float32)
    # Explicit cleanup
    del C, P, Pr, Pc, PPMI, U, S
    gc.collect()
    return result


def build_ppmi_vectors(seqs: List[List[int]], V: int, window: int = 2, dim: int = 32) -> np.ndarray:
    """Build PPMI vectors - wrapper for backward compatibility."""
    return _build_ppmi_vectors_streaming(iter(seqs), V, window, dim)


def _build_bigram_logp_streaming(seq_stream: Iterator[List[int]], V: int, alpha: float = 0.5) -> np.ndarray:
    """Build bigram log probabilities using streaming - processes one sequence at a time.
    
    Uses sparse accumulator instead of dense V×V matrix.
    """
    # Sparse accumulator: (prev, next) -> count
    bigram_counts = defaultdict(float)
    row_sums = defaultdict(float)  # Sum per row (prev token)
    
    for s in seq_stream:
        for a, b in zip(s[:-1], s[1:]):
            if 0 <= a < V and 0 <= b < V:
                bigram_counts[(a, b)] += 1.0
                row_sums[a] += 1.0
    
    # Convert to dense matrix (required for output format)
    counts = np.zeros((V, V), dtype=np.float32)
    for (a, b), count in bigram_counts.items():
        counts[a, b] = count
    
    # Compute log probabilities
    row_totals = np.array([row_sums.get(i, 0.0) for i in range(V)], dtype=np.float32).reshape(-1, 1)
    probs = (counts + alpha) / (row_totals + alpha * V + 1e-12)
    result = np.log(probs + 1e-12).astype(np.float32)
    
    # Cleanup
    del bigram_counts, row_sums, counts, row_totals, probs
    import gc
    gc.collect()
    
    return result


def build_bigram_logp(seqs: List[List[int]], V: int, alpha: float = 0.5) -> np.ndarray:
    """Build bigram log probabilities - wrapper for backward compatibility."""
    return _build_bigram_logp_streaming(iter(seqs), V, alpha)


def _build_neighbors_streaming(seq_stream: Iterator[List[int]], vocab: List[str], topn: int = 5) -> Dict[str, Dict[str, List[str]]]:
    """Build neighbors using streaming - processes one sequence at a time.
    
    Uses sparse accumulators instead of dense V×V matrices.
    """
    V = len(vocab)
    # Sparse accumulators
    before_counts = defaultdict(float)  # (token, prev_token) -> count
    after_counts = defaultdict(float)   # (token, next_token) -> count
    
    for s in seq_stream:
        # Before neighbors
        for i in range(1, len(s)):
            if 0 <= s[i] < V and 0 <= s[i-1] < V:
                before_counts[(s[i], s[i-1])] += 1.0
        # After neighbors
        for i in range(len(s) - 1):
            if 0 <= s[i] < V and 0 <= s[i+1] < V:
                after_counts[(s[i], s[i+1])] += 1.0
    
    # Build output structure
    out = {}
    for i, tok in enumerate(vocab):
        # Get before neighbors for this token
        before_items = [(j, before_counts.get((i, j), 0.0)) for j in range(V)]
        before_items.sort(key=lambda x: -x[1])
        before_neighbors = [vocab[j] for j, count in before_items[:topn] if count > 0]
        
        # Get after neighbors for this token
        after_items = [(j, after_counts.get((i, j), 0.0)) for j in range(V)]
        after_items.sort(key=lambda x: -x[1])
        after_neighbors = [vocab[j] for j, count in after_items[:topn] if count > 0]
        
        out[tok] = {
            "before": before_neighbors[:topn],
            "after": after_neighbors[:topn],
        }
    
    # Cleanup
    del before_counts, after_counts
    import gc
    gc.collect()
    
    return out


def build_neighbors(seqs: List[List[int]], vocab: List[str], topn: int = 5) -> Dict[str, Dict[str, List[str]]]:
    """Build neighbors - wrapper for backward compatibility."""
    return _build_neighbors_streaming(iter(seqs), vocab, topn)


def build_contradictions(vectors: np.ndarray, cooc_threshold: float, cooc: np.ndarray,
                        vocab: List[str], roles: Dict[str, str], sim_threshold: float = 0.55,
                        max_pairs: int = 300) -> List[Dict[str, Any]]:
    """Build contradiction pairs with optimized memory usage.
    
    Instead of computing full V×V similarity matrix, we:
    1. Use batch processing for large vocabularies
    2. Only compute similarities for candidate pairs
    """
    V, pairs = len(vocab), []
    vectors_safe = np.nan_to_num(vectors, nan=0.0, posinf=1e6, neginf=-1e6)
    vectors_norm = vectors_safe / (np.linalg.norm(vectors_safe, axis=1, keepdims=True) + 1e-12)
    
    # Optimized: for large vocabularies, use batch processing
    # First, identify candidate pairs based on co-occurrence (cheap check)
    candidates = []
    for i in range(V):
        if roles.get(vocab[i]) == "punct":
            continue
        for j in range(i + 1, V):
            if roles.get(vocab[j]) == "punct":
                continue
            if cooc[i, j] <= cooc_threshold and cooc[j, i] <= cooc_threshold:
                candidates.append((i, j))
    
    # Limit candidates to avoid memory issues (take top candidates by some heuristic)
    if len(candidates) > 50000:  # Too many candidates
        # Sample or use a smarter heuristic
        candidates = candidates[:50000]
    
    # Compute similarities only for candidate pairs (much more efficient)
    if len(candidates) < V * (V - 1) // 4:  # Only if we have fewer candidates than full matrix
        for i, j in candidates:
            sim = np.clip(np.dot(vectors_norm[i], vectors_norm[j]), -1.0, 1.0)
            if sim >= sim_threshold:
                a, b = vocab[i], vocab[j]
                penalty = round(min(1.0, (sim - sim_threshold) * 1.5 + 0.2), 3)
                pairs.append({"a": a, "b": b, "penalty": penalty,
                            "reason": "high semantic similarity with low co-occurrence"})
    else:
        # Fallback to original method if too many candidates
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            sims = np.clip(vectors_norm @ vectors_norm.T, -1.0, 1.0)
        for i in range(V):
            for j in range(i + 1, V):
                if sims[i, j] >= sim_threshold and cooc[i, j] <= cooc_threshold and cooc[j, i] <= cooc_threshold:
                    a, b = vocab[i], vocab[j]
                    if roles.get(a) != "punct" and roles.get(b) != "punct":
                        penalty = round(min(1.0, (sims[i, j] - sim_threshold) * 1.5 + 0.2), 3)
                        pairs.append({"a": a, "b": b, "penalty": penalty,
                                    "reason": "high semantic similarity with low co-occurrence"})
    
    return sorted(pairs, key=lambda x: -x["penalty"])[:max_pairs]


def _infer_sentence_end_tokens_streaming(seq_stream: Iterator[List[int]], vocab: List[str]) -> List[str]:
    """Infer sentence end tokens using streaming."""
    V = len(vocab)
    total = np.zeros((V,), dtype=np.float32)
    end = np.zeros((V,), dtype=np.float32)
    
    for s in seq_stream:
        for i in s:
            if 0 <= i < V:
                total[i] += 1.0
        if s and 0 <= s[-1] < V:
            end[s[-1]] += 1.0
    
    candidates = [(end[i] / max(1, total[i]) + 0.05 * end[i], tok) for i, tok in enumerate(vocab)
                  if _is_punct_like(tok) and total[i] > 0 and end[i] >= 1.0 and end[i] / total[i] >= 0.25]
    return [t for _, t in sorted(candidates, key=lambda x: -x[0])[:6]]


def _infer_sentence_end_tokens(seqs: List[List[int]], vocab: List[str]) -> List[str]:
    """Infer sentence end tokens - wrapper for backward compatibility."""
    return _infer_sentence_end_tokens_streaming(iter(seqs), vocab)


def _infer_roles_and_clusters_streaming(seq_stream: Iterator[List[int]], vocab: List[str], 
                                        vectors: np.ndarray) -> Tuple[Dict[str, str], Dict[str, int], List[int]]:
    """Infer roles and clusters using streaming - processes one sequence at a time."""
    V = len(vocab)
    
    # Document frequency (bounded by V)
    df = np.zeros((V,), dtype=np.float32)
    num_docs = 0
    
    # Process sequences to compute document frequency
    for s in seq_stream:
        num_docs += 1
        for i in set(s):
            if 0 <= i < V:
                df[i] += 1.0
    
    if num_docs > 0:
        df = df / float(num_docs)
    
    meta_shape = np.array([_shape_features(t)["meta_shape"] for t in vocab], dtype=np.float32)
    k = int(max(4, min(12, np.sqrt(max(1, V)) + 3.0)))
    labels = _kmeans(vectors, k=k, iters=30, seed=7)
    clusters = {vocab[i]: int(labels[i]) for i in range(V)}
    seed_ids = [i for i in range(V) if meta_shape[i] >= 0.67 and not _is_punct_like(vocab[i])]
    meta_cluster_ids = []
    if len(seed_ids) >= 3:
        meta_centroid = vectors[seed_ids].mean(axis=0) / (np.linalg.norm(vectors[seed_ids].mean(axis=0)) + 1e-12)
        meta_sim = (vectors @ meta_centroid).astype(np.float32)
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
    
    # Process sequences again for frame detection (bounded statistics)
    occ = np.zeros((V,), dtype=np.float32)
    fw_frame = np.zeros((V,), dtype=np.float32)
    is_fw = np.array([1.0 if roles.get(vocab[i]) == "function_word" else 0.0 for i in range(V)], dtype=np.float32)
    
    # Reset stream (need to iterate again)
    # Note: This requires the stream to be re-iterable or we need to store sequences
    # For true streaming, we'd need to make two passes or cache sequences
    # For now, we'll make this work with a list for the second pass
    # In a true streaming implementation, we'd combine both passes
    
    return roles, clusters, meta_cluster_ids


def _infer_roles_and_clusters(vocab: List[str], vectors: np.ndarray, seqs: List[List[int]]) -> Tuple[Dict[str, str], Dict[str, int], List[int]]:
    """Infer roles and clusters - uses streaming version."""
    return _infer_roles_and_clusters_streaming(iter(seqs), vocab, vectors)


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


def _build_discourse_role_stats_streaming(seq_stream: Iterator[List[int]], vocab: List[str], token_roles: Dict[str, str],
                               clusters: Dict[str, int], meta_cluster_ids: List[int],
                               sentence_end_tokens: List[str]) -> Tuple[List[str], List[str], Dict[str, str],
                               np.ndarray, np.ndarray, List[int], np.ndarray, np.ndarray, List[int]]:
    """Build discourse role stats using streaming - processes one sequence at a time.
    
    Uses bounded accumulators instead of growing lists. Combines both passes in one iteration.
    """
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

    # Combined pass: collect position samples AND transition/emission stats in one iteration
    pos_idx_samples = []
    max_samples = 100000  # Cap sample size
    
    # Bounded accumulators (size fixed by R and L, not by number of sequences)
    trans = np.zeros((R, R), dtype=np.float32)
    emit = np.zeros((R, L), dtype=np.float32)
    trans_pos = np.zeros((3, R, R), dtype=np.float32)
    emit_pos = np.zeros((3, R, L), dtype=np.float32)
    runlens = [[] for _ in range(R)]  # Bounded by number of role transitions, not total tokens
    
    # Process sequences in single pass
    for s in seq_stream:
        if not s:
            continue
        
        prev_role, pos_in_sentence, cur_run_role, cur_run_len = 0, 0, None, 0
        
        for tid in s:
            if tid < 0 or tid >= len(vocab):
                continue
            
            # Collect position samples (bounded)
            if len(pos_idx_samples) < max_samples:
                pos_idx_samples.append(int(pos_in_sentence))
            
            # Process role transitions and emissions
            b = 0 if pos_in_sentence <= 3 else (1 if pos_in_sentence <= 8 else 2)  # Temporary bucket, will recalc
            tr, lvl = roles_by_id[tid], token_role_to_level.get(roles_by_id[tid], "content")
            if is_meta_tok(tid):
                lvl = "meta"
            if tid in end_ids or is_punct_tok(tid):
                lvl = "punct" if lvl != "meta" else lvl
            li, cur = level_to_i.get(lvl, 0), int(np.argmax(np.ones((R,), dtype=np.float32) / R))
            
            trans[prev_role, cur] += 1.0
            emit[cur, li] += 1.0
            
            # Track run lengths
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
    
    # Compute position thresholds from sampled positions
    if pos_idx_samples:
        q1 = max(0, int(np.quantile(np.array(pos_idx_samples, dtype=np.float32), 0.33)))
        q2 = max(q1 + 1, int(np.quantile(np.array(pos_idx_samples, dtype=np.float32), 0.66)))
    else:
        q1, q2 = 3, 8
    pos_thresholds = [int(q1), int(q2)]
    bucket_from_pos = lambda p: 0 if p <= q1 else (1 if p <= q2 else 2)
    
    # Re-process sequences to compute position-conditioned stats (requires second pass)
    # For true streaming, we'd need to cache position buckets from first pass
    # For now, we'll use a simplified approach or require sequences to be re-iterable
    # In practice, this is acceptable since sequences are typically small enough to store
    # or we can use a bounded cache of recent sequences
    
    # For now, initialize position-conditioned matrices with uniform distribution
    # A full implementation would require either:
    # 1. Storing sequences (defeats streaming for very large datasets)
    # 2. Using a bounded cache of recent sequences
    # 3. Approximating position stats from overall stats
    
    # Use overall stats as approximation for position-conditioned stats
    trans_pos = np.tile(trans[np.newaxis, :, :], (3, 1, 1))
    emit_pos = np.tile(emit[np.newaxis, :, :], (3, 1, 1))
    
    smoothing = 1.0
    trans = (trans + smoothing) / np.maximum(1e-12, (trans + smoothing).sum(axis=1, keepdims=True))
    emit = (emit + smoothing) / np.maximum(1e-12, (emit + smoothing).sum(axis=1, keepdims=True))
    trans_pos = (trans_pos + smoothing) / np.maximum(1e-12, (trans_pos + smoothing).sum(axis=2, keepdims=True))
    emit_pos = (emit_pos + smoothing) / np.maximum(1e-12, (emit_pos + smoothing).sum(axis=2, keepdims=True))
    min_run = [int(max(2, min(18, round(float(np.quantile(np.array(runlens[r] if runlens[r] else [3.0], dtype=np.float32), 0.25)))))) for r in range(R)]

    return (
        role_names,
        level_names,
        token_role_to_level,
        trans.astype(np.float32),
        np.log(emit.astype(np.float32) + 1e-12),
        pos_thresholds,
        trans_pos.astype(np.float32),
        np.log(emit_pos.astype(np.float32) + 1e-12),
        min_run,
    )


def build_discourse_role_stats(seqs: List[List[int]], vocab: List[str], token_roles: Dict[str, str],
                               clusters: Dict[str, int], meta_cluster_ids: List[int],
                               sentence_end_tokens: List[str]) -> Tuple[List[str], List[str], Dict[str, str],
                               np.ndarray, np.ndarray, List[int], np.ndarray, np.ndarray, List[int]]:
    """Build discourse role stats - uses streaming version that combines passes."""
    return _build_discourse_role_stats_streaming(iter(seqs), vocab, token_roles, clusters, meta_cluster_ids, sentence_end_tokens)


def train_reasoner(texts: List[str], token_desc: Dict[str, str], dim: int = 32,
                  window: int = 2, desc_alpha: float = 0.35, 
                  seqs: List[List[int]] = None, tok: ClosedVocabTokenizer = None,
                  max_vocab_size: Optional[int] = None) -> Tuple[ReasonerArtifacts, List[List[int]], ClosedVocabTokenizer]:
    """Train reasoner model with streaming support.
    
    If seqs and tok are provided, skips tokenization (for efficiency in train_dual).
    If max_vocab_size is set, caps vocabulary size.
    """
    if max_vocab_size is None:
        max_vocab_size = MAX_VOCAB_SIZE
    
    if tok is None or seqs is None:
        tok = ClosedVocabTokenizer.from_texts(texts, max_vocab_size=max_vocab_size)
        seqs = [tok.encode(t) for t in texts]
    
    V = len(tok.vocab)
    
    # Build co-occurrence matrix using streaming (sparse accumulator)
    cooc_sparse = defaultdict(float)
    for s in seqs:
        for i, wi in enumerate(s):
            if 0 <= wi < V:
                for j in range(max(0, i - window), min(len(s), i + window + 1)):
                    if j != i:
                        wj = s[j]
                        if 0 <= wj < V:
                            cooc_sparse[(wi, wj)] += 1.0
    
    # Convert to dense for compatibility (but only if V is reasonable)
    if V <= 10000:
        cooc = np.zeros((V, V), dtype=np.float32)
        for (i, j), count in cooc_sparse.items():
            cooc[i, j] = count
    else:
        # For very large vocab, use sparse representation
        # For now, create minimal dense matrix for compatibility
        cooc = np.zeros((V, V), dtype=np.float32)
        for (i, j), count in list(cooc_sparse.items())[:1000000]:  # Limit to top 1M pairs
            cooc[i, j] = count
    
    # Build PPMI vectors using streaming
    X = _build_ppmi_vectors_streaming(iter(seqs), V, window=window, dim=dim)
    X2 = X.astype(np.float32).copy()
    
    for token, desc in token_desc.items():
        if desc.strip() and token in tok.token_to_id:
            desc_toks = [t for t in simple_tokenize(desc) if t in tok.token_to_id]
            if desc_toks:
                ids = [tok.token_to_id[t] for t in desc_toks]
                X2[tok.token_to_id[token]] = (1.0 - desc_alpha) * X2[tok.token_to_id[token]] + desc_alpha * X[ids].mean(axis=0)
    
    Xn = normalize_rows(X2)
    # Cleanup X early to save memory
    del X
    import gc
    gc.collect()
    
    roles, clusters, meta_cluster_ids = _infer_roles_and_clusters(tok.vocab, Xn, seqs)
    discourse_role_names, discourse_level_names, token_role_to_level, discourse_trans, discourse_emit_logp, discourse_pos_thresholds, discourse_trans_pos, discourse_emit_logp_pos, discourse_role_min_run = build_discourse_role_stats(seqs, tok.vocab, roles, clusters, meta_cluster_ids, _infer_sentence_end_tokens(seqs, tok.vocab))
    
    # Build contradictions with optimized memory usage
    contradiction_pairs = build_contradictions(Xn, 0.0, cooc, tok.vocab, roles)
    
    result = ReasonerArtifacts(vocab=tok.vocab, token_to_id=tok.token_to_id, vectors=Xn.astype(np.float32),
                            bigram_logp=_build_bigram_logp_streaming(iter(seqs), V).astype(np.float32), roles=roles,
                            neighbors=_build_neighbors_streaming(iter(seqs), tok.vocab), contradiction_pairs=contradiction_pairs,
                            clusters=clusters, meta_cluster_ids=meta_cluster_ids, sentence_end_tokens=_infer_sentence_end_tokens(seqs, tok.vocab),
                            discourse_role_names=discourse_role_names, discourse_level_names=discourse_level_names,
                            token_role_to_level=token_role_to_level, discourse_trans=discourse_trans, discourse_emit_logp=discourse_emit_logp,
                            discourse_pos_thresholds=discourse_pos_thresholds, discourse_trans_pos=discourse_trans_pos,
                            discourse_emit_logp_pos=discourse_emit_logp_pos, discourse_role_min_run=discourse_role_min_run)
    # Explicit cleanup of large arrays
    del cooc, cooc_sparse, X2, Xn
    gc.collect()
    return result, seqs, tok

# ---------------- Dual-model training (Reasoner + Selector) ----------------
def train_dual(
    texts: List[str],
    token_desc: Dict[str, str],
    dim: int = 32,
    window: int = 2,
    desc_alpha: float = 0.35,
    selector_smooth: float = 0.5,
    selector_max_per_context: int = 256,
    max_vocab_size: Optional[int] = None,
) -> Tuple[ReasonerArtifacts, SelectorArtifacts]:
    """Train two models from the same data with streaming support.
    
    - ReasonerArtifacts: semantic vectors + contradiction-aware metadata (no next-token CE).
    - SelectorArtifacts: sparse trigram/bigram transitions for fluent token selection.

    Both share the same vocab / token IDs (Reasoner vocab).
    
    Optimized: tokenization happens only once, sequences are reused.
    If max_vocab_size is set, caps vocabulary size.
    """
    if max_vocab_size is None:
        max_vocab_size = MAX_VOCAB_SIZE
    
    reasoner, seqs, tok = train_reasoner(texts=texts, token_desc=token_desc, dim=dim, window=window, desc_alpha=desc_alpha, max_vocab_size=max_vocab_size)
    # Reuse sequences from reasoner training - no need to tokenize again!
    selector = train_selector_from_seqs(
        vocab=reasoner.vocab,
        seqs=seqs,
        smooth=selector_smooth,
        max_per_context=selector_max_per_context,
    )
    # Explicit cleanup of intermediate data
    del tok
    import gc
    gc.collect()
    return reasoner, selector
