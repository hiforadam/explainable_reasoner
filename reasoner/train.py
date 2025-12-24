"""Training module with Learning Tracker - מעקב אחרי צעדי הלמידה."""
import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Iterator
from collections import defaultdict

from .tokenizer import ClosedVocabTokenizer, simple_tokenize
from .utils import normalize_rows
from .model import ReasonerArtifacts
from .selector import SelectorArtifacts, train_selector_from_seqs
from .learning_tracker import LearningTracker
from .config import ModelConfig, default_config

_PUNCT_RE = re.compile(r"[^\w\s]+\Z", re.UNICODE)


def _is_punct_like(token: str) -> bool:
    return bool(_PUNCT_RE.fullmatch(token))


def _shape_features(token: str, config: ModelConfig = default_config) -> Dict[str, float]:
    has_alpha = any(c.isalpha() for c in token)
    has_digit = any(c.isdigit() for c in token)
    has_underscore = "_" in token
    non_alnum = sum(1 for c in token if not c.isalnum())
    non_alnum_ratio = non_alnum / max(1, len(token))
    meta_shape = (float(has_underscore) + float(has_digit) + float(has_alpha and has_digit) + 
                  (config.meta_shape_non_alnum_weight if non_alnum_ratio >= config.non_alnum_threshold else 0.0)) / config.meta_shape_divisor
    return {
        "has_alpha": float(has_alpha), "has_digit": float(has_digit),
        "has_underscore": float(has_underscore), "non_alnum_ratio": float(non_alnum_ratio),
        "meta_shape": float(meta_shape), "len": float(len(token)), "is_punct": float(_is_punct_like(token)),
    }


def _kmeans(X: np.ndarray, k: int, iters: Optional[int] = None, seed: Optional[int] = None, config: ModelConfig = default_config) -> np.ndarray:
    if iters is None:
        iters = config.kmeans_iters
    if seed is None:
        seed = config.kmeans_seed
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


def _relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation function."""
    return np.maximum(0, x)


def _build_neural_network_vectors(seq_stream: Iterator[List[int]], V: int,
                                   window: int = 2, dim: int = 32,
                                   num_iterations: Optional[int] = None,
                                   learning_rate: Optional[float] = None,
                                   hidden_dim: Optional[int] = None,
                                   seed: Optional[int] = None,
                                   config: ModelConfig = default_config) -> Tuple[np.ndarray, LearningTracker]:
    """בונה וקטורים סמנטיים עם רשת נוירונים אמיתית ומעקב אחרי כל נוירון."""
    
    if num_iterations is None:
        num_iterations = config.num_iterations
    if learning_rate is None:
        learning_rate = config.learning_rate
    if hidden_dim is None:
        hidden_dim = config.hidden_dim
    
    if seed is not None:
        np.random.seed(seed)
    
    # שלב 1: מחשב co-occurrence matrix
    cooc_sparse = defaultdict(float)
    total_count = 0.0
    
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
        return np.zeros((V, dim), dtype=np.float32), LearningTracker()
    
    # שלב 2: מחשב PPMI matrix (כ-target)
    row_sums = defaultdict(float)
    col_sums = defaultdict(float)
    for (i, j), count in cooc_sparse.items():
        row_sums[i] += count
        col_sums[j] += count
    
    if V > 10000:
        top_pairs = sorted(cooc_sparse.items(), key=lambda x: -x[1])[:min(100000, len(cooc_sparse))]
        C = np.zeros((V, V), dtype=np.float32)
        for (i, j), count in top_pairs:
            C[i, j] = count
    else:
        C = np.zeros((V, V), dtype=np.float32)
        for (i, j), count in cooc_sparse.items():
            C[i, j] = count
    
    total = float(total_count) + 1e-12
    P = C / total
    Pr = np.array([row_sums.get(i, 0.0) / total for i in range(V)], dtype=np.float32).reshape(-1, 1)
    Pc = np.array([col_sums.get(j, 0.0) / total for j in range(V)], dtype=np.float32).reshape(1, -1)
    
    denominator = np.maximum(Pr * Pc, 1e-12)
    with np.errstate(divide='ignore', invalid='ignore'):
        PPMI_target = np.maximum(np.log((P + 1e-12) / denominator), 0.0)
    
    # שלב 3: רשת נוירונים אמיתית
    tracker = LearningTracker(window_size=config.learning_tracker_window)
    
    # אתחול רשת נוירונים:
    # Embedding: V x dim
    # Layer 1: dim x hidden_dim
    # Layer 2: hidden_dim x dim
    # Output: V x dim
    
    # Embedding weights (V x dim)
    embedding = np.random.randn(V, dim).astype(np.float32) * 0.1
    embedding = normalize_rows(embedding)
    
    # Layer 1 weights (dim x hidden_dim)
    W1 = np.random.randn(dim, hidden_dim).astype(np.float32) * 0.1
    b1 = np.zeros((hidden_dim,), dtype=np.float32)
    
    # Layer 2 weights (hidden_dim x dim)
    W2 = np.random.randn(hidden_dim, dim).astype(np.float32) * 0.1
    b2 = np.zeros((dim,), dtype=np.float32)
    
    base_lr = learning_rate
    
    # אימון איטרטיבי עם מעקב אחרי כל נוירון
    for iteration in range(num_iterations):
        # Forward pass
        # x = embedding (V x dim)
        h1 = _relu(embedding @ W1 + b1)  # V x hidden_dim
        output = h1 @ W2 + b2  # V x dim
        output = normalize_rows(output)
        
        # מחשב target: וקטורים שמשחזרים את PPMI
        # משתמש ב-PPMI כדי ליצור target vectors
        target_vectors = normalize_rows(PPMI_target @ output)  # V x dim
        
        # Loss: כמה ה-output רחוק מה-target
        loss = np.mean((output - target_vectors) ** 2)
        
        # Backward pass - מחשב gradients
        # Gradient של output
        d_output = 2 * (output - target_vectors) / V  # V x dim
        
        # Gradient של Layer 2
        d_h1 = d_output @ W2.T  # V x hidden_dim
        d_h1 = d_h1 * (h1 > 0).astype(np.float32)  # ReLU derivative
        
        d_W2 = h1.T @ d_output  # hidden_dim x dim
        d_b2 = np.sum(d_output, axis=0)  # dim
        
        # Gradient של Layer 1
        d_embedding = d_h1 @ W1.T  # V x dim
        
        d_W1 = embedding.T @ d_h1  # dim x hidden_dim
        d_b1 = np.sum(d_h1, axis=0)  # hidden_dim
        
        # Gradient של Embedding
        d_embedding_final = d_embedding  # V x dim
        
        # עדכון עם adaptive learning rate לכל נוירון
        # עוקב אחרי כל weight בנפרד
        
        # עדכון Embedding - כל נוירון בנפרד
        for i in range(V):
            for j in range(dim):
                neuron_id = f"embedding_{i}_{j}"
                weight_before = embedding[i, j]
                gradient = d_embedding_final[i, j]
                
                # מקבל learning rate מותאם לפי הטריגרים של הנוירון
                adaptive_lr = tracker.get_adaptive_learning_rate(neuron_id, base_lr)
                
                # מעדכן את ה-weight
                embedding[i, j] = embedding[i, j] - adaptive_lr * gradient
                
                # עוקב אחרי הצעד
                tracker.track_neuron_step(neuron_id, iteration, weight_before, gradient, embedding[i, j], loss)
        
        # עדכון Layer 1 - כל נוירון בנפרד
        for i in range(dim):
            for j in range(hidden_dim):
                neuron_id = f"layer1_{i}_{j}"
                weight_before = W1[i, j]
                gradient = d_W1[i, j]
                
                adaptive_lr = tracker.get_adaptive_learning_rate(neuron_id, base_lr)
                W1[i, j] = W1[i, j] - adaptive_lr * gradient
                tracker.track_neuron_step(neuron_id, iteration, weight_before, gradient, W1[i, j], loss)
        
        # עדכון bias Layer 1
        for j in range(hidden_dim):
            neuron_id = f"bias1_{j}"
            weight_before = b1[j]
            gradient = d_b1[j]
            
            adaptive_lr = tracker.get_adaptive_learning_rate(neuron_id, base_lr)
            b1[j] = b1[j] - adaptive_lr * gradient
            tracker.track_neuron_step(neuron_id, iteration, weight_before, gradient, b1[j], loss)
        
        # עדכון Layer 2 - כל נוירון בנפרד
        for i in range(hidden_dim):
            for j in range(dim):
                neuron_id = f"layer2_{i}_{j}"
                weight_before = W2[i, j]
                gradient = d_W2[i, j]
                
                adaptive_lr = tracker.get_adaptive_learning_rate(neuron_id, base_lr)
                W2[i, j] = W2[i, j] - adaptive_lr * gradient
                tracker.track_neuron_step(neuron_id, iteration, weight_before, gradient, W2[i, j], loss)
        
        # עדכון bias Layer 2
        for j in range(dim):
            neuron_id = f"bias2_{j}"
            weight_before = b2[j]
            gradient = d_b2[j]
            
            adaptive_lr = tracker.get_adaptive_learning_rate(neuron_id, base_lr)
            b2[j] = b2[j] - adaptive_lr * gradient
            tracker.track_neuron_step(neuron_id, iteration, weight_before, gradient, b2[j], loss)
        
        # נרמול Embedding אחרי עדכון
        embedding = normalize_rows(embedding)
    
    # מחזיר את ה-embedding כוקטורים סמנטיים
    return embedding.astype(np.float32), tracker


def _build_ppmi_vectors_with_learning_tracker(seq_stream: Iterator[List[int]], V: int, 
                                               window: int = 2, dim: int = 32,
                                               num_iterations: Optional[int] = None, 
                                               learning_rate: Optional[float] = None,
                                               seed: Optional[int] = None,
                                               config: ModelConfig = default_config) -> Tuple[np.ndarray, LearningTracker]:
    """בונה וקטורים סמנטיים עם רשת נוירונים אמיתית."""
    return _build_neural_network_vectors(seq_stream, V, window, dim, num_iterations, learning_rate, hidden_dim=config.hidden_dim, seed=seed, config=config)


def _build_ppmi_vectors_streaming(seq_stream: Iterator[List[int]], V: int, window: int = 2, dim: int = 32, config: ModelConfig = default_config) -> np.ndarray:
    """Build PPMI vectors using streaming - processes one sequence at a time."""
    vectors, _ = _build_ppmi_vectors_with_learning_tracker(seq_stream, V, window, dim, num_iterations=config.num_iterations, config=config)
    return vectors


def _build_bigram_logp_sparse(seq_stream: Iterator[List[int]], V: int, alpha: float = 0.5, top_k: int = 150) -> Dict[str, List[Tuple[int, float]]]:
    """Build sparse bigram log probabilities - only top-K transitions per token."""
    import math
    
    bigram_counts: Dict[int, Dict[int, float]] = {}
    num_seqs = 0
    
    for s in seq_stream:
        num_seqs += 1
        if len(s) < 2:
            continue
        for a, b in zip(s[:-1], s[1:]):
            if 0 <= a < V and 0 <= b < V:
                bigram_counts.setdefault(a, {})
                bigram_counts[a][b] = bigram_counts[a].get(b, 0.0) + 1.0
    
    adaptive_alpha = max(alpha, min(2.0, 1.0 + (100.0 / max(num_seqs, 1))))
    
    bigram_logp: Dict[str, List[Tuple[int, float]]] = {}
    for p1, counts in bigram_counts.items():
        items = list(counts.items())
        if not items:
            continue
        total = sum(v for _, v in items) + adaptive_alpha * len(items)
        out = []
        for k, v in items:
            p = (v + adaptive_alpha) / max(1e-12, total)
            out.append((int(k), float(math.log(p + 1e-12))))
        out.sort(key=lambda x: x[1], reverse=True)
        bigram_logp[str(int(p1))] = out[:top_k]
    
    return bigram_logp


def _build_neighbors_streaming(seq_stream: Iterator[List[int]], vocab: List[str], topn: int = 5) -> Dict[str, Dict[str, List[str]]]:
    """Build neighbors using streaming."""
    V = len(vocab)
    before_counts = defaultdict(float)
    after_counts = defaultdict(float)
    
    for s in seq_stream:
        for i in range(1, len(s)):
            if 0 <= s[i] < V and 0 <= s[i-1] < V:
                before_counts[(s[i], s[i-1])] += 1.0
        for i in range(len(s) - 1):
            if 0 <= s[i] < V and 0 <= s[i+1] < V:
                after_counts[(s[i], s[i+1])] += 1.0
    
    out = {}
    for i, tok in enumerate(vocab):
        before_items = [(j, before_counts.get((i, j), 0.0)) for j in range(V)]
        before_items.sort(key=lambda x: -x[1])
        before_neighbors = [vocab[j] for j, count in before_items[:topn] if count > 0]
        
        after_items = [(j, after_counts.get((i, j), 0.0)) for j in range(V)]
        after_items.sort(key=lambda x: -x[1])
        after_neighbors = [vocab[j] for j, count in after_items[:topn] if count > 0]
        
        out[tok] = {"before": before_neighbors[:topn], "after": after_neighbors[:topn]}
    
    del before_counts, after_counts
    import gc
    gc.collect()
    
    return out


def _build_contradictions(vectors: np.ndarray, vocab: List[str], roles: Dict[str, str],
                          sim_threshold: Optional[float] = None, max_pairs: Optional[int] = None, 
                          seed: Optional[int] = None, config: ModelConfig = default_config) -> List[Dict[str, Any]]:
    """Build contradiction pairs."""
    if sim_threshold is None:
        sim_threshold = config.similarity_threshold
    if max_pairs is None:
        max_pairs = config.contradiction_max_pairs
    
    V, pairs = len(vocab), []
    if V == 0:
        return pairs
    
    vectors_safe = np.nan_to_num(vectors, nan=0.0, posinf=1e6, neginf=-1e6)
    vectors_norm = vectors_safe / (np.linalg.norm(vectors_safe, axis=1, keepdims=True) + 1e-12)
    
    if V > 5000:
        import random
        if seed is not None:
            random.seed(seed)
        candidates = []
        non_punct_indices = [i for i in range(V) if roles.get(vocab[i]) != "punct"]
        if len(non_punct_indices) < 2:
            return pairs
        
        num_samples = min(10000, len(non_punct_indices) * (len(non_punct_indices) - 1) // 2)
        for _ in range(num_samples):
            i, j = random.sample(non_punct_indices, 2)
            if i < j:
                candidates.append((i, j))
        
        for i, j in candidates:
            sim = float(np.clip(np.dot(vectors_norm[i], vectors_norm[j]), -1.0, 1.0))
            if sim >= sim_threshold:
                penalty = round(min(1.0, (sim - sim_threshold) * 1.5 + 0.2), 3)
                pairs.append({
                    "a": vocab[i],
                    "b": vocab[j],
                    "penalty": penalty,
                    "reason": "high semantic similarity"
                })
    else:
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            sims = np.clip(vectors_norm @ vectors_norm.T, -1.0, 1.0)
        
        for i in range(V):
            for j in range(i + 1, V):
                if sims[i, j] >= sim_threshold and roles.get(vocab[i]) != "punct" and roles.get(vocab[j]) != "punct":
                    penalty = round(min(1.0, (sims[i, j] - sim_threshold) * 1.5 + 0.2), 3)
                    pairs.append({
                        "a": vocab[i],
                        "b": vocab[j],
                        "penalty": penalty,
                        "reason": "high semantic similarity"
                    })
    
    return sorted(pairs, key=lambda x: -x["penalty"])[:max_pairs]


def _infer_sentence_end_tokens_streaming(seq_stream: Iterator[List[int]], vocab: List[str], config: ModelConfig = default_config) -> List[str]:
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
    
    candidates = [(end[i] / max(1, total[i]) + config.sentence_end_bonus * end[i], tok) for i, tok in enumerate(vocab)
                  if _is_punct_like(tok) and total[i] > 0 and end[i] >= config.sentence_end_min_count and end[i] / total[i] >= config.sentence_end_ratio_threshold]
    return [t for _, t in sorted(candidates, key=lambda x: -x[0])[:config.max_sentence_end_tokens]]


def _infer_roles_and_clusters(seqs: List[List[int]], vocab: List[str], vectors: np.ndarray, config: ModelConfig = default_config) -> Tuple[Dict[str, str], Dict[str, int], List[int]]:
    """Infer roles and clusters."""
    V = len(vocab)
    
    df = np.zeros((V,), dtype=np.float32)
    num_docs = 0
    for s in seqs:
        num_docs += 1
        for i in set(s):
            if 0 <= i < V:
                df[i] += 1.0
    
    if num_docs > 0:
        df = df / float(num_docs)
    
    meta_shape = np.array([_shape_features(t, config=config)["meta_shape"] for t in vocab], dtype=np.float32)
    k = int(max(config.kmeans_min_clusters, min(config.kmeans_max_clusters, np.sqrt(max(1, V)) + config.kmeans_sqrt_factor)))
    labels = _kmeans(vectors, k=k, iters=config.kmeans_iters, seed=config.kmeans_seed, config=config)
    clusters = {vocab[i]: int(labels[i]) for i in range(V)}
    
    seed_ids = [i for i in range(V) if meta_shape[i] >= config.meta_shape_threshold and not _is_punct_like(vocab[i])]
    meta_cluster_ids = []
    if len(seed_ids) >= 3:
        meta_centroid = vectors[seed_ids].mean(axis=0) / (np.linalg.norm(vectors[seed_ids].mean(axis=0)) + 1e-12)
        meta_sim = (vectors @ meta_centroid).astype(np.float32)
        cluster_scores = {ci: config.cluster_score_meta_weight * meta_sim[np.where(labels == ci)[0]].mean() + config.cluster_score_df_weight * df[np.where(labels == ci)[0]].mean()
                         for ci in range(k) if np.any(labels == ci)}
        if cluster_scores:
            cutoff = float(np.quantile(list(cluster_scores.values()), config.cluster_cutoff_quantile))
            meta_cluster_ids = [int(ci) for ci, sc in sorted(cluster_scores.items(), key=lambda x: -x[1])
                              if sc >= cutoff][:config.max_meta_clusters]
    
    roles = {}
    for i, tok in enumerate(vocab):
        if _is_punct_like(tok):
            roles[tok] = "punct"
        elif meta_cluster_ids and int(labels[i]) in meta_cluster_ids:
            roles[tok] = "dataset_meta"
        elif df[i] >= config.df_threshold and tok.isalpha() and len(tok) <= config.max_token_length_for_function:
            roles[tok] = "function_word"
        else:
            roles[tok] = "content_word"
    
    return roles, clusters, meta_cluster_ids


def _build_discourse_stats(seqs: List[List[int]], vocab: List[str], token_roles: Dict[str, str],
                          clusters: Dict[str, int], meta_cluster_ids: List[int],
                          sentence_end_tokens: List[str], config: ModelConfig = default_config) -> Tuple[List[str], List[str], Dict[str, str],
                          np.ndarray, np.ndarray, List[int], np.ndarray, np.ndarray, List[int]]:
    """Build discourse role statistics."""
    role_names = [f"role_{i}" for i in range(config.num_discourse_roles)]
    level_names = config.discourse_levels.copy()
    token_role_to_level = {
        "punct": "punct",
        "dataset_meta": "meta",
        "function_word": "function",
        "adverb_like": "function",
        "content_word": "content"
    }
    
    R, L = len(role_names), len(level_names)
    roles_by_id = [token_roles.get(t, "content_word") for t in vocab]
    token_to_id = {t: i for i, t in enumerate(vocab)}
    end_ids = set(int(token_to_id[t]) for t in sentence_end_tokens if t in token_to_id)
    
    trans = np.ones((R, R), dtype=np.float32) / R
    emit = np.ones((R, L), dtype=np.float32) / L
    trans_pos = np.tile(trans[np.newaxis, :, :], (3, 1, 1))
    emit_pos = np.tile(emit[np.newaxis, :, :], (3, 1, 1))
    
    pos_thresholds = config.discourse_pos_thresholds.copy()
    min_run = config.discourse_role_min_run.copy()
    if len(min_run) < R:
        min_run.extend([2] * (R - len(min_run)))
    min_run = min_run[:R]
    
    return (role_names, level_names, token_role_to_level, trans, np.log(emit + 1e-12),
            pos_thresholds, trans_pos, np.log(emit_pos + 1e-12), min_run)


def train_reasoner(texts: List[str], token_desc: Dict[str, str], dim: int = 16,
                  window: int = 2, desc_alpha: float = 0.35,
                  max_vocab_size: Optional[int] = None, bigram_top_k: Optional[int] = None,
                  seed: Optional[int] = None, config: ModelConfig = default_config) -> Tuple[ReasonerArtifacts, List[List[int]], ClosedVocabTokenizer]:
    """Train reasoner model with Learning Tracker."""
    if max_vocab_size is None:
        max_vocab_size = config.max_vocab_size
    if bigram_top_k is None:
        bigram_top_k = config.bigram_top_k
    
    if seed is not None:
        np.random.seed(seed)
    
    tok = ClosedVocabTokenizer.from_texts(texts, max_vocab_size=max_vocab_size)
    seqs = [tok.encode(t) for t in texts if t.strip()]
    
    if not seqs:
        raise ValueError("No valid sequences after tokenization")
    
    V = len(tok.vocab)
    
    # Build PPMI vectors with Learning Tracker
    X, learning_tracker = _build_ppmi_vectors_with_learning_tracker(
        iter(seqs), V, window=window, dim=dim, num_iterations=config.num_iterations, 
        learning_rate=config.learning_rate, seed=seed, config=config
    )
    X2 = X.astype(np.float32).copy()
    
    # Incorporate token descriptions
    for token, desc in token_desc.items():
        if desc.strip() and token in tok.token_to_id:
            desc_toks = [t for t in simple_tokenize(desc) if t in tok.token_to_id]
            if desc_toks:
                ids = [tok.token_to_id[t] for t in desc_toks]
                X2[tok.token_to_id[token]] = (1.0 - desc_alpha) * X2[tok.token_to_id[token]] + desc_alpha * X[ids].mean(axis=0)
    
    Xn = normalize_rows(X2)
    del X, X2
    import gc
    gc.collect()
    
    # Infer roles and clusters
    roles, clusters, meta_cluster_ids = _infer_roles_and_clusters(seqs, tok.vocab, Xn, config=config)
    sentence_end_tokens = _infer_sentence_end_tokens_streaming(iter(seqs), tok.vocab, config=config)
    
    # Build discourse stats
    discourse_role_names, discourse_level_names, token_role_to_level, discourse_trans, discourse_emit_logp, \
    discourse_pos_thresholds, discourse_trans_pos, discourse_emit_logp_pos, discourse_role_min_run = \
        _build_discourse_stats(seqs, tok.vocab, roles, clusters, meta_cluster_ids, sentence_end_tokens, config=config)
    
    # Build other artifacts using streaming
    neighbors = _build_neighbors_streaming(iter(seqs), tok.vocab)
    contradiction_pairs = _build_contradictions(Xn, tok.vocab, roles, seed=seed, config=config)
    bigram_logp = _build_bigram_logp_sparse(iter(seqs), V, top_k=bigram_top_k)
    
    artifacts = ReasonerArtifacts(
        vocab=tok.vocab,
        token_to_id=tok.token_to_id,
        vectors=Xn.astype(np.float32),
        bigram_logp=bigram_logp,
        roles=roles,
        neighbors=neighbors,
        contradiction_pairs=contradiction_pairs,
        clusters=clusters,
        meta_cluster_ids=meta_cluster_ids,
        sentence_end_tokens=sentence_end_tokens,
        discourse_role_names=discourse_role_names,
        discourse_level_names=discourse_level_names,
        token_role_to_level=token_role_to_level,
        discourse_trans=discourse_trans,
        discourse_emit_logp=discourse_emit_logp,
        discourse_pos_thresholds=discourse_pos_thresholds,
        discourse_trans_pos=discourse_trans_pos,
        discourse_emit_logp_pos=discourse_emit_logp_pos,
        discourse_role_min_run=discourse_role_min_run,
    )
    
    # שמירת learning signals ב-artifacts
    artifacts.learning_signals = learning_tracker.get_learning_signals()
    
    return artifacts, seqs, tok


def train_dual(texts: List[str], token_desc: Dict[str, str], dim: int = 16,
              window: int = 2, desc_alpha: float = 0.35,
              selector_smooth: Optional[float] = None, selector_max_per_context: Optional[int] = None,
              max_vocab_size: Optional[int] = None, bigram_top_k: Optional[int] = None,
              seed: Optional[int] = None, config: ModelConfig = default_config) -> Tuple[ReasonerArtifacts, SelectorArtifacts]:
    """Train dual model (reasoner + selector)."""
    if selector_smooth is None:
        selector_smooth = config.selector_smooth
    if selector_max_per_context is None:
        selector_max_per_context = config.selector_max_per_context
    
    reasoner, seqs, tok = train_reasoner(
        texts=texts, token_desc=token_desc, dim=dim, window=window,
        desc_alpha=desc_alpha, max_vocab_size=max_vocab_size, bigram_top_k=bigram_top_k, seed=seed, config=config
    )
    
    selector = train_selector_from_seqs(
        vocab=reasoner.vocab,
        seqs=seqs,
        smooth=selector_smooth,
        max_per_context=selector_max_per_context,
    )
    
    return reasoner, selector
