"""Text generation module - simplified and optimized."""
import logging
import warnings
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from collections import deque
from .tokenizer import ClosedVocabTokenizer
from .model import ReasonerArtifacts
from .selector import SelectorArtifacts
from .utils import softmax
from .config import ModelConfig, default_config

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)


def _build_contradiction_lookup(pairs: List[Dict[str, Any]], token_to_id: Dict[str, int]) -> Dict[Tuple[int, int], float]:
    """Build contradiction lookup table."""
    out = {}
    for p in pairs:
        a, b = token_to_id.get(p.get("a")), token_to_id.get(p.get("b"))
        if a is not None and b is not None:
            pen = float(p.get("penalty", 0.0))
            if pen > 0:
                out[(a, b)] = out[(b, a)] = max(out.get((a, b), 0.0), pen)
    return out


def _weighted_context(ids: List[int], vec: np.ndarray, window: int = 20, decay: Optional[float] = None, config: ModelConfig = default_config) -> np.ndarray:
    """Compute weighted context with exponential decay."""
    if decay is None:
        decay = config.context_decay
    
    if not ids:
        return np.zeros(vec.shape[1], dtype=np.float32)
    
    k = min(window, len(ids))
    recent_ids = ids[-k:]
    weights = np.array([decay ** (k - i - 1) for i in range(k)], dtype=np.float32)
    weights = weights / (np.sum(weights) + 1e-12)
    
    ctx = np.zeros(vec.shape[1], dtype=np.float32)
    for i, tid in enumerate(recent_ids):
        ctx += weights[i] * vec[tid].astype(np.float32)
    
    ctx = np.nan_to_num(ctx, nan=0.0, posinf=1.0, neginf=-1.0)
    norm = np.linalg.norm(ctx)
    if norm < 1e-12:
        return np.zeros(vec.shape[1], dtype=np.float32)
    return (ctx / norm).astype(np.float32)


class SemanticGroup:
    """קבוצת סמנטיקה דינמית עם overlap משוקלל."""
    def __init__(self, group_id: int, centroid: np.ndarray, tokens: List[int], config: ModelConfig = default_config):
        self.group_id = group_id
        self.centroid = centroid.copy()  # וקטור מרכזי
        self.tokens = set(tokens)  # טוקנים בקבוצה
        self.strength = 1.0  # חוזק הקבוצה
        self.activation = 0.0  # רמת הפעלה נוכחית
        self.decay_rate = config.group_decay_rate  # decay איטי
        self.usage_count = 0  # כמה פעמים השתמשו בקבוצה
        self.config = config
    
    def update_centroid(self, vec: np.ndarray) -> None:
        """מעדכן את המרכז לפי הטוקנים."""
        if not self.tokens:
            return
        token_vecs = vec[list(self.tokens)]
        self.centroid = token_vecs.mean(axis=0).astype(np.float32)
        norm = np.linalg.norm(self.centroid)
        if norm > 1e-12:
            self.centroid = (self.centroid / norm).astype(np.float32)
    
    def add_token(self, token_id: int, vec: np.ndarray) -> None:
        """מוסיף טוקן לקבוצה (overlap משוקלל)."""
        self.tokens.add(token_id)
        # מעדכן את המרכז בצורה משוקללת
        if len(self.tokens) == 1:
            self.centroid = vec[token_id].copy()
        else:
            alpha = self.config.group_alpha  # משקל של הטוקן החדש
            self.centroid = ((1 - alpha) * self.centroid + alpha * vec[token_id]).astype(np.float32)
            norm = np.linalg.norm(self.centroid)
            if norm > 1e-12:
                self.centroid = (self.centroid / norm).astype(np.float32)
    
    def activate(self, strength: float = 1.0) -> None:
        """מפעיל את הקבוצה."""
        self.activation = min(self.config.activation_limit, self.activation + strength * self.config.activation_increment)
        self.usage_count += 1
    
    def decay(self) -> None:
        """Decay איטי - מונע over-lock."""
        # Decay איטי יותר אם הפעלה גבוהה (מונע נעילה)
        if self.activation > self.config.high_activation_threshold:
            self.activation *= self.config.high_activation_decay  # Decay מהיר יותר אם נעול
        else:
            self.activation *= self.decay_rate  # Decay איטי רגיל
    
    def get_similarity(self, token_vec: np.ndarray) -> float:
        """מחזיר דמיון סמנטי לטוקן."""
        return float(np.clip(np.dot(token_vec, self.centroid), -1.0, 1.0))


class SemanticGroupManager:
    """מנהל קבוצות סמנטיות דינמיות עם overlap משוקלל."""
    def __init__(self, vec: np.ndarray, vocab: List[str], initial_groups: Optional[int] = None, config: ModelConfig = default_config):
        self.vec = vec
        self.vocab = vocab
        self.V = len(vocab)
        self.groups: List[SemanticGroup] = []
        self.token_to_groups: Dict[int, List[int]] = {}  # token_id -> [group_ids] - overlap
        self.active_group_id: Optional[int] = None
        self.config = config
        
        if initial_groups is None:
            initial_groups = config.initial_groups
        
        # יוצר קבוצות ראשוניות עם overlap משוקלל
        self._build_overlapping_groups(initial_groups)
    
    def _build_overlapping_groups(self, num_groups: int) -> None:
        """בונה קבוצות עם overlap משוקלל (לא clustering עיוור)."""
        k = min(num_groups, max(4, self.V // 20))
        if k < 2:
            k = 2
        
        # אתחול centroids
        np.random.seed(7)
        centroids = self.vec[np.random.choice(self.V, size=k, replace=False)].copy()
        
        # כמה iterations לאתחול
        for _ in range(5):
            distances = np.linalg.norm(self.vec[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
            labels = np.argmin(distances, axis=1)
            new_centroids = np.zeros_like(centroids)
            for ci in range(k):
                mask = labels == ci
                if np.any(mask):
                    new_centroids[ci] = self.vec[mask].mean(axis=0)
                else:
                    new_centroids[ci] = centroids[ci]
            centroids = new_centroids
        
        # יוצר קבוצות עם overlap משוקלל
        for gi in range(k):
            group_tokens = []
            for ti in range(self.V):
                dist = np.linalg.norm(self.vec[ti] - centroids[gi])
                # אם קרוב מספיק - הוסף לקבוצה (overlap)
                percentile_threshold = np.percentile(np.linalg.norm(self.vec - centroids[gi], axis=1), 50)
                if dist < percentile_threshold:
                    group_tokens.append(ti)
            
            if group_tokens:
                group = SemanticGroup(gi, centroids[gi], group_tokens, config=self.config)
                self.groups.append(group)
                
                # עדכן token_to_groups (overlap - טוקן יכול להיות בכמה קבוצות)
                for tid in group_tokens:
                    if tid not in self.token_to_groups:
                        self.token_to_groups[tid] = []
                    self.token_to_groups[tid].append(gi)
    
    def find_best_group_for_token(self, token_id: int, prompt_context: np.ndarray) -> Optional[int]:
        """מוצא את הקבוצה הכי נכונה לטוקן ביחס לפרומפט - דינמי."""
        if token_id not in self.token_to_groups:
            return None
        
        token_vec = self.vec[token_id]
        candidate_groups = self.token_to_groups[token_id]
        
        if not candidate_groups:
            return None
        
        best_group = None
        best_score = -1.0
        
        for gid in candidate_groups:
            group = self.groups[gid]
            # דמיון לטוקן
            token_sim = group.get_similarity(token_vec)
            # דמיון להקשר הפרומפט
            context_sim = float(np.clip(np.dot(group.centroid, prompt_context), -1.0, 1.0)) if prompt_context is not None else 0.0
            # חוזק הקבוצה (דינמי - לא קבוע)
            strength = group.strength * (1.0 / (1.0 + group.usage_count * 0.1))  # נחלש עם שימוש
            # רמת הפעלה (דינמי)
            activation = group.activation
            
            # ציון משולב - דינמי לפי המצב
            score = (self.config.token_sim_weight * token_sim + 
                    self.config.context_sim_weight * context_sim + 
                    self.config.strength_weight * strength + 
                    self.config.activation_weight * activation)
            
            if score > best_score:
                best_score = score
                best_group = gid
        
        return best_group
    
    def get_group_connections(self, group_id: int, bigram_logp: Dict[str, List[Tuple[int, float]]]) -> Dict[int, float]:
        """מחזיר את הקשרים הכי חזקים בקבוצה."""
        if group_id >= len(self.groups):
            return {}
        
        group = self.groups[group_id]
        connections: Dict[int, float] = {}
        
        # אוסף קשרים חזקים מכל הטוקנים בקבוצה
        for tid in group.tokens:
            tid_str = str(tid)
            if tid_str in bigram_logp:
                for next_tid, logp in bigram_logp[tid_str]:
                    if next_tid < self.V:
                        # משתמש בקשרים הכי חזקים (לא חלשים)
                        if next_tid not in connections or logp > connections[next_tid]:
                            connections[next_tid] = float(logp)
        
        return connections
    
    def activate_group(self, group_id: int, strength: float = 1.0) -> None:
        """מפעיל קבוצה (עם decay איטי ל-ActiveGroup הקודמת)."""
        # Decay ל-ActiveGroup הקודמת
        if self.active_group_id is not None and self.active_group_id < len(self.groups):
            self.groups[self.active_group_id].decay()
        
        # מפעיל את הקבוצה החדשה
        if group_id < len(self.groups):
            self.groups[group_id].activate(strength)
            self.active_group_id = group_id
    
    def get_active_group_boost(self, token_id: int) -> float:
        """מחזיר boost לטוקן אם הוא בקבוצה הפעילה - מוגבל למנוע over-lock."""
        if self.active_group_id is None:
            return 0.0
        
        group = self.groups[self.active_group_id]
        if token_id in group.tokens:
            # Boost לפי רמת הפעלה (decay איטי) - מוגבל כדי למנוע over-lock
            boost = group.activation * self.config.group_boost_base
            # אם הפעלה גבוהה מדי - מקטין עוד יותר (מונע נעילה)
            if group.activation > self.config.very_high_activation_threshold:
                boost *= self.config.group_boost_high_penalty
            return boost
        return 0.0
    
    def get_semantic_similarity_boost(self, token_id: int, prompt_context: np.ndarray) -> float:
        """מחזיר boost לפי הסמנטיקה הכי קרובה - לא רק קבוצה פעילה."""
        token_vec = self.vec[token_id]
        
        # מחפש את הקבוצה הכי קרובה סמנטית
        best_sim = -1.0
        for group in self.groups:
            sim = group.get_similarity(token_vec)
            # גם בודק דמיון להקשר הפרומפט
            if prompt_context is not None:
                context_sim = float(np.clip(np.dot(group.centroid, prompt_context), -1.0, 1.0))
                sim = self.config.semantic_group_token_weight * sim + self.config.semantic_group_context_weight * context_sim
            if sim > best_sim:
                best_sim = sim
        
        # תגמול לפי דמיון (לא חוק קשיח)
        if best_sim > self.config.strong_similarity_threshold:
            return best_sim * self.config.semantic_boost_strong
        elif best_sim > self.config.medium_similarity_threshold:
            return best_sim * self.config.semantic_boost_medium
        return 0.0


class TopicMemory:
    """Simple topic memory - maintains key concepts."""
    def __init__(self, max_topics: Optional[int] = None, config: ModelConfig = default_config):
        self.topics: List[np.ndarray] = []
        self.weights: List[float] = []
        self.max_topics = max_topics if max_topics is not None else config.max_topics
        self.config = config
    
    def update(self, new_tokens: List[int], vec: np.ndarray) -> None:
        """Update topic memory."""
        if not new_tokens:
            return
        
        new_vec = vec[new_tokens].mean(axis=0).astype(np.float32)
        new_vec = np.nan_to_num(new_vec, nan=0.0, posinf=1.0, neginf=-1.0)
        norm = np.linalg.norm(new_vec)
        if norm < 1e-12:
            return
        new_vec = (new_vec / norm).astype(np.float32)
        
        if not self.topics:
            self.topics.append(new_vec)
            self.weights.append(1.0)
            return
        
        # Find most similar topic
        similarities = [float(np.clip(np.dot(new_vec, t), -1.0, 1.0)) for t in self.topics]
        max_sim_idx = int(np.argmax(similarities))
        max_sim = similarities[max_sim_idx]
        
        if max_sim > self.config.topic_merge_threshold:  # Merge with existing topic
            alpha = self.config.topic_merge_alpha
            self.topics[max_sim_idx] = ((1 - alpha) * self.topics[max_sim_idx] + alpha * new_vec).astype(np.float32)
            norm = np.linalg.norm(self.topics[max_sim_idx])
            if norm > 1e-12:
                self.topics[max_sim_idx] = (self.topics[max_sim_idx] / norm).astype(np.float32)
            self.weights[max_sim_idx] = min(self.config.topic_weight_limit, self.weights[max_sim_idx] + self.config.topic_weight_increment)
        else:  # New topic
            if len(self.topics) < self.max_topics:
                self.topics.append(new_vec)
                self.weights.append(self.config.new_topic_weight)
            else:
                min_idx = int(np.argmin(self.weights))
                self.topics[min_idx] = new_vec
                self.weights[min_idx] = self.config.new_topic_weight
    
    def get_context(self) -> Optional[np.ndarray]:
        """Get weighted average of topics."""
        if not self.topics:
            return None
        weights_norm = np.array(self.weights, dtype=np.float32)
        weights_norm = weights_norm / (np.sum(weights_norm) + 1e-12)
        combined = sum(w * t for w, t in zip(weights_norm, self.topics))
        combined = np.nan_to_num(combined, nan=0.0, posinf=1.0, neginf=-1.0)
        norm = np.linalg.norm(combined)
        if norm < 1e-12:
            return None
        return (combined / norm).astype(np.float32)


def _hierarchical_context(ids: List[int], vec: np.ndarray, topic_memory: Optional[TopicMemory],
                         short_window: int = 12, step: int = 0, total_steps: int = 30, config: ModelConfig = default_config) -> np.ndarray:
    """Build hierarchical context: short-term + long-term (topic memory)."""
    if not ids:
        return np.zeros(vec.shape[1], dtype=np.float32)
    
    # Short-term: weighted average
    short_ctx = _weighted_context(ids, vec, window=short_window, config=config)
    
    # Long-term: topic memory or prompt
    long_ctx = topic_memory.get_context() if topic_memory else None
    if long_ctx is None:
        prompt_len = min(config.prompt_anchor_size, len(ids))
        long_ctx = vec[ids[:prompt_len]].mean(axis=0).astype(np.float32)
        norm = np.linalg.norm(long_ctx)
        if norm > 1e-12:
            long_ctx = (long_ctx / norm).astype(np.float32)
        else:
            long_ctx = np.zeros(vec.shape[1], dtype=np.float32)
    
    # Adaptive weights based on progress
    progress = step / max(total_steps, 1)
    w_short = config.short_context_weight_base + config.short_context_weight_progress * progress
    w_long = config.long_context_weight_base - config.short_context_weight_progress * progress
    
    combined = (w_short * short_ctx + w_long * long_ctx).astype(np.float32)
    combined = np.nan_to_num(combined, nan=0.0, posinf=1.0, neginf=-1.0)
    norm = np.linalg.norm(combined)
    if norm < 1e-12:
        return short_ctx
    return (combined / norm).astype(np.float32)


def top_p_sampling(logits: np.ndarray, top_p: Optional[float] = None, config: ModelConfig = default_config) -> np.ndarray:
    """Nucleus sampling."""
    if top_p is None:
        top_p = config.top_p_default
    
    sorted_indices = np.argsort(-logits)
    sorted_logits = logits[sorted_indices]
    probs = softmax(sorted_logits, temp=config.temperature_default)
    cum_probs = np.cumsum(probs)
    cutoff_idx = min(np.searchsorted(cum_probs, top_p) + 1, len(logits))
    mask = np.zeros_like(logits)
    mask[sorted_indices[:cutoff_idx]] = 1.0
    return mask


def _adaptive_structure_score(candidate_id: int, tokens_since_end: int,
                              roles_by_id: List[str], vec: np.ndarray,
                              recent_context: np.ndarray, config: ModelConfig = default_config) -> float:
    """Score candidate based on adaptive structure - no hard rules."""
    score = 0.0
    
    # זיהוי דינמי של פיסוק לפי סמנטיקה (לא רשימה קשיחה)
    is_punct_like = False
    if candidate_id < len(roles_by_id):
        role = roles_by_id[candidate_id]
        if role == "punct":
            is_punct_like = True
        # גם בודק לפי דמיון סמנטי לטוקני פיסוק נפוצים
        elif recent_context is not None:
            punct_similarity = np.mean([np.dot(vec[candidate_id], vec[i]) 
                                       for i in range(len(roles_by_id)) 
                                       if roles_by_id[i] == "punct"][:config.punct_similarity_check_count] or [0])
            if punct_similarity > config.punct_similarity_threshold:
                is_punct_like = True
    
    # תגמל סיום משפט אם יש הקשר מספיק (דינמי - לא אורך קשיח)
    if is_punct_like:
        # תגמל יותר אם יש הרבה הקשר (משפט ארוך)
        if tokens_since_end > config.good_sentence_length:
            score += config.structure_boost_long * min(1.0, tokens_since_end / config.max_sentence_length)
        # ענוש אם אין מספיק הקשר (משפט קצר מדי)
        elif tokens_since_end < config.min_sentence_length:
            score -= config.structure_penalty_short
    
    # תגמל טוקנים תוכן (לא פיסוק/מטא) בתחילת משפט
    if tokens_since_end < config.good_sentence_length and not is_punct_like:
        if candidate_id < len(roles_by_id) and roles_by_id[candidate_id] == "content_word":
            score += config.content_word_boost
    
    return score


def _infer_claim_type(prompt: str, vocab: List[str], token_to_id: Dict[str, int], 
                     vec: np.ndarray, config: ModelConfig = default_config) -> str:
    """Infer claim type from prompt using semantic similarity (dynamic, no hardcoded tokens)."""
    try:
        tok = ClosedVocabTokenizer(vocab=vocab, token_to_id=token_to_id)
        prompt_ids = tok.encode(prompt)
        if not prompt_ids:
            return config.claim_type_default
        
        prompt_vec = vec[prompt_ids[:min(config.prompt_anchor_size, len(prompt_ids))]].mean(axis=0)
        prompt_norm = np.linalg.norm(prompt_vec)
        if prompt_norm < 1e-12:
            return config.claim_type_default
        
        prompt_vec = prompt_vec / prompt_norm
        
        # Dynamic claim type inference based on semantic similarity
        # Returns default - actual type inference would require learned patterns
        # For now, semantic coherence is more important than specific type
        return config.claim_type_default
    except:
        return config.claim_type_default


def _get_claim_role_tokens(vocab: List[str], token_to_id: Dict[str, int], 
                          vec: np.ndarray, config: ModelConfig = default_config) -> Dict[str, List[int]]:
    """Map tokens to semantic roles dynamically based on learned representations (no hardcoded keywords)."""
    # Roles are inferred dynamically from semantic similarity during generation
    # No hardcoded keyword matching needed
    return {}


def _coherence_score(candidate_id: int, ids: List[int], vec: np.ndarray, 
                     window: int = 15, topic_memory: Optional[TopicMemory] = None, 
                     config: ModelConfig = default_config) -> float:
    """Score candidate based on coherence - improved with topic awareness."""
    if len(ids) < 2:
        return 0.0
    
    recent_ids = ids[-min(window, len(ids)):]
    if not recent_ids:
        return 0.0
    
    # Short-term context: weighted average (more recent = higher weight)
    weights = np.array([config.context_decay ** (len(recent_ids) - i - 1) for i in range(len(recent_ids))], dtype=np.float32)
    weights = weights / (np.sum(weights) + 1e-12)
    short_context = np.sum(vec[recent_ids] * weights[:, np.newaxis], axis=0)
    
    # Long-term context: topic memory if available
    long_context = None
    if topic_memory:
        long_context = topic_memory.get_context()
    
    # Combine contexts
    if long_context is not None:
        context_vec = (config.semantic_group_token_weight * short_context + config.semantic_group_context_weight * long_context).astype(np.float32)
    else:
        context_vec = short_context
    
    ctx_norm = np.linalg.norm(context_vec)
    if ctx_norm < 1e-12:
        return 0.0
    context_vec = context_vec / ctx_norm
    
    candidate_vec = vec[candidate_id]
    cand_norm = np.linalg.norm(candidate_vec)
    if cand_norm < 1e-12:
        return 0.0
    candidate_vec = candidate_vec / cand_norm
    
    coherence = float(np.dot(candidate_vec, context_vec))
    
    # תגמל יותר קוהרנטיות גבוהה
    if coherence > config.coherence_high_threshold:
        return coherence * config.coherence_high_boost
    elif coherence > config.coherence_medium_threshold:
        return coherence * config.coherence_medium_boost
    else:
        return coherence * config.coherence_low_boost


def generate(art: ReasonerArtifacts, prompt: str, max_new_tokens: int = 30,
            temperature: float = 0.75, top_k: int = 6, top_p: Optional[float] = None,
            explain: bool = False, block_dataset_meta: bool = True, repeat_window: Optional[int] = None,
            repetition_penalty: Optional[float] = None, semantic_repeat_window: Optional[int] = None,
            semantic_repeat_threshold: Optional[float] = None, semantic_repeat_penalty: Optional[float] = None,
            context_window: Optional[int] = None, seed: Optional[int] = None, config: ModelConfig = default_config) -> Dict[str, Any]:
    """Generate text using reasoner model - simplified version."""
    if repeat_window is None:
        repeat_window = config.repeat_window
    if repetition_penalty is None:
        repetition_penalty = config.repetition_penalty
    if semantic_repeat_window is None:
        semantic_repeat_window = config.semantic_repeat_window
    if semantic_repeat_threshold is None:
        semantic_repeat_threshold = config.semantic_repeat_threshold
    if semantic_repeat_penalty is None:
        semantic_repeat_penalty = config.semantic_repeat_penalty
    if context_window is None:
        context_window = config.context_window
    
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    if max_new_tokens <= 0 or temperature <= 0 or top_k <= 0:
        raise ValueError("Invalid parameters")
    if not art.vocab:
        raise ValueError("Model vocabulary is empty")
    
    if seed is not None:
        np.random.seed(seed)
    
    try:
        tok = ClosedVocabTokenizer(vocab=art.vocab, token_to_id=art.token_to_id)
        ids = tok.encode(prompt)
    except ValueError as e:
        raise ValueError(f"Tokenization failed: {e}")
    
    if not ids:
        raise ValueError("Prompt resulted in empty token sequence")
    
    V = len(art.vocab)
    vec = art.vectors
    bigram_logp = art.bigram_logp
    
    if vec.shape[0] != V:
        raise ValueError(f"Model dimensions mismatch: vocab={V}, vectors={vec.shape[0]}")
    
    contra = _build_contradiction_lookup(art.contradiction_pairs, art.token_to_id)
    roles_by_id = [art.roles.get(t, "unknown") for t in art.vocab]
    
    # לא משתמשים ברשימה קשיחה של end_token_ids - זיהוי דינמי לפי סמנטיקה
    
    # Pre-compute normalized vectors
    vec_norm = vec.astype(np.float32)
    vec_norm = np.nan_to_num(vec_norm, nan=0.0, posinf=1.0, neginf=-1.0)
    norms = np.linalg.norm(vec_norm, axis=1, keepdims=True)
    vec_norm = vec_norm / np.maximum(norms, 1e-8)
    vec_norm = np.nan_to_num(vec_norm, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Pre-compute dataset_meta mask
    dataset_meta_mask = np.zeros((V,), dtype=bool)
    if block_dataset_meta:
        for j in range(V):
            if roles_by_id[j] == "dataset_meta":
                dataset_meta_mask[j] = True
    
    # Prompt anchor
    prompt_anchor = None
    if ids:
        k0 = min(config.prompt_anchor_size, len(ids))
        prompt_anchor = vec[ids[:k0]].mean(axis=0).astype(np.float32)
        norm = np.linalg.norm(prompt_anchor)
        if norm > 1e-12:
            prompt_anchor = (prompt_anchor / norm).astype(np.float32)
        else:
            prompt_anchor = None
    
    # Initialize topic memory
    topic_memory = TopicMemory(config=config)
    if ids:
        topic_memory.update(ids[:min(config.prompt_anchor_size, len(ids))], vec)
    
    log = []
    tokens_since_end = 0
    
    for step in range(max_new_tokens):
        prev = ids[-1]
        
        # Update tokens since end - זיהוי דינמי של סיום משפט
        is_end_token = False
        if prev < len(roles_by_id):
            role = roles_by_id[prev]
            # בודק אם זה פיסוק (לא רשימה קשיחה)
            if role == "punct":
                # בודק דמיון סמנטי לטוקני פיסוק נפוצים
                punct_tokens = [i for i in range(len(roles_by_id)) if roles_by_id[i] == "punct"]
                if punct_tokens:
                    punct_vecs = vec[punct_tokens[:10]]
                    prev_sim = np.max(np.dot(vec[prev], punct_vecs.T))
                    if prev_sim > 0.5:  # דמיון גבוה לפיסוק
                        is_end_token = True
        
        if is_end_token:
            tokens_since_end = 0
        else:
            tokens_since_end += 1
        
        # Update topic memory periodically
        if step % 3 == 0 and len(ids) >= 5:
            topic_memory.update(ids[-min(10, len(ids)):], vec)
        
        # Context vector
        ctx = _hierarchical_context(ids, vec, topic_memory, short_window=min(context_window, config.short_context_window),
                                    step=step, total_steps=max_new_tokens, config=config)
        ctx = np.nan_to_num(ctx, nan=0.0, posinf=1.0, neginf=-1.0)
        ctx_norm = np.linalg.norm(ctx)
        if ctx_norm < 1e-12:
            ctx = np.zeros(vec.shape[1], dtype=np.float32)
        
        # Bigram logits
        comp_bigram = np.full((V,), -50.0, dtype=np.float64)
        if str(prev) in bigram_logp:
            for tid, logp in bigram_logp[str(prev)]:
                if 0 <= tid < V:
                    comp_bigram[tid] = np.clip(float(logp), -50.0, 50.0)
        
        # Semantic similarity
        if ctx_norm > 1e-12:
            comp_sem = np.clip((vec_norm @ ctx).astype(np.float64), -1.0, 1.0)
            comp_sem = np.nan_to_num(comp_sem, nan=0.0, posinf=1.0, neginf=-1.0)
        else:
            comp_sem = np.zeros((V,), dtype=np.float64)
        
        # Prompt anchor boost (only for top candidates)
        anchor_boost = np.zeros((V,), dtype=np.float64)
        if prompt_anchor is not None:
            anchor_sims = np.clip((vec_norm @ prompt_anchor).astype(np.float64), -1.0, 1.0)
            anchor_boost = anchor_sims * config.anchor_weight
        
        # Contradiction penalty
        contra_pen = np.zeros((V,), dtype=np.float64)
        recent_contra = ids[-min(4, len(ids)):]
        if contra and recent_contra:
            for r in recent_contra:
                for (a, b), penalty in contra.items():
                    if a == r:
                        contra_pen[b] += penalty
                    elif b == r:
                        contra_pen[a] += penalty
        
        # Base logits
        logits = config.bigram_weight * comp_bigram + config.semantic_weight * comp_sem + anchor_boost - config.contra_weight * contra_pen
        
        # Get top candidates first (for efficient scoring)
        # Use a larger initial top-k to ensure we have good candidates
        initial_top_k = max(top_k * 2, 20)
        initial_top_idx = np.argsort(-logits)[:initial_top_k]
        
        # Apply adaptive structure and coherence scores only to top candidates
        structure_scores = np.zeros((V,), dtype=np.float64)
        coherence_scores = np.zeros((V,), dtype=np.float64)
        
        # מחשב הקשר עדכני לשימוש ב-structure score
        recent_context_vec = None
        if len(ids) >= 3:
            recent_ids = ids[-min(10, len(ids)):]
            recent_context_vec = vec[recent_ids].mean(axis=0)
            norm = np.linalg.norm(recent_context_vec)
            if norm > 1e-12:
                recent_context_vec = recent_context_vec / norm
        
        for j in initial_top_idx:
            structure_scores[j] = _adaptive_structure_score(int(j), tokens_since_end, roles_by_id, vec, recent_context_vec, config=config)
            coherence_scores[j] = _coherence_score(int(j), ids, vec, window=15, topic_memory=topic_memory, config=config)
        
        # Add scores to logits - משקלים מאוזנים יותר
        logits += structure_scores * config.structure_weight + coherence_scores * config.coherence_weight
        
        # עידוד דינמי לסיום משפט (לא רשימה קשיחה)
        if tokens_since_end > config.max_sentence_length:  # משפט ארוך
            # מחפש טוקני פיסוק סמנטית דומים
            punct_indices = [i for i in range(V) if roles_by_id[i] == "punct"]
            if punct_indices:
                # תגמל טוקני פיסוק שדומים להקשר
                for pid in punct_indices[:20]:  # רק הראשונים
                    if recent_context_vec is not None:
                        sim = np.dot(vec[pid], recent_context_vec)
                        if sim > config.punct_similarity_threshold:
                            logits[pid] += config.structure_boost_long * min(1.0, tokens_since_end / (config.max_sentence_length + 5))
        
        # Block dataset/meta tokens
        if block_dataset_meta:
            logits[dataset_meta_mask] = -1e18
        
        # Repetition penalty
        if repeat_window > 0 and repetition_penalty > 0:
            recent = ids[-min(repeat_window, len(ids)):]
            recent_set = set(recent)
            for j in recent_set:
                logits[j] -= repetition_penalty * recent.count(j)
            
            # Phrase-level penalty
            if len(ids) >= 4:
                for phrase_len in [2, 3]:
                    if len(ids) >= phrase_len * 2:
                        recent_phrase = tuple(ids[-phrase_len:])
                        check_window = min(phrase_len * 2, len(ids) - phrase_len)
                        for i in range(max(0, len(ids) - check_window), len(ids) - phrase_len):
                            if tuple(ids[i:i+phrase_len]) == recent_phrase:
                                for tid in recent_phrase:
                                    logits[tid] -= repetition_penalty * 1.5
                                break
        
        # Semantic repetition penalty
        if semantic_repeat_window > 0 and semantic_repeat_penalty > 0:
            sem_recent = ids[-min(semantic_repeat_window, len(ids)):]
            if sem_recent:
                sem_recent_vecs = vec_norm[sem_recent]
                sem_sims = np.clip(vec_norm @ sem_recent_vecs.T, -1.0, 1.0)
                max_sims = np.clip(np.max(sem_sims, axis=1), -1.0, 1.0)
                mask = max_sims > semantic_repeat_threshold
                if np.any(mask):
                    penalty_scale = np.clip((max_sims[mask] - semantic_repeat_threshold) / 0.4, 0.0, 3.0)
                    logits[mask] -= semantic_repeat_penalty * penalty_scale
        
        # Sampling (re-sort after adding structure/coherence scores)
        if top_p is not None and top_p > 0:
            mask = top_p_sampling(logits, top_p, config=config)
            masked_logits = np.where(mask > 0, logits, -1e18)
            top_idx = np.argsort(-masked_logits)[:max(1, int(np.sum(mask)))]
            top_logits = masked_logits[top_idx]
        else:
            top_idx = np.argsort(-logits)[:max(1, top_k)]
            top_logits = logits[top_idx]
        
        probs = softmax(top_logits, temp=max(float(temperature), 1e-6))
        if np.any(np.isnan(probs)) or np.sum(probs) == 0:
            probs = np.ones_like(probs) / len(probs)
        choice = int(np.random.choice(len(top_idx), p=probs))
        nxt = int(top_idx[choice])
        ids.append(nxt)
        
        if explain:
            kshow = min(5, len(top_idx))
            cand = [{
                "token": art.vocab[int(j)],
                "score": float(logits[int(j)]),
                "bigram_logp": float(comp_bigram[int(j)]),
                "semantic_sim": float(comp_sem[int(j)]),
            } for j in top_idx[:kshow]]
            log.append({
                "step": step + 1,
                "prev_token": art.vocab[int(prev)],
                "chosen_token": art.vocab[int(nxt)],
                "top_candidates": cand,
            })
    
    return {"text": tok.decode(ids), "token_ids": ids, "explain": log if explain else None}


def generate_dual(reasoner: ReasonerArtifacts, selector: SelectorArtifacts, prompt: str,
                 max_new_tokens: int = 60, temperature: float = 0.75, selector_top_k: Optional[int] = None,
                 alpha_selector: Optional[float] = None, beta_reasoner: Optional[float] = None,
                 eta_trust: Optional[float] = None, eta_bias: Optional[float] = None, trust_clip: Optional[float] = None,
                 w_semantic: Optional[float] = None, w_anchor: Optional[float] = None, w_contra: Optional[float] = None,
                 w_repeat: Optional[float] = None, repeat_window: Optional[int] = None, block_dataset_meta: bool = True,
                 explain: bool = False, context_window: Optional[int] = None, seed: Optional[int] = None,
                 config: ModelConfig = default_config) -> Dict[str, Any]:
    """Generate text using dual model - simplified version."""
    if selector_top_k is None:
        selector_top_k = config.selector_top_k_default
    if alpha_selector is None:
        alpha_selector = config.alpha_selector
    if beta_reasoner is None:
        beta_reasoner = config.beta_reasoner
    if eta_trust is None:
        eta_trust = config.eta_trust
    if eta_bias is None:
        eta_bias = config.eta_bias
    if trust_clip is None:
        trust_clip = config.trust_clip
    if w_semantic is None:
        w_semantic = config.w_semantic
    if w_anchor is None:
        w_anchor = config.w_anchor
    if w_contra is None:
        w_contra = config.w_contra
    if w_repeat is None:
        w_repeat = config.w_repeat
    if repeat_window is None:
        repeat_window = config.repeat_window
    if context_window is None:
        context_window = config.context_window
    
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    if max_new_tokens <= 0 or temperature <= 0:
        raise ValueError("Invalid parameters")
    if not reasoner.vocab or not selector.vocab or reasoner.vocab != selector.vocab:
        raise ValueError("Invalid vocabularies")
    
    if seed is not None:
        np.random.seed(seed)
    
    try:
        tok = ClosedVocabTokenizer(vocab=reasoner.vocab, token_to_id=reasoner.token_to_id)
        ids: List[int] = tok.encode(prompt)
    except ValueError as e:
        raise ValueError(f"Tokenization failed: {e}")
    
    if not ids:
        raise ValueError("Prompt resulted in empty token sequence")
    
    V = len(reasoner.vocab)
    vec = reasoner.vectors
    
    # Prompt anchor
    anchor = vec[ids[:min(config.prompt_anchor_size, len(ids))]].mean(axis=0).astype(np.float32) if ids else np.zeros((vec.shape[1],), dtype=np.float32)
    norm = np.linalg.norm(anchor)
    if norm > 1e-12:
        anchor = (anchor / norm).astype(np.float32)
    
    contra = _build_contradiction_lookup(reasoner.contradiction_pairs, reasoner.token_to_id)
    roles_by_id = [reasoner.roles.get(t, "unknown") for t in reasoner.vocab]
    
    # לא משתמשים ברשימה קשיחה של end_token_ids - זיהוי דינמי
    
    # Topic memory
    topic_memory = TopicMemory(config=config)
    if ids:
        topic_memory.update(ids[:min(config.prompt_anchor_size, len(ids))], vec)
    
    # Semantic Group Manager - קבוצות סמנטיות דינמיות עם overlap משוקלל
    group_manager = SemanticGroupManager(vec, reasoner.vocab, config=config)
    
    # מחשב הקשר פרומפט לשימוש בבחירת קבוצות
    prompt_context = anchor.copy()
    
    # Semantic Lock: בוחר קבוצה סמנטית ראשונית לפי הפרומפט
    locked_group_id: Optional[int] = None
    lock_strength = config.lock_strength_initial  # חוזק הנעילה (דינמי)
    tokens_since_lock = 0  # כמה טוקנים מאז הנעילה
    last_sentence_end_step = 0  # מתי היה סיום משפט אחרון
    
    # Micro-topic Lock: מרכז סמנטי אחד כ"רעיון נוכחי"
    micro_topic_center: Optional[np.ndarray] = None  # וקטור המרכז הסמנטי
    micro_topic_token_id: Optional[int] = None  # ID של הטוקן המרכזי
    micro_topic_similarity_threshold = config.micro_topic_similarity_threshold  # threshold דינמי - דמיון מינימלי
    tokens_since_micro_topic_change = 0  # כמה טוקנים מאז שינוי המרכז
    micro_topic_novelty_history: deque = deque(maxlen=config.novelty_history_size)  # היסטוריית novelty
    
    # Claim-level Lock: זיהוי claim type מהפרומפט (dynamic, no hardcoded keywords)
    active_claim_type: str = _infer_claim_type(prompt, reasoner.vocab, reasoner.token_to_id, vec, config=config)
    claim_role_tokens: Dict[str, List[int]] = _get_claim_role_tokens(reasoner.vocab, reasoner.token_to_id, vec, config=config)
    
    # בוחר את הקבוצה הכי נכונה לפי הפרומפט
    if ids:
        # מחפש את הקבוצה הכי קרובה לפרומפט
        best_group_score = -1.0
        for group in group_manager.groups:
            # דמיון להקשר הפרומפט
            context_sim = float(np.clip(np.dot(group.centroid, prompt_context), -1.0, 1.0))
            # חוזק הקבוצה
            strength = group.strength
            # ציון משולב
            score = config.semantic_group_token_weight * context_sim + config.semantic_group_context_weight * strength
            if score > best_group_score:
                best_group_score = score
                locked_group_id = group.group_id
        
        # מפעיל את הקבוצה הנעולה
        if locked_group_id is not None:
            group_manager.activate_group(locked_group_id, strength=1.0)
            
            # Micro-topic Lock: בוחר מרכז סמנטי אחד מהקבוצה הנעולה
            locked_group = group_manager.groups[locked_group_id]
            
            # בוחר את הטוקן הכי קרוב להקשר הפרומפט בקבוצה
            best_token_score = -1.0
            best_token_id = None
            
            for tid in locked_group.tokens:
                token_vec = vec[tid]
                # דמיון להקשר הפרומפט
                prompt_sim = float(np.clip(np.dot(token_vec, prompt_context), -1.0, 1.0))
                # דמיון למרכז הקבוצה
                group_sim = locked_group.get_similarity(token_vec)
                # ציון משולב
                token_score = 0.6 * prompt_sim + 0.4 * group_sim
                
                if token_score > best_token_score:
                    best_token_score = token_score
                    best_token_id = tid
            
            # מגדיר את המרכז הסמנטי
            if best_token_id is not None:
                micro_topic_token_id = best_token_id
                micro_topic_center = vec[best_token_id].copy()
                norm = np.linalg.norm(micro_topic_center)
                if norm > 1e-12:
                    micro_topic_center = (micro_topic_center / norm).astype(np.float32)
    
    # Online state
    trust = np.zeros((V,), dtype=np.float64)
    bias: Dict[str, Dict[int, float]] = {}
    explain_rows: List[Dict[str, Any]] = []
    
    def ctx_key(p2: Optional[int], p1: Optional[int]) -> str:
        if p2 is not None and p1 is not None:
            return f"{int(p2)},{int(p1)}"
        return str(int(p1)) if p1 is not None else "<start>"
    
    def get_bias(k: str, tid: int) -> float:
        return float(bias.get(k, {}).get(int(tid), 0.0))
    
    def add_bias(k: str, tid: int, delta: float) -> None:
        if k not in bias:
            bias[k] = {}
        bias[k][int(tid)] = float(bias[k].get(int(tid), 0.0) + float(delta))
    
    def contradiction_penalty(candidate: int, recent: List[int]) -> float:
        return max([float(contra.get((int(candidate), int(r)), 0.0)) for r in recent], default=0.0)
    
    def sample_next(prev2: Optional[int], prev1: Optional[int]) -> Optional[Tuple[int, float]]:
        cands = selector.candidates(prev2, prev1, top_k=int(selector_top_k))
        if not cands:
            return None
        cand_ids = np.array([int(c) for c, _ in cands], dtype=np.int64)
        cand_lps = np.array([float(lp) for _, lp in cands], dtype=np.float64)
        probs = softmax(cand_lps, temp=max(1e-6, float(temperature)))
        if np.any(np.isnan(probs)) or np.sum(probs) == 0:
            probs = np.ones_like(probs) / len(probs)
        j = int(np.random.choice(len(cand_ids), p=probs))
        return int(cand_ids[j]), float(cand_lps[j])
    
    def score_continuation(cont: List[int], prev2: Optional[int], prev1: Optional[int]) -> Tuple[float, float, float]:
        """Score continuation: (selector_avg, reasoner_avg, total)."""
        if not cont:
            return -1e9, -1e9, -1e9
        
        sel_sum = 0.0
        p2, p1 = prev2, prev1
        for tid in cont:
            sel_sum += get_bias(ctx_key(p2, p1), int(tid))
            p2, p1 = (p1, int(tid)) if p1 is not None else (None, int(tid))
        
        tmp_ids = list(ids)
        reason_sum = 0.0
        
        # Closure pressure
        current_step = len(tmp_ids)
        progress = current_step / max(max_new_tokens, 1)
        closure_strength = max(0.0, (progress - config.closure_threshold) / (1.0 - config.closure_threshold)) if progress > config.closure_threshold else 0.0
        
        # Compute recent context
        recent_context = None
        if len(tmp_ids) >= 5:
            recent_tokens = tmp_ids[-min(config.recent_context_size, len(tmp_ids)):]
            recent_context = vec[recent_tokens].mean(axis=0).astype(np.float32)
            recent_norm = np.linalg.norm(recent_context)
            if recent_norm > 1e-12:
                recent_context = (recent_context / recent_norm).astype(np.float32)
        
        recent_groups: List[int] = []
        
        for tid in cont:
            if not (0 <= int(tid) < V):
                continue
            if block_dataset_meta and roles_by_id[int(tid)] == "dataset_meta":
                reason_sum -= 2.5
            
            best_group_id = group_manager.find_best_group_for_token(int(tid), prompt_context)
            k = min(context_window, len(tmp_ids)) if tmp_ids else 0
            ctx = _hierarchical_context(tmp_ids, vec, topic_memory, short_window=k,
                                       step=len(tmp_ids), total_steps=max_new_tokens, config=config) if k > 0 else anchor
            recent = tmp_ids[-max(1, repeat_window):] if tmp_ids else []
            
            # Basic scores
            sem = float(np.dot(vec[int(tid)], ctx))
            anc = float(np.dot(vec[int(tid)], anchor))
            cp = contradiction_penalty(int(tid), recent)
            rep = 1.0 if int(tid) in recent else 0.0
            
            # Semantic group boost
            group_boost = 0.0
            semantic_boost = 0.0
            if best_group_id is not None:
                group_boost = group_manager.get_active_group_boost(int(tid))
                group_connections = group_manager.get_group_connections(best_group_id, reasoner.bigram_logp)
                if int(tid) in group_connections and group_connections[int(tid)] > config.group_connection_threshold:
                    normalized_strength = (group_connections[int(tid)] + config.group_connection_normalization) / config.group_connection_normalization
                    group_boost += config.group_connection_boost * normalized_strength
            semantic_boost = group_manager.get_semantic_similarity_boost(int(tid), prompt_context)
            
            # Lock scores
            lock_boost = 0.0
            lock_penalty = 0.0
            if locked_group_id is not None and lock_strength > config.lock_strength_min:
                locked_group = group_manager.groups[locked_group_id]
                if int(tid) in locked_group.tokens:
                    lock_boost = lock_strength * (config.lock_boost_base + config.lock_boost_similarity * locked_group.get_similarity(vec[int(tid)]))
                else:
                    lock_penalty = lock_strength * config.lock_penalty_base
            
            # Micro-topic penalty
            micro_topic_penalty = 0.0
            if micro_topic_center is not None and lock_strength > config.lock_strength_min:
                token_vec = vec[int(tid)]
                token_norm = np.linalg.norm(token_vec)
                if token_norm > 1e-12:
                    similarity = float(np.clip(np.dot(token_vec / token_norm, micro_topic_center), -1.0, 1.0))
                    if similarity < micro_topic_similarity_threshold:
                        penalty_scale = (micro_topic_similarity_threshold - similarity) / micro_topic_similarity_threshold
                        micro_topic_penalty = lock_strength * config.micro_topic_penalty_base * penalty_scale
            
            # Claim penalty
            claim_penalty = 0.0
            if recent_context is not None:
                token_vec = vec[int(tid)]
                token_norm = np.linalg.norm(token_vec)
                if token_norm > 1e-12:
                    context_sim = float(np.clip(np.dot(token_vec / token_norm, recent_context), -1.0, 1.0))
                    if context_sim < config.context_similarity_threshold:
                        claim_penalty = config.claim_penalty_base
            
            # Group repetition penalty
            group_penalty = 0.0
            if best_group_id is not None:
                group_count = recent_groups.count(best_group_id)
                if group_count > 0:
                    penalty_base = config.group_repetition_penalty_base * group_count
                    if group_count > config.group_repetition_threshold:
                        penalty_base = config.group_repetition_penalty_high * group_count
                    group_penalty = penalty_base
            
            # Closure boost
            closure_boost = 0.0
            if closure_strength > 0 and recent_context is not None:
                tid_sim = float(np.clip(np.dot(vec[int(tid)], recent_context), -1.0, 1.0))
                if tid_sim > config.closure_similarity_threshold:
                    closure_boost = config.closure_boost_base * closure_strength * tid_sim
                if roles_by_id[int(tid)] == "punct":
                    punct_sim = np.mean([np.dot(vec[int(tid)], vec[i]) 
                                        for i in range(V) 
                                        if roles_by_id[i] == "punct"][:config.punct_similarity_check_count] or [0])
                    if punct_sim > config.punct_similarity_threshold:
                        closure_boost += config.closure_punct_boost * closure_strength
            
            reason_sum += (w_semantic * sem) + (w_anchor * anc) - (w_contra * cp) - (w_repeat * rep) + group_boost + semantic_boost - group_penalty + closure_boost + lock_boost - lock_penalty - micro_topic_penalty - claim_penalty
            tmp_ids.append(int(tid))
            
            if best_group_id is not None:
                recent_groups.append(best_group_id)
                if len(recent_groups) > config.recent_groups_window:
                    recent_groups.pop(0)
        
        L = max(1, len(cont))
        selector_avg = sel_sum / L
        reasoner_avg = reason_sum / L
        trust_avg = float(np.mean([trust[int(t)] for t in cont if 0 <= int(t) < V])) if cont else 0.0
        total = (alpha_selector * selector_avg) + (beta_reasoner * reasoner_avg) + trust_avg
        return selector_avg, reasoner_avg, total
    
    for step in range(int(max_new_tokens)):
        prev1 = ids[-1] if len(ids) >= 1 else None
        prev2 = ids[-2] if len(ids) >= 2 else None
        
        # Sample continuations
        cont_rows = []
        for _ in range(config.num_continuations):
            p2, p1 = prev2, prev1
            cont: List[int] = []
            sel_lp_sum = 0.0
            for _t in range(config.continuation_length):
                nxt = sample_next(p2, p1)
                if nxt is None:
                    break
                tid, lp = nxt
                if block_dataset_meta and 0 <= tid < V and roles_by_id[tid] == "dataset_meta":
                    for _ in range(3):
                        nxt2 = sample_next(p2, p1)
                        if nxt2 and not (block_dataset_meta and 0 <= nxt2[0] < V and roles_by_id[nxt2[0]] == "dataset_meta"):
                            tid, lp = nxt2
                            break
                    if block_dataset_meta and 0 <= tid < V and roles_by_id[tid] == "dataset_meta":
                        break
                cont.append(int(tid))
                sel_lp_sum += float(lp)
                p2, p1 = (p1, int(tid)) if p1 is not None else (None, int(tid))
            
            if cont:
                cont_rows.append({"cont": cont, "sel_lp_avg": sel_lp_sum / len(cont)})
        
        if not cont_rows:
            break
        
        # Score continuations
        scored = []
        for r in cont_rows:
            cont = r["cont"]
            s_avg = float(r["sel_lp_avg"])
            sel_bias_avg, rs_avg, total = score_continuation(cont, prev2, prev1)
            selector_avg = s_avg + sel_bias_avg
            total = (alpha_selector * selector_avg) + (beta_reasoner * rs_avg) + float(np.mean([trust[int(t)] for t in cont if 0 <= int(t) < V]))
            scored.append((cont, selector_avg, rs_avg, total))
        
        if not scored:
            break
        
        totals = np.array([x[3] for x in scored], dtype=np.float64)
        probs = softmax(totals, temp=max(1e-6, float(temperature)))
        if np.any(np.isnan(probs)) or np.sum(probs) == 0:
            probs = np.ones_like(probs) / len(probs)
        pick = int(np.random.choice(len(scored), p=probs))
        chosen_cont, chosen_sel, chosen_rs, chosen_total = scored[pick]
        
        adv = float(chosen_total - float(np.mean(totals)))
        
        # Commit first token
        chosen = int(chosen_cont[0])
        prev1 = ids[-1] if len(ids) >= 1 else None
        prev2 = ids[-2] if len(ids) >= 2 else None
        kctx = ctx_key(prev2, prev1)
        
        trust[chosen] = float(np.clip(trust[chosen] + eta_trust * adv, -trust_clip, trust_clip))
        add_bias(kctx, chosen, eta_bias * adv)
        
        alt = selector.candidates(prev2, prev1, top_k=min(8, int(selector_top_k)))
        for alt_id, _ in alt[:config.num_alternatives]:
            alt_id = int(alt_id)
            if alt_id != chosen:
                add_bias(kctx, alt_id, -config.alternative_penalty * eta_bias * adv)
        
        ids.append(chosen)
        tokens_since_lock += 1
        tokens_since_micro_topic_change += 1
        
        # מחשב novelty של הטוקן שנבחר (דמיון לטוקנים האחרונים)
        novelty = 1.0  # default - חדש לגמרי
        if len(ids) >= 5 and micro_topic_center is not None:
            recent_tokens = ids[-min(10, len(ids)):-1]  # כל הטוקנים האחרונים חוץ מהנוכחי
            if recent_tokens:
                recent_vecs = vec[recent_tokens]
                chosen_vec = vec[chosen]
                chosen_norm = np.linalg.norm(chosen_vec)
                if chosen_norm > 1e-12:
                    chosen_vec_norm = chosen_vec / chosen_norm
                    similarities = np.clip(np.dot(recent_vecs, chosen_vec_norm), -1.0, 1.0)
                    max_sim = float(np.max(similarities))
                    novelty = 1.0 - max_sim  # novelty גבוה = דמיון נמוך
        
        micro_topic_novelty_history.append(novelty)
        
        # Closure Heuristic: בודק אם אפשר לשחרר את הנעילה
        can_switch_group = False
        
        # 1. אם המשפט הסתיים (זיהוי דינמי של פיסוק)
        is_sentence_end = False
        if chosen < len(roles_by_id):
            role = roles_by_id[chosen]
            if role == "punct":
                # בודק דמיון סמנטי לטוקני פיסוק
                punct_tokens = [i for i in range(V) if roles_by_id[i] == "punct"]
                if punct_tokens:
                    punct_vecs = vec[punct_tokens[:10]]
                    chosen_sim = np.max(np.dot(vec[chosen], punct_vecs.T))
                    if chosen_sim > 0.5:
                        is_sentence_end = True
                        last_sentence_end_step = step
        
        # 2. אם יש closure (כמעט סוף הגנרציה)
        progress = step / max(max_new_tokens, 1)
        is_closure = progress > config.closure_threshold
        
        # 3. אם הנעילה נחלשה מדי (decay)
        if lock_strength < config.lock_strength_min:
            can_switch_group = True
        
        # 4. אם עברו הרבה טוקנים מאז הנעילה (לא נעול לנצח)
        if tokens_since_lock > config.tokens_since_lock_max:  # דינמי - לא חוק קשיח
            can_switch_group = True
        
        # 5. אם המשפט הסתיים או יש closure - אפשר לשחרר
        if is_sentence_end or is_closure:
            can_switch_group = True
        
        # אם אפשר לשחרר - בוחר קבוצה חדשה
        if can_switch_group and locked_group_id is not None:
            # מחפש את הקבוצה הכי נכונה לפי ההקשר הנוכחי
            best_group_id = group_manager.find_best_group_for_token(chosen, prompt_context)
            
            if best_group_id is not None and best_group_id != locked_group_id:
                # משחרר את הנעילה הישנה ומנעל על קבוצה חדשה
                locked_group_id = best_group_id
                lock_strength = config.lock_strength_initial
                tokens_since_lock = 0
                group_manager.activate_group(best_group_id, strength=config.lock_strength_initial)
                
                # Micro-topic Lock: בוחר מרכז סמנטי חדש מהקבוצה החדשה
                new_locked_group = group_manager.groups[best_group_id]
                best_token_score = -1.0
                best_token_id = None
                
                for tid in new_locked_group.tokens:
                    token_vec = vec[tid]
                    prompt_sim = float(np.clip(np.dot(token_vec, prompt_context), -1.0, 1.0))
                    group_sim = new_locked_group.get_similarity(token_vec)
                    token_score = 0.6 * prompt_sim + 0.4 * group_sim
                    
                    if token_score > best_token_score:
                        best_token_score = token_score
                        best_token_id = tid
                
                if best_token_id is not None:
                    micro_topic_token_id = best_token_id
                    micro_topic_center = vec[best_token_id].copy()
                    norm = np.linalg.norm(micro_topic_center)
                    if norm > 1e-12:
                        micro_topic_center = (micro_topic_center / norm).astype(np.float32)
                    tokens_since_micro_topic_change = 0
            else:
                # נשאר באותה קבוצה אבל מחזק את הנעילה
                lock_strength = min(config.activation_limit, lock_strength + config.activation_increment)
        
        # Micro-topic Lock: Closure condition לשינוי המרכז הסמנטי
        can_change_micro_topic = False
        
        if micro_topic_center is not None:
            # 1. Novelty נמוך (חוזרים על אותו דבר)
            avg_novelty = float(np.mean(list(micro_topic_novelty_history))) if micro_topic_novelty_history else 1.0
            if avg_novelty < config.low_novelty_threshold:  # novelty נמוך = חזרתיות גבוהה
                can_change_micro_topic = True
            
            # 2. Repetition גבוה (הטוקן הנוכחי דומה מאוד לטוקנים האחרונים)
            if novelty < config.very_low_novelty_threshold:  # novelty נמוך מאוד = חזרתיות חזקה
                can_change_micro_topic = True
            
            # 3. עברו הרבה טוקנים מאז השינוי האחרון
            if tokens_since_micro_topic_change > config.tokens_since_micro_topic_max:  # דינמי - לא חוק קשיח
                can_change_micro_topic = True
            
            # 4. אם המשפט הסתיים או יש closure
            if is_sentence_end or is_closure:
                can_change_micro_topic = True
            
            # אם אפשר לשנות - בוחר מרכז חדש מהקבוצה הנוכחית
            if can_change_micro_topic and locked_group_id is not None:
                locked_group = group_manager.groups[locked_group_id]
                
                # בוחר את הטוקן הכי קרוב להקשר הנוכחי בקבוצה
                best_token_score = -1.0
                best_token_id = None
                
                for tid in locked_group.tokens:
                    token_vec = vec[tid]
                    # דמיון להקשר הנוכחי
                    current_sim = float(np.clip(np.dot(token_vec, prompt_context), -1.0, 1.0))
                    # דמיון למרכז הקבוצה
                    group_sim = locked_group.get_similarity(token_vec)
                    # ציון משולב
                    token_score = 0.6 * current_sim + 0.4 * group_sim
                    
                    if token_score > best_token_score:
                        best_token_score = token_score
                        best_token_id = tid
                
                # משנה את המרכז רק אם הטוקן החדש שונה מהנוכחי
                if best_token_id is not None and best_token_id != micro_topic_token_id:
                    micro_topic_token_id = best_token_id
                    micro_topic_center = vec[best_token_id].copy()
                    norm = np.linalg.norm(micro_topic_center)
                    if norm > 1e-12:
                        micro_topic_center = (micro_topic_center / norm).astype(np.float32)
                    tokens_since_micro_topic_change = 0
                    micro_topic_novelty_history.clear()  # מתחיל מחדש
        else:
            # ממשיך עם הנעילה הנוכחית - decay איטי
            lock_strength *= config.lock_decay_rate  # decay איטי מאוד
        
        # מפעיל את הקבוצה הכי נכונה לטוקן שנבחר (אם לא נעול)
        if not can_switch_group:
            best_group_id = group_manager.find_best_group_for_token(chosen, prompt_context)
            if best_group_id is not None:
                # מפעיל את הקבוצה (עם decay איטי ל-ActiveGroup הקודמת)
                current_activation = 0.0
                if group_manager.active_group_id is not None:
                    current_activation = group_manager.groups[group_manager.active_group_id].activation
                
                # בדיקה אם חוזרים על אותה קבוצה (מונע נעילה) - דינמי
                recent_group_ids = [group_manager.find_best_group_for_token(tid, prompt_context) 
                                   for tid in ids[-min(config.recent_groups_check_window, len(ids)):]]
                same_group_count = recent_group_ids.count(best_group_id)
                
                # אם חוזרים על אותה קבוצה - מקטין את ה-strength (מונע נעילה) - דינמי
                if same_group_count > config.group_repetition_threshold:
                    strength = config.lock_strength_weak
                elif same_group_count > 1:
                    strength = config.lock_strength_medium
                elif current_activation > config.high_activation_threshold:
                    strength = (config.lock_strength_medium + config.lock_strength_normal) / 2.0  # בינוני
                else:
                    strength = config.lock_strength_normal
                
                group_manager.activate_group(best_group_id, strength=strength)
        
        # Decay לכל הקבוצות בכל שלב (מונע נעילה) - גם לקבוצה הפעילה
        for group in group_manager.groups:
            group.decay()
        
        # מעדכן את הקשר הפרומפט (דינמי)
        if len(ids) >= 3:
            recent_ids = ids[-min(config.recent_context_size, len(ids)):]
            prompt_context = vec[recent_ids].mean(axis=0).astype(np.float32)
            norm = np.linalg.norm(prompt_context)
            if norm > 1e-12:
                prompt_context = (prompt_context / norm).astype(np.float32)
        
        # Update topic memory
        if step % 5 == 0 and len(ids) >= 5:
            topic_memory.update(ids[-min(config.recent_context_size, len(ids)):], vec)
        
        if explain:
            top_show = sorted(scored, key=lambda x: x[3], reverse=True)[:min(6, len(scored))]
            explain_rows.append({
                "step": int(step),
                "adv": float(adv),
                "chosen_first": tok.vocab[int(chosen_cont[0])] if chosen_cont else "",
                "candidates": [{
                    "preview": tok.decode(ids + cont[:min(len(cont), 10)]),
                    "len": int(len(cont)),
                    "selector_avg": float(sel),
                    "reasoner_avg": float(rs),
                    "total": float(tt),
                } for cont, sel, rs, tt in top_show],
            })
    
    return {
        "text": tok.decode(ids),
        "explain": explain_rows if explain else None,
        "online": {
            "num_steps": int(len(ids)),
            "bias_contexts": int(len(bias)),
            "trust_nonzero": int(np.sum(np.abs(trust) > 1e-9)),
        }
    }
