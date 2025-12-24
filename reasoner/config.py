"""Configuration module - כל הפרמטרים הקבועים במקום אחד."""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ModelConfig:
    """קונפיגורציה למודל - כל הפרמטרים במקום אחד."""
    
    # Architecture parameters
    hidden_dim: int = 64
    num_iterations: int = 10
    learning_rate: float = 0.01
    window_size: int = 2
    vector_dim: int = 32
    
    # Vocabulary parameters
    max_vocab_size: int = 50000
    bigram_top_k: int = 150
    
    # Semantic thresholds
    similarity_threshold: float = 0.55
    contradiction_max_pairs: int = 300
    semantic_repeat_threshold: float = 0.6
    semantic_repeat_penalty: float = 0.85
    micro_topic_similarity_threshold: float = 0.4
    context_similarity_threshold: float = 0.2
    
    # Semantic boost thresholds
    strong_similarity_threshold: float = 0.6
    medium_similarity_threshold: float = 0.4
    topic_merge_threshold: float = 0.7
    
    # Weights for scoring
    token_sim_weight: float = 0.4
    context_sim_weight: float = 0.4
    strength_weight: float = 0.1
    activation_weight: float = 0.1
    
    # Generation weights
    bigram_weight: float = 1.0
    semantic_weight: float = 0.9
    anchor_weight: float = 0.2
    contra_weight: float = 1.1
    structure_weight: float = 0.25
    coherence_weight: float = 0.35
    
    # Decay rates
    context_decay: float = 0.95
    group_decay_rate: float = 0.95
    lock_decay_rate: float = 0.98
    high_activation_decay: float = 0.90
    
    # Activation parameters
    activation_increment: float = 0.3
    activation_limit: float = 1.0
    high_activation_threshold: float = 0.7
    very_high_activation_threshold: float = 0.8
    
    # Boost parameters
    group_boost_base: float = 0.15
    group_boost_high_penalty: float = 0.4
    semantic_boost_strong: float = 0.2
    semantic_boost_medium: float = 0.1
    lock_boost_base: float = 0.4
    lock_boost_similarity: float = 0.2
    lock_penalty_base: float = 0.5
    micro_topic_penalty_base: float = 0.6
    claim_penalty_base: float = 0.2
    
    # Group parameters
    initial_groups: int = 12
    group_alpha: float = 0.1
    group_connection_threshold: float = -3.0
    group_connection_normalization: float = 5.0
    group_connection_boost: float = 0.12
    
    # Topic memory parameters
    max_topics: int = 3
    topic_merge_alpha: float = 0.3
    topic_weight_increment: float = 0.1
    topic_weight_limit: float = 1.0
    new_topic_weight: float = 0.5
    
    # Structure parameters
    min_sentence_length: int = 3
    good_sentence_length: int = 5
    max_sentence_length: int = 20
    structure_boost_long: float = 0.3
    structure_penalty_short: float = 0.2
    content_word_boost: float = 0.15
    
    # Closure parameters
    closure_threshold: float = 0.7
    closure_boost_base: float = 0.5
    closure_punct_boost: float = 0.3
    
    # Repetition parameters
    repeat_window: int = 6
    semantic_repeat_window: int = 2
    repetition_penalty: float = 1.5
    phrase_penalty_multiplier: float = 1.5
    
    # Context parameters
    context_window: int = 40
    short_context_window: int = 12
    prompt_anchor_size: int = 16
    recent_context_size: int = 10
    
    # Hierarchical context weights
    short_context_weight_base: float = 0.3
    short_context_weight_progress: float = 0.4
    long_context_weight_base: float = 0.7
    
    # Sentence end detection
    sentence_end_ratio_threshold: float = 0.25
    sentence_end_min_count: float = 1.0
    sentence_end_bonus: float = 0.05
    max_sentence_end_tokens: int = 6
    
    # Role inference parameters
    df_threshold: float = 0.55
    max_token_length_for_function: int = 4
    meta_shape_threshold: float = 0.67
    cluster_score_meta_weight: float = 0.75
    cluster_score_df_weight: float = 0.25
    cluster_cutoff_quantile: float = 0.85
    max_meta_clusters: int = 2
    
    # K-means parameters
    kmeans_seed: Optional[int] = None  # None = random
    kmeans_iters: int = 30
    kmeans_min_clusters: int = 4
    kmeans_max_clusters: int = 12
    kmeans_sqrt_factor: float = 3.0
    
    # Discourse parameters
    num_discourse_roles: int = 5
    discourse_levels: List[str] = field(default_factory=lambda: ["punct", "meta", "function", "content"])
    discourse_role_min_run: List[int] = field(default_factory=lambda: [2] * 5)
    discourse_pos_thresholds: List[int] = field(default_factory=lambda: [3, 8])
    
    # Lock parameters
    lock_strength_initial: float = 1.0
    lock_strength_min: float = 0.1
    lock_strength_weak: float = 0.2
    lock_strength_medium: float = 0.4
    lock_strength_normal: float = 0.7
    tokens_since_lock_max: int = 25
    tokens_since_micro_topic_max: int = 15
    
    # Novelty parameters
    novelty_history_size: int = 10
    low_novelty_threshold: float = 0.3
    very_low_novelty_threshold: float = 0.2
    
    # Group repetition parameters
    group_repetition_penalty_base: float = 0.3
    group_repetition_penalty_high: float = 0.6
    group_repetition_threshold: int = 2
    recent_groups_window: int = 8
    recent_groups_check_window: int = 5
    
    # Sampling parameters
    top_p_default: float = 0.9
    temperature_default: float = 1.0
    
    # Selector parameters
    selector_smooth: float = 0.5
    selector_max_per_context: int = 256
    selector_top_k_default: int = 24
    
    # Dual model parameters
    alpha_selector: float = 0.55
    beta_reasoner: float = 0.45
    eta_trust: float = 0.10
    eta_bias: float = 0.08
    trust_clip: float = 2.0
    w_semantic: float = 1.0
    w_anchor: float = 0.55
    w_contra: float = 1.25
    w_repeat: float = 0.85
    
    # Generation parameters
    num_continuations: int = 12
    continuation_length: int = 6
    num_alternatives: int = 4
    alternative_penalty: float = 0.25
    
    # Tokenization regex (can be customized)
    token_regex: str = r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]"
    
    # Learning tracker parameters
    learning_tracker_window: int = 10
    learning_rate_accelerate: float = 1.2
    learning_rate_slow_down: float = 0.8
    learning_rate_freeze: float = 0.1
    learning_rate_min: float = 1e-7
    learning_rate_max: float = 1e-1
    gradient_strengthening_threshold: float = 0.5
    gradient_struggling_threshold: float = 1.5
    convergence_threshold: float = 1e-5
    convergence_window: int = 5
    
    # Adaptive structure parameters
    punct_similarity_threshold: float = 0.3
    punct_similarity_check_count: int = 5
    sentence_end_similarity_threshold: float = 0.5
    
    # Claim type inference (will be learned, but defaults provided)
    claim_type_default: str = "DEFINE"
    claim_type_similarity_top_k: int = 20
    claim_type_check_top_k: int = 10
    
    # Shape features parameters
    non_alnum_threshold: float = 0.34
    meta_shape_divisor: float = 3.5
    meta_shape_non_alnum_weight: float = 0.5
    
    # Semantic group similarity weights
    semantic_group_token_weight: float = 0.6
    semantic_group_context_weight: float = 0.4
    
    # Coherence score thresholds
    coherence_high_threshold: float = 0.7
    coherence_medium_threshold: float = 0.5
    coherence_high_boost: float = 0.4
    coherence_medium_boost: float = 0.3
    coherence_low_boost: float = 0.2
    
    # Closure similarity threshold
    closure_similarity_threshold: float = 0.5


# Global default config instance
default_config = ModelConfig()

