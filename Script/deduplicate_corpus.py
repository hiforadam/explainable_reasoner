#!/usr/bin/env python3
"""Deduplicate and filter similar examples from corpus."""
import json
import numpy as np
from typing import List, Dict, Tuple
from reasoner.tokenizer import simple_tokenize
from collections import Counter

def jaccard_similarity(text1: str, text2: str) -> float:
    """Calculate Jaccard similarity between two texts."""
    tokens1 = set(simple_tokenize(text1))
    tokens2 = set(simple_tokenize(text2))
    if not tokens1 and not tokens2:
        return 1.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0

def token_overlap_ratio(text1: str, text2: str) -> float:
    """Calculate token overlap ratio."""
    tokens1 = set(simple_tokenize(text1))
    tokens2 = set(simple_tokenize(text2))
    if not tokens1 or not tokens2:
        return 0.0
    overlap = len(tokens1 & tokens2)
    min_len = min(len(tokens1), len(tokens2))
    return overlap / min_len if min_len > 0 else 0.0

def text_similarity(text1: str, text2: str) -> float:
    """Calculate combined similarity score."""
    jaccard = jaccard_similarity(text1, text2)
    overlap = token_overlap_ratio(text1, text2)
    # Weighted combination
    return 0.6 * jaccard + 0.4 * overlap

def find_duplicates_and_similar(examples: List[Dict], similarity_threshold: float = 0.85) -> Tuple[List[Dict], List[int]]:
    """Find and remove duplicates and very similar examples."""
    print(f"Analyzing {len(examples)} examples for duplicates and similarities...")
    
    # First pass: exact duplicates
    seen_texts = {}
    exact_duplicates = set()
    for i, ex in enumerate(examples):
        text = ex.get('text', '').strip()
        text_lower = text.lower()
        if text_lower in seen_texts:
            exact_duplicates.add(i)
            print(f"Found exact duplicate: {ex.get('id', i)} matches {seen_texts[text_lower]}")
        else:
            seen_texts[text_lower] = ex.get('id', i)
    
    print(f"Found {len(exact_duplicates)} exact duplicates")
    
    # Second pass: very similar examples
    similar_pairs = []
    to_remove = set(exact_duplicates)
    
    print("Checking for similar examples...")
    for i in range(len(examples)):
        if i in to_remove:
            continue
        if i % 50 == 0:
            print(f"  Processed {i}/{len(examples)} examples...")
        
        text1 = examples[i].get('text', '').strip()
        if not text1:
            to_remove.add(i)
            continue
        
        for j in range(i + 1, len(examples)):
            if j in to_remove:
                continue
            
            text2 = examples[j].get('text', '').strip()
            if not text2:
                to_remove.add(j)
                continue
            
            similarity = text_similarity(text1, text2)
            if similarity >= similarity_threshold:
                # Keep the longer or first one, remove the shorter
                if len(text1) >= len(text2):
                    to_remove.add(j)
                    similar_pairs.append((examples[i].get('id', i), examples[j].get('id', j), similarity))
                else:
                    to_remove.add(i)
                    similar_pairs.append((examples[j].get('id', j), examples[i].get('id', i), similarity))
                    break
    
    print(f"Found {len(similar_pairs)} similar pairs (threshold: {similarity_threshold})")
    if similar_pairs:
        print("Sample similar pairs:")
        for id1, id2, sim in similar_pairs[:10]:
            print(f"  {id1} ~ {id2} (similarity: {sim:.3f})")
    
    # Filter out duplicates and similar
    filtered = [ex for i, ex in enumerate(examples) if i not in to_remove]
    
    return filtered, sorted(to_remove)

def filter_short_examples(examples: List[Dict], min_length: int = 200) -> List[Dict]:
    """Filter out very short examples."""
    filtered = []
    removed = 0
    for ex in examples:
        text = ex.get('text', '').strip()
        if len(text) >= min_length:
            filtered.append(ex)
        else:
            removed += 1
    
    if removed > 0:
        print(f"Removed {removed} examples shorter than {min_length} characters")
    
    return filtered

def main():
    input_path = "data/corpus.jsonl"
    output_path = "data/corpus.jsonl"
    
    # Load examples
    print(f"Loading corpus from {input_path}...")
    examples = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                examples.append(data)
            except json.JSONDecodeError:
                continue
    
    print(f"Loaded {len(examples)} examples")
    
    # Filter short examples first
    examples = filter_short_examples(examples, min_length=200)
    
    # Find and remove duplicates and similar
    filtered, removed_indices = find_duplicates_and_similar(examples, similarity_threshold=0.85)
    
    print(f"\nOriginal: {len(examples)} examples")
    print(f"Removed: {len(removed_indices)} examples")
    print(f"Filtered: {len(filtered)} examples")
    print(f"Reduction: {len(removed_indices) / len(examples) * 100:.1f}%")
    
    # Save filtered corpus
    print(f"\nSaving filtered corpus to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in filtered:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print("Done!")
    
    # Show statistics
    lengths = [len(ex.get('text', '')) for ex in filtered]
    if lengths:
        print(f"\nStatistics:")
        print(f"  Average length: {np.mean(lengths):.0f} characters")
        print(f"  Min length: {min(lengths)} characters")
        print(f"  Max length: {max(lengths)} characters")

if __name__ == "__main__":
    main()

