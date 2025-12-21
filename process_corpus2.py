#!/usr/bin/env python3
"""
Script to process corpus2.txt, clean and filter the data,
convert it to JSONL format, and add to corpus.jsonl without duplicates.
"""

import json
import re
from pathlib import Path
from typing import List, Set, Dict


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    # Normalize newlines to single space (we'll add proper newlines later)
    text = re.sub(r'\n+', ' ', text)
    # Remove multiple spaces
    text = re.sub(r' +', ' ', text)
    return text


def split_into_chunks(text: str, min_length: int = 100) -> List[str]:
    """Split text into logical chunks (paragraphs or sections)."""
    chunks = []
    
    # First, try splitting by double newlines (paragraphs)
    paragraphs = re.split(r'\n\s*\n+', text)
    
    current_chunk = ""
    for para in paragraphs:
        para = clean_text(para)
        if not para or len(para) < 20:  # Skip very short paragraphs
            continue
            
        # If current chunk + new paragraph is reasonable, combine them
        if current_chunk and len(current_chunk) + len(para) < 2000:
            current_chunk += "\n\n" + para
        else:
            # Save current chunk if it's long enough
            if len(current_chunk) >= min_length:
                chunks.append(current_chunk)
            current_chunk = para
    
    # Add the last chunk
    if len(current_chunk) >= min_length:
        chunks.append(current_chunk)
    
    return chunks


def normalize_for_comparison(text: str) -> str:
    """Normalize text for duplicate detection."""
    # Convert to lowercase
    text = text.lower()
    # Remove all whitespace
    text = re.sub(r'\s+', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w]', '', text)
    return text


def is_duplicate(new_text: str, existing_texts: Set[str], threshold: float = 0.95) -> bool:
    """Check if text is a duplicate or near-duplicate."""
    normalized_new = normalize_for_comparison(new_text)
    
    # Check exact duplicates
    if normalized_new in existing_texts:
        return True
    
    # Check near-duplicates using simple character overlap
    # This is a simple approach; for better results, consider using fuzzy matching libraries
    if len(normalized_new) < 50:
        # For short texts, require exact match
        return normalized_new in existing_texts
    
    # For longer texts, check if a significant portion matches
    for existing in existing_texts:
        if len(existing) < 50:
            continue
        
        # Calculate simple overlap ratio
        overlap = 0
        min_len = min(len(normalized_new), len(existing))
        max_len = max(len(normalized_new), len(existing))
        
        # Check if one is a substring of the other (with high similarity)
        if normalized_new in existing or existing in normalized_new:
            overlap = min_len / max_len
        else:
            # Simple character n-gram overlap (using 10-char windows)
            new_ngrams = set()
            for i in range(len(normalized_new) - 9):
                new_ngrams.add(normalized_new[i:i+10])
            
            existing_ngrams = set()
            for i in range(len(existing) - 9):
                existing_ngrams.add(existing[i:i+10])
            
            if new_ngrams and existing_ngrams:
                overlap = len(new_ngrams & existing_ngrams) / len(new_ngrams | existing_ngrams)
        
        if overlap >= threshold:
            return True
    
    return False


def load_existing_corpus(corpus_path: Path) -> tuple[List[Dict], Set[str], int]:
    """Load existing corpus and return entries, normalized texts, and max ID."""
    entries = []
    normalized_texts = set()
    max_id = -1
    
    if corpus_path.exists():
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    entries.append(entry)
                    normalized_texts.add(normalize_for_comparison(entry['text']))
                    
                    # Extract ID number
                    id_str = entry.get('id', '')
                    if id_str.startswith('ptpack_'):
                        try:
                            id_num = int(id_str.replace('ptpack_', ''))
                            max_id = max(max_id, id_num)
                        except ValueError:
                            pass
                except json.JSONDecodeError:
                    continue
    
    return entries, normalized_texts, max_id


def process_corpus2(input_path: Path, output_path: Path):
    """Process corpus2.txt and add to corpus.jsonl."""
    print(f"Reading {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    print("Splitting into chunks...")
    chunks = split_into_chunks(raw_text, min_length=100)
    print(f"Found {len(chunks)} chunks")
    
    print("Loading existing corpus...")
    existing_entries, normalized_texts, max_id = load_existing_corpus(output_path)
    print(f"Existing corpus has {len(existing_entries)} entries, max ID: {max_id}")
    
    new_entries = []
    duplicates_found = 0
    
    print("Processing chunks and checking for duplicates...")
    for i, chunk in enumerate(chunks):
        chunk = clean_text(chunk)
        
        # Skip if too short
        if len(chunk) < 100:
            continue
        
        # Check for duplicates
        if is_duplicate(chunk, normalized_texts):
            duplicates_found += 1
            print(f"  Chunk {i+1}: Duplicate found, skipping")
            continue
        
        # Create new entry
        new_id = max_id + 1 + len(new_entries)
        entry = {
            "id": f"ptpack_{new_id:06d}",
            "text": chunk
        }
        
        new_entries.append(entry)
        normalized_texts.add(normalize_for_comparison(chunk))
        print(f"  Chunk {i+1}: Added as {entry['id']}")
    
    if not new_entries:
        print("\nNo new entries to add (all were duplicates or too short).")
        return
    
    print(f"\nAdding {len(new_entries)} new entries to {output_path}...")
    
    # Append new entries to corpus.jsonl
    with open(output_path, 'a', encoding='utf-8') as f:
        for entry in new_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"✓ Successfully added {len(new_entries)} entries")
    print(f"✓ Skipped {duplicates_found} duplicates")
    print(f"✓ Total entries in corpus: {len(existing_entries) + len(new_entries)}")


if __name__ == "__main__":
    data_dir = Path("data")
    input_file = data_dir / "corpus2.txt"
    output_file = data_dir / "corpus.jsonl"
    
    if not input_file.exists():
        print(f"Error: {input_file} not found!")
        exit(1)
    
    process_corpus2(input_file, output_file)

