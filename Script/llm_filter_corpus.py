#!/usr/bin/env python3
"""Use LLM to filter and edit corpus examples for high-quality pretraining."""
import json
import requests
import time
from typing import List, Dict, Optional, Tuple
import re

def call_ollama(model: str, prompt: str, max_tokens: int = 500) -> str:
    """Call Ollama API."""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,  # Lower temperature for more consistent filtering
                    "max_tokens": max_tokens,
                }
            },
            timeout=60
        )
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            print(f"  Error: HTTP {response.status_code}")
            return ""
    except Exception as e:
        print(f"  Error calling Ollama: {e}")
        return ""

def evaluate_example_for_pretraining(text: str, model: str = "qwen2.5:7b-instruct") -> Tuple[str, bool]:
    """
    Use LLM to evaluate if example is good for pretraining.
    Returns: (edited_text_or_original, should_keep)
    """
    prompt = f"""You are evaluating text examples for a language model pretraining corpus. Your task is to determine if a text example is suitable for pretraining and optionally edit it to improve quality.

Guidelines for good pretraining text:
1. Should be coherent, well-written prose (not bullet points, lists, or fragments)
2. Should have natural sentence structure and flow
3. Should be informative and educational
4. Should not contain excessive repetition
5. Should not be too short (<200 chars) or too long (>5000 chars)
6. Should not contain artifacts like "this paper", "we propose", "in this work" excessively
7. Should maintain the original meaning and style

Text to evaluate:
{text[:2000]}  # Limit to avoid token limits

Respond in JSON format:
{{
  "keep": true/false,
  "reason": "brief reason",
  "edited_text": "edited version if needed, or original if no edits needed"
}}

If the text is good as-is, set "keep": true and "edited_text" to the original text.
If the text needs minor edits, set "keep": true and provide edited version.
If the text should be removed, set "keep": false."""

    response = call_ollama(model, prompt, max_tokens=800)
    
    if not response:
        return text, True  # Keep if LLM fails
    
    # Try to parse JSON response
    try:
        # Extract JSON from response (might have extra text)
        json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            keep = result.get("keep", True)
            edited = result.get("edited_text", text)
            
            # Validate edited text
            if edited and len(edited) >= 200 and len(edited) <= 5000:
                return edited, keep
            elif keep and len(text) >= 200:
                return text, True  # Keep original if edit is invalid
            else:
                return text, False
        else:
            # If no JSON, check if response suggests keeping
            if "keep" in response.lower() and "false" in response.lower():
                return text, False
            return text, True
    except json.JSONDecodeError:
        # If JSON parsing fails, try to infer from response
        if "remove" in response.lower() or "delete" in response.lower() or "not suitable" in response.lower():
            return text, False
        return text, True

def process_corpus(input_path: str, output_path: str, model: str = "qwen2.5:7b-instruct", 
                   batch_size: int = 10, start_from: int = 0):
    """Process corpus with LLM filtering and editing."""
    print(f"Loading corpus from {input_path}...")
    
    # Load examples
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
    print(f"Processing from index {start_from}...")
    
    # Process examples
    filtered = []
    removed_count = 0
    edited_count = 0
    
    for i, example in enumerate(examples):
        if i < start_from:
            # Keep examples before start_from
            filtered.append(example)
            continue
        
        text = example.get('text', '').strip()
        if not text:
            removed_count += 1
            continue
        
        # Evaluate with LLM
        if (i - start_from) % batch_size == 0:
            print(f"Processing example {i+1}/{len(examples)}... (removed: {removed_count}, edited: {edited_count})")
        
        edited_text, should_keep = evaluate_example_for_pretraining(text, model)
        
        if should_keep:
            # Update example with edited text if different
            if edited_text != text:
                example['text'] = edited_text
                edited_count += 1
            filtered.append(example)
        else:
            removed_count += 1
        
        # Be polite to Ollama API
        time.sleep(0.5)
        
        # Progress update
        if (i - start_from + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(examples)} processed, {len(filtered)} kept, {removed_count} removed, {edited_count} edited")
    
    # Save filtered corpus
    print(f"\nSaving filtered corpus to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for ex in filtered:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"\nResults:")
    print(f"  Original: {len(examples)} examples")
    print(f"  Kept: {len(filtered)} examples")
    print(f"  Removed: {removed_count} examples")
    print(f"  Edited: {edited_count} examples")
    print(f"  Reduction: {removed_count / len(examples) * 100:.1f}%")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Filter corpus using LLM")
    parser.add_argument("--input", default="data/corpus.jsonl", help="Input corpus file")
    parser.add_argument("--output", default="data/corpus.jsonl", help="Output corpus file")
    parser.add_argument("--model", default="qwen2.5:7b-instruct", help="Ollama model to use")
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for progress updates")
    parser.add_argument("--start-from", type=int, default=0, help="Start processing from this index")
    args = parser.parse_args()
    
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("Error: Ollama is not running. Please start Ollama first.")
            return
    except:
        print("Error: Cannot connect to Ollama. Please make sure Ollama is running on localhost:11434")
        return
    
    process_corpus(args.input, args.output, args.model, args.batch_size, args.start_from)

if __name__ == "__main__":
    main()

