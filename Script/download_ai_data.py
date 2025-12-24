#!/usr/bin/env python3
"""Download and clean AI-related text data from various sources."""
import json
import re
import requests
from typing import List, Dict
from reasoner.tokenizer import simple_tokenize
import time

def clean_text(text: str) -> str:
    """Clean text: remove numbers, normalize whitespace, remove PDF artifacts."""
    # Remove numbers (but keep words with numbers like "AI2" or "GPT-3" as single tokens)
    # Remove standalone numbers and sequences of digits
    text = re.sub(r'\b\d+\b', '', text)
    
    # Remove PDF artifacts
    text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\d+/\d+', '', text)  # Remove page numbers like "1/10"
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
    
    # Remove very short lines (likely artifacts)
    lines = text.split('\n')
    lines = [line.strip() for line in lines if len(line.strip()) > 10]
    text = '\n'.join(lines)
    
    return text.strip()

def download_wikitext_simple(max_examples: int = 200) -> List[str]:
    """Download WikiText data using simple HTTP requests."""
    texts = []
    try:
        # Use a simple text source - Wikipedia API for random articles
        print("Downloading Wikipedia articles via API...")
        
        # Expanded list of AI/CS topics
        topics = [
            "artificial intelligence", "machine learning", "deep learning", "neural network",
            "natural language processing", "computer vision", "reinforcement learning",
            "transformer", "attention mechanism", "backpropagation", "gradient descent",
            "convolutional neural network", "recurrent neural network", "long short-term memory",
            "generative adversarial network", "variational autoencoder", "autoencoder",
            "support vector machine", "random forest", "decision tree", "k-means clustering",
            "principal component analysis", "linear regression", "logistic regression",
            "naive bayes", "k-nearest neighbors", "ensemble learning", "boosting",
            "bagging", "cross-validation", "overfitting", "regularization", "dropout",
            "batch normalization", "adam optimizer", "stochastic gradient descent",
            "backpropagation through time", "word embedding", "word2vec", "glove",
            "bert", "gpt", "transformer architecture", "self-attention", "multi-head attention",
            "positional encoding", "layer normalization", "residual connection",
            "feedforward network", "activation function", "sigmoid", "tanh", "relu",
            "leaky relu", "softmax", "loss function", "cross-entropy", "mean squared error",
            "optimization", "hyperparameter tuning", "grid search", "random search",
            "bayesian optimization", "early stopping", "learning rate", "momentum",
            "weight decay", "gradient clipping", "batch size", "epoch", "iteration",
            "training set", "validation set", "test set", "feature engineering",
            "data preprocessing", "normalization", "standardization", "one-hot encoding",
            "feature selection", "dimensionality reduction", "manifold learning",
            "unsupervised learning", "supervised learning", "semi-supervised learning",
            "transfer learning", "fine-tuning", "domain adaptation", "few-shot learning",
            "zero-shot learning", "meta-learning", "continual learning", "lifelong learning",
            "online learning", "incremental learning", "active learning", "curriculum learning",
            "reinforcement learning", "q-learning", "policy gradient", "actor-critic",
            "deep q-network", "proximal policy optimization", "trust region policy optimization",
            "imitation learning", "inverse reinforcement learning", "multi-agent reinforcement learning",
            "game theory", "nash equilibrium", "markov decision process", "partially observable markov decision process",
            "temporal difference learning", "monte carlo method", "dynamic programming",
            "value function", "policy function", "exploration", "exploitation", "epsilon-greedy",
            "thompson sampling", "upper confidence bound", "multi-armed bandit",
            "computer vision", "image classification", "object detection", "semantic segmentation",
            "instance segmentation", "image generation", "style transfer", "super-resolution",
            "optical flow", "stereo vision", "structure from motion", "simultaneous localization and mapping",
            "face recognition", "facial recognition", "emotion recognition", "gesture recognition",
            "action recognition", "video analysis", "video understanding", "video generation",
            "natural language processing", "text classification", "sentiment analysis",
            "named entity recognition", "part-of-speech tagging", "dependency parsing",
            "constituency parsing", "semantic role labeling", "coreference resolution",
            "question answering", "machine translation", "text summarization",
            "text generation", "language modeling", "next word prediction",
            "sequence-to-sequence", "encoder-decoder", "attention mechanism",
            "beam search", "greedy decoding", "nucleus sampling", "top-k sampling",
            "temperature sampling", "repetition penalty", "length penalty",
            "bleu score", "rouge score", "perplexity", "cross-entropy loss",
            "distributed systems", "parallel computing", "gpu computing",
            "tensor processing unit", "field-programmable gate array",
            "edge computing", "federated learning", "differential privacy",
            "adversarial machine learning", "adversarial examples", "robustness",
            "interpretability", "explainability", "feature importance",
            "attention visualization", "gradient-based methods", "perturbation methods",
            "model compression", "quantization", "pruning", "knowledge distillation",
            "neural architecture search", "automated machine learning",
            "automl", "hyperparameter optimization", "neural architecture search",
        ]
        
        print(f"Downloading summaries for {len(topics)} topics...")
        for i, topic in enumerate(topics):
            if len(texts) >= max_examples:
                break
            try:
                # Get article by title
                url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{topic.replace(' ', '_')}"
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    text = data.get('extract', '')
                    if text and len(text) > 200:
                        cleaned = clean_text(text)
                        if len(cleaned) > 200:
                            texts.append(cleaned)
                if (i + 1) % 20 == 0:
                    print(f"  Downloaded {len(texts)}/{max_examples} summaries...")
                time.sleep(0.3)  # Be polite but faster
            except Exception as e:
                if i % 50 == 0:  # Print errors less frequently
                    print(f"  Error downloading {topic}: {e}")
                continue
        
        print(f"Downloaded {len(texts)} Wikipedia summaries")
        return texts
    except Exception as e:
        print(f"Error downloading Wikipedia: {e}")
        return []

def download_wikipedia_full_articles(topic: str, max_paragraphs: int = 10) -> List[str]:
    """Download full Wikipedia articles using REST API."""
    texts = []
    try:
        # Get article content via REST API
        url = f"https://en.wikipedia.org/api/rest_v1/page/html/{topic.replace(' ', '_')}"
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            # Extract text from HTML (simple approach)
            html = response.text
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', html)
            # Remove references
            text = re.sub(r'\[.*?\]', '', text)
            text = re.sub(r'\(.*?\)', '', text)
            
            cleaned = clean_text(text)
            
            # Split into paragraphs
            paragraphs = [p.strip() for p in cleaned.split('\n\n') if len(p.strip()) > 200]
            
            for para in paragraphs[:max_paragraphs]:  # Keep multiple paragraphs
                if len(para) > 200:
                    texts.append(para)
        
        time.sleep(0.5)  # Be polite but faster
        return texts
    except Exception as e:
        return []  # Silent fail to avoid spam

def download_arxiv_abstracts_simple(query: str, max_results: int = 100, start: int = 0) -> List[str]:
    """Download arXiv abstracts using RSS feed."""
    texts = []
    try:
        # Use arXiv RSS API with pagination
        url = f"http://export.arxiv.org/api/query?search_query=all:{query.replace(' ', '+')}&start={start}&max_results={max_results}"
        response = requests.get(url, timeout=20)
        
        if response.status_code == 200:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(response.content)
            
            # Parse RSS/Atom feed
            namespace = {'atom': 'http://www.w3.org/2005/Atom'}
            entries = root.findall('atom:entry', namespace)
            
            for entry in entries:
                summary_elem = entry.find('atom:summary', namespace)
                if summary_elem is not None:
                    abstract = summary_elem.text
                    if abstract and len(abstract) > 200:
                        cleaned = clean_text(abstract)
                        if len(cleaned) > 200:
                            texts.append(cleaned)
        
        return texts
    except Exception as e:
        return []  # Silent fail

def create_examples_from_texts(texts: List[str], base_id: int) -> List[Dict]:
    """Create corpus examples from texts."""
    examples = []
    for i, text in enumerate(texts):
        if len(text) < 200:  # Skip very short texts
            continue
        
        example = {
            "id": f"ptpack_{base_id + i:06d}",
            "text": text
        }
        examples.append(example)
    
    return examples

def main():
    print("=" * 60)
    print("Downloading AI-related text data")
    print("=" * 60)
    
    all_texts = []
    base_id = 0
    
    # Try to load existing corpus to determine next ID
    try:
        with open('data/corpus.jsonl', 'r', encoding='utf-8') as f:
            existing = [json.loads(line) for line in f if line.strip()]
            base_id = len(existing)
            print(f"Found {base_id} existing examples, starting from ID {base_id}")
    except FileNotFoundError:
        print("No existing corpus found, starting from ID 0")
    
    target_examples = 1000
    print(f"\nTarget: {target_examples} examples")
    
    # 1. Download Wikipedia summaries (quick and reliable)
    print("\n1. Downloading Wikipedia summaries...")
    texts = download_wikitext_simple(max_examples=min(300, target_examples))
    all_texts.extend(texts)
    print(f"Total so far: {len(all_texts)}/{target_examples}")
    
    # 2. Download Wikipedia full articles on AI topics
    print("\n2. Downloading Wikipedia full articles...")
    ai_topics = [
        "artificial intelligence", "machine learning", "deep learning", "neural network",
        "natural language processing", "computer vision", "reinforcement learning",
        "transformer", "attention mechanism", "backpropagation", "gradient descent",
        "convolutional neural network", "recurrent neural network", "long short-term memory",
        "generative adversarial network", "variational autoencoder", "autoencoder",
        "support vector machine", "random forest", "decision tree", "k-means clustering",
        "principal component analysis", "linear regression", "logistic regression",
        "naive bayes", "k-nearest neighbors", "ensemble learning", "boosting",
        "bagging", "cross-validation", "overfitting", "regularization", "dropout",
        "batch normalization", "adam optimizer", "stochastic gradient descent",
        "word embedding", "word2vec", "glove", "bert", "gpt",
        "computer vision", "image classification", "object detection", "semantic segmentation",
        "natural language processing", "text classification", "sentiment analysis",
        "machine translation", "text summarization", "question answering",
        "distributed systems", "parallel computing", "gpu computing",
        "adversarial machine learning", "model compression", "quantization",
    ]
    
    print(f"Downloading full articles for {len(ai_topics)} topics...")
    for i, topic in enumerate(ai_topics):
        if len(all_texts) >= target_examples:
            break
        texts = download_wikipedia_full_articles(topic, max_paragraphs=3)
        all_texts.extend(texts)
        if (i + 1) % 10 == 0:
            print(f"  Downloaded {len(all_texts)}/{target_examples} examples...")
        time.sleep(0.3)  # Be polite but faster
    
    # 3. Download arXiv abstracts (multiple batches)
    print("\n3. Downloading arXiv abstracts...")
    arxiv_queries = [
        "artificial intelligence", "machine learning", "deep learning", "neural networks",
        "natural language processing", "computer vision", "reinforcement learning",
        "transformer", "attention", "backpropagation", "gradient", "optimization",
        "convolutional", "recurrent", "generative", "adversarial", "autoencoder",
        "classification", "regression", "clustering", "dimensionality reduction",
        "feature learning", "representation learning", "transfer learning",
        "meta-learning", "few-shot", "zero-shot", "continual learning",
        "robustness", "adversarial", "interpretability", "explainability",
    ]
    
    print(f"Downloading arXiv abstracts for {len(arxiv_queries)} queries...")
    for i, query in enumerate(arxiv_queries):
        if len(all_texts) >= target_examples:
            break
        
        # Download multiple batches
        for batch_start in [0, 100, 200]:
            if len(all_texts) >= target_examples:
                break
            texts = download_arxiv_abstracts_simple(query, max_results=100, start=batch_start)
            all_texts.extend(texts)
            if len(texts) > 0:
                print(f"  Query '{query}' batch {batch_start}: +{len(texts)} abstracts (total: {len(all_texts)})")
            time.sleep(1)  # Be polite to arXiv
        
        if (i + 1) % 5 == 0:
            print(f"  Progress: {len(all_texts)}/{target_examples} examples...")
    
    print(f"\nTotal downloaded: {len(all_texts)} examples")
    
    # Filter texts: keep only substantial ones
    print(f"\nFiltering texts...")
    filtered_texts = []
    for text in all_texts:
        if len(text) >= 200 and len(text) <= 5000:  # Reasonable length
            # Check token diversity
            tokens = set(simple_tokenize(text))
            if len(tokens) >= 20:  # At least 20 unique tokens
                filtered_texts.append(text)
    
    print(f"Total texts collected: {len(all_texts)}")
    print(f"After filtering: {len(filtered_texts)}")
    
    # Create examples
    examples = create_examples_from_texts(filtered_texts, base_id)
    
    print(f"\nCreated {len(examples)} new examples")
    
    # Save to temporary file
    output_file = "data/new_ai_data.jsonl"
    print(f"\nSaving to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(examples)} examples to {output_file}")
    print("\nNext steps:")
    print(f"1. Review {output_file}")
    print(f"2. Run: python3 deduplicate_corpus.py")
    print(f"3. Train the model")

if __name__ == "__main__":
    main()

