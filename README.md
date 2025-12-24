# Explainable Token-Reasoning Engine

A lightweight, explainable text generation system that learns semantic patterns from training data using a closed vocabulary approach.

## Features

- **Closed Vocabulary**: Only uses tokens from training data
- **Automatic Learning**: Discourse roles, transitions, and semantic patterns learned from data
- **Dual Model**: Optional reasoner + selector architecture for improved generation
- **Configurable**: All parameters centralized in `config.py`
- **Explainable**: Detailed generation traces available

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Token Descriptions

```bash
python main.py build-token-descriptions --data data/corpus.jsonl --out data/token_descriptions.jsonl
```

### 2. Train Model

```bash
python main.py train --data data/corpus.jsonl --token_desc data/token_descriptions.jsonl --artifacts artifacts
```

### 3. Generate Text

```bash
python main.py generate --artifacts artifacts --prompt "Large language models" --max_new_tokens 40
```

## Dual Model (Recommended)

The dual model combines a reasoner and selector for better generation quality:

### Train Dual Model

```bash
python main.py train-dual --data data/corpus.jsonl --token_desc data/token_descriptions.jsonl --artifacts artifacts
```

### Generate with Dual Model

```bash
python main.py generate-dual --artifacts artifacts --prompt "Machine learning models" --max_new_tokens 40 --seed 42
```

## Data Format

Training data should be in JSONL format:

```json
{"id": "example_001", "text": "Your training text here..."}
{"id": "example_002", "text": "Another example..."}
```

## Configuration

All model parameters are configurable via `reasoner/config.py`. Key parameters include:

- Architecture: `hidden_dim`, `vector_dim`, `learning_rate`
- Generation: `temperature`, `top_p`, `repetition_penalty`
- Semantic: `similarity_threshold`, `context_window`, `decay_rates`

## Project Structure

```
reasoner/
  ├── config.py          # Centralized configuration
  ├── train.py           # Training logic
  ├── generate.py        # Text generation
  ├── tokenizer.py       # Closed vocabulary tokenizer
  ├── model.py           # Model artifacts
  ├── selector.py        # Selector model
  └── data.py            # Data loading utilities
```

## Requirements

- Python 3.8+
- numpy
- See `requirements.txt` for full list

## License

See LICENSE file for details.
