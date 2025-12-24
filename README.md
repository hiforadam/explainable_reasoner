# Explainable Token-Reasoning Engine

A lightweight, explainable text generation system that learns semantic patterns from training data.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Train Dual Model

```bash
python main.py train-dual --data data/corpus.jsonl --token_desc data/token_descriptions.jsonl --artifacts artifacts
```

### Generate Text

```bash
python main.py generate-dual --artifacts artifacts --prompt "Your prompt here" --max_new_tokens 40
```

## Data Format

Training data should be in JSONL format:

```json
{"id": "example_001", "text": "Your training text here..."}
```

## Configuration

All parameters are configurable via `reasoner/config.py`.

## License

See LICENSE file for details.
