# Explainable Token-Reasoning Engine (Prototype)

This project implements an **explainable, token-by-token reasoning engine** that learns **semantic/transition structure automatically**
from a small dataset (≤20 examples) while keeping **token meaning descriptions manual**.

## Key constraints implemented
- **Vocabulary is closed**: tokens are taken only from `data/examples.jsonl`.
- `data/token_descriptions.jsonl` contains **ONLY** `{ "token": ..., "description": ... }` (no positions/appears_in metadata).
- All other information (roles, neighbors, constraints, contradiction penalties, transition weights) is **computed automatically** during training.
- **No hardcoded LLM direction**: All discourse roles, level mappings, and transition patterns are learned from data. No hardcoded role names (like "define", "explain", "qualify") or manual role assignments - everything is data-driven.

## Files
- `data/examples.jsonl` — up to 20 training examples.
- `data/token_descriptions.jsonl` — manual token descriptions scaffold (editable).
- `main.py` — CLI to build vocab, train, generate, and explain.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# (optional) rebuild token_descriptions scaffold from examples
python main.py build-token-descriptions --data data/examples.jsonl --out data/token_descriptions.jsonl

# train
rm -rf artifacts
mkdir -p artifacts
python main.py train --data data/examples.jsonl --token_desc data/token_descriptions.jsonl --artifacts artifacts

# generate (token-by-token) + show explanation
python main.py generate --artifacts artifacts --prompt "because the system" --max_new_tokens 25 --explain 1
```

## Notes
- This is a **research prototype**: the "semantic learning" is implemented via co-occurrence (PPMI) vectors + transition statistics,
  then combined with manual descriptions **without introducing new tokens**.
- If a prompt contains a token not present in the closed vocabulary, generation will raise an error.
- **Data-driven architecture**: Discourse roles and levels are inferred from corpus patterns, not hardcoded. Role transitions, emission probabilities, and level mappings are all learned automatically from training data.


## Dual-model mode (Reasoner + Selector debate)

This repo can optionally train and run two cooperating models from the same JSONL data:

- **Reasoner**: pattern/semantic critic (no next-token cross-entropy).
- **Selector**: sparse trigram/bigram token selector (language fluency).

Train both:

```bash
python main.py train-dual --data data/ptpack_llm_pretrain_60.jsonl --token_desc data/token_descriptions.jsonl --artifacts artifacts
```

Generate with debate:

```bash
python main.py generate-dual --artifacts artifacts --prompt "Large language models" --max_new_tokens 80 --explain
```

You can tune the debate at runtime using `--alpha_selector/--beta_reasoner` and the online adaptation rates `--eta_trust/--eta_bias`.
