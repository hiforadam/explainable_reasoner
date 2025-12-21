import json
from typing import List, Dict, Tuple, Iterator

def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_jsonl(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def stream_examples(path: str) -> Iterator[Tuple[str, str]]:
    """Stream examples one at a time to avoid loading all into memory."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            yield (r["id"], r["text"])

def load_examples(path: str) -> List[Tuple[str,str]]:
    """Load all examples (for backward compatibility). Use stream_examples for large datasets."""
    return list(stream_examples(path))

def load_token_descriptions(path: str) -> Dict[str, str]:
    """Load token descriptions. This is small enough to keep in memory."""
    desc = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tok = r["token"]
            desc[tok] = r.get("description","") or ""
    return desc
