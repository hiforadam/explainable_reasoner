import json
from typing import List, Dict, Tuple

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

def load_examples(path: str) -> List[Tuple[str,str]]:
    rows = read_jsonl(path)
    out = []
    for r in rows:
        out.append((r["id"], r["text"]))
    return out

def load_token_descriptions(path: str) -> Dict[str, str]:
    rows = read_jsonl(path)
    desc = {}
    for r in rows:
        tok = r["token"]
        desc[tok] = r.get("description","") or ""
    return desc
