"""Data loading utilities."""
import json
import logging
from typing import List, Dict, Tuple, Iterator
from pathlib import Path

logger = logging.getLogger(__name__)


def read_jsonl(path: str) -> List[Dict]:
    """Read JSONL file with error handling."""
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    rows = []
    line_num = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num} in {path}: {e}")
                continue
    
    if not rows:
        logger.warning(f"No valid rows found in {path}")
    
    return rows


def write_jsonl(path: str, rows: List[Dict]) -> None:
    """Write JSONL file with error handling."""
    if not rows:
        logger.warning("Writing empty JSONL file")
    
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    except IOError as e:
        logger.error(f"Failed to write JSONL file {path}: {e}")
        raise


def stream_examples(path: str) -> Iterator[Tuple[str, str]]:
    """Stream examples one at a time with error handling."""
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    line_num = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if "id" not in r or "text" not in r:
                    logger.warning(f"Skipping invalid example at line {line_num}: missing 'id' or 'text'")
                    continue
                if not r["text"] or not r["text"].strip():
                    logger.warning(f"Skipping empty text at line {line_num}")
                    continue
                yield (r["id"], r["text"])
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                continue


def load_examples(path: str) -> List[Tuple[str, str]]:
    """Load all examples with validation."""
    examples = list(stream_examples(path))
    if not examples:
        raise ValueError(f"No valid examples found in {path}")
    return examples


def load_token_descriptions(path: str) -> Dict[str, str]:
    """Load token descriptions with error handling."""
    if not Path(path).exists():
        logger.warning(f"Token descriptions file not found: {path}")
        return {}
    
    desc = {}
    line_num = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if "token" not in r:
                    logger.warning(f"Skipping invalid entry at line {line_num}: missing 'token'")
                    continue
                desc[r["token"]] = r.get("description", "") or ""
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                continue
    
    return desc

