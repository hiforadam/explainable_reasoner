import re
from dataclasses import dataclass
from typing import List, Dict, Tuple

_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]")

def simple_tokenize(text: str) -> List[str]:
    # lower-case, keep punctuation as separate tokens
    return _TOKEN_RE.findall(text.lower())

@dataclass
class ClosedVocabTokenizer:
    vocab: List[str]
    token_to_id: Dict[str, int]

    @classmethod
    def from_texts(cls, texts: List[str]) -> "ClosedVocabTokenizer":
        seen = {}
        for tx in texts:
            for tok in simple_tokenize(tx):
                seen[tok] = seen.get(tok, 0) + 1
        vocab = sorted(seen.keys())
        token_to_id = {t:i for i,t in enumerate(vocab)}
        return cls(vocab=vocab, token_to_id=token_to_id)

    def encode(self, text: str) -> List[int]:
        ids = []
        missing = []
        for tok in simple_tokenize(text):
            if tok not in self.token_to_id:
                missing.append(tok)
                continue
            ids.append(self.token_to_id[tok])
        if missing and not ids:
            raise ValueError(f"Token(s) not in closed vocabulary: {missing!r}")
        return ids

    def decode(self, ids: List[int]) -> str:
        toks = [self.vocab[i] for i in ids]
        # simple detokenization: space between alphanumerics, no space before punctuation
        out = []
        for t in toks:
            if not out:
                out.append(t)
                continue
            if re.fullmatch(r"[^\w\s]", t):  # punctuation
                out[-1] = out[-1] + t
            else:
                out.append(" " + t)
        return "".join(out)
