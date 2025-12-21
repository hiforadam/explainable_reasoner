import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Iterator, Optional
from collections import Counter

_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]")

def simple_tokenize(text: str) -> List[str]:
    # lower-case, keep punctuation as separate tokens
    return _TOKEN_RE.findall(text.lower())

@dataclass
class ClosedVocabTokenizer:
    vocab: List[str]
    token_to_id: Dict[str, int]

    @classmethod
    def from_texts(cls, texts: List[str], max_vocab_size: Optional[int] = None) -> "ClosedVocabTokenizer":
        """Build vocab from texts. If max_vocab_size is set, keep only top tokens by frequency."""
        seen = Counter()
        for tx in texts:
            for tok in simple_tokenize(tx):
                seen[tok] += 1
        
        # Sort by frequency (descending), then alphabetically
        if max_vocab_size and len(seen) > max_vocab_size:
            # Keep top N by frequency
            top_tokens = [tok for tok, _ in seen.most_common(max_vocab_size)]
            vocab = sorted(top_tokens)
        else:
            vocab = sorted(seen.keys())
        
        token_to_id = {t: i for i, t in enumerate(vocab)}
        return cls(vocab=vocab, token_to_id=token_to_id)
    
    @classmethod
    def from_text_stream(cls, text_stream: Iterator[str], max_vocab_size: Optional[int] = None, 
                         sample_size: int = 100000) -> "ClosedVocabTokenizer":
        """Build vocab by streaming texts. Uses first sample_size texts for vocab building."""
        seen = Counter()
        count = 0
        for tx in text_stream:
            if count >= sample_size:
                break
            for tok in simple_tokenize(tx):
                seen[tok] += 1
            count += 1
        
        # Sort by frequency (descending), then alphabetically
        if max_vocab_size and len(seen) > max_vocab_size:
            top_tokens = [tok for tok, _ in seen.most_common(max_vocab_size)]
            vocab = sorted(top_tokens)
        else:
            vocab = sorted(seen.keys())
        
        token_to_id = {t: i for i, t in enumerate(vocab)}
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
