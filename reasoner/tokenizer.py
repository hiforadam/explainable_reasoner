"""Closed vocabulary tokenizer."""
import logging
import re
from dataclasses import dataclass
from typing import List, Dict, Optional
from collections import Counter

logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]")


def simple_tokenize(text: str) -> List[str]:
    """Tokenize text (lowercase, keep punctuation as separate tokens)."""
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


@dataclass
class ClosedVocabTokenizer:
    """Closed vocabulary tokenizer."""
    vocab: List[str]
    token_to_id: Dict[str, int]

    @classmethod
    def from_texts(cls, texts: List[str], max_vocab_size: Optional[int] = None) -> "ClosedVocabTokenizer":
        """
        Build vocab from texts.
        
        Args:
            texts: List of text strings
            max_vocab_size: Maximum vocabulary size (None for unlimited)
        
        Returns:
            ClosedVocabTokenizer instance
        
        Raises:
            ValueError: If texts is empty or no tokens found
        """
        if not texts:
            raise ValueError("Cannot build vocabulary from empty text list")
        
        seen = Counter()
        for tx in texts:
            if not tx or not tx.strip():
                continue
            for tok in simple_tokenize(tx):
                seen[tok] += 1
        
        if not seen:
            raise ValueError("No tokens found in texts")
        
        if max_vocab_size and len(seen) > max_vocab_size:
            top_tokens = [tok for tok, _ in seen.most_common(max_vocab_size)]
            vocab = sorted(top_tokens)
            logger.info(f"Limited vocabulary to top {max_vocab_size} tokens (from {len(seen)} unique tokens)")
        else:
            vocab = sorted(seen.keys())
        
        token_to_id = {t: i for i, t in enumerate(vocab)}
        return cls(vocab=vocab, token_to_id=token_to_id)

    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text string
        
        Returns:
            List of token IDs
        
        Raises:
            ValueError: If text is empty or contains tokens not in vocabulary
        """
        if not text or not text.strip():
            return []
        
        ids = []
        missing = []
        for tok in simple_tokenize(text):
            if tok not in self.token_to_id:
                missing.append(tok)
                continue
            ids.append(self.token_to_id[tok])
        
        if missing and not ids:
            raise ValueError(f"Token(s) not in closed vocabulary: {missing[:10]!r}{'...' if len(missing) > 10 else ''}")
        elif missing:
            logger.warning(f"Some tokens not in vocabulary (ignored): {len(missing)} tokens")
        
        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Decode token IDs to text.
        
        Args:
            ids: List of token IDs
        
        Returns:
            Decoded text string
        
        Raises:
            IndexError: If any ID is out of range
        """
        if not ids:
            return ""
        
        # Validate IDs
        invalid_ids = [i for i in ids if i < 0 or i >= len(self.vocab)]
        if invalid_ids:
            raise IndexError(f"Invalid token IDs: {invalid_ids[:10]}")
        
        toks = [self.vocab[i] for i in ids]
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

