from __future__ import annotations

from pathlib import Path

import torch

from Shared.interfaces import ITokenizer


class CharTokenizer(ITokenizer):
    """
    Character-level tokenizer for transcription.

    Vocabulary: <blank> + space + a-z + apostrophe
    """

    def __init__(self):
        self._chars = " " + "abcdefghijklmnopqrstuvwxyz" + "'"
        self._blank = "<blank>"
        self._char_to_idx: dict[str, int] = {c: i + 1 for i, c in enumerate(self._chars)}
        self._idx_to_char: dict[int, str] = {i + 1: c for i, c in enumerate(self._chars)}
        # Index 0 is reserved for CTC blank
        self._idx_to_char[0] = ""

    def encode(self, text: str) -> torch.Tensor:
        text = text.lower().strip()
        indices = []
        for c in text:
            if c in self._char_to_idx:
                indices.append(self._char_to_idx[c])
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        chars = []
        prev = -1
        for idx in tokens.tolist():
            if idx == self.blank_id:
                prev = idx
                continue
            if idx != prev:
                chars.append(self._idx_to_char.get(idx, ""))
            prev = idx
        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        return len(self._chars) + 1  # +1 for blank

    @property
    def blank_id(self) -> int:
        return 0


class BPETokenizer(ITokenizer):
    """
    BPE (Byte Pair Encoding) tokenizer for transcription.

    Uses a simple word-level BPE trained on the fly from training transcripts.
    Fewer output steps = better CTC alignment.
    Token 0 is reserved for CTC blank.
    """

    def __init__(self, vocab_size: int = 1000):
        self._target_vocab_size = vocab_size
        # Start with character-level vocab
        self._chars = list(" abcdefghijklmnopqrstuvwxyz'")
        # Token 0 = blank, tokens 1..N = actual symbols
        self._token_to_id: dict[str, int] = {c: i + 1 for i, c in enumerate(self._chars)}
        self._id_to_token: dict[int, str] = {i + 1: c for i, c in enumerate(self._chars)}
        self._id_to_token[0] = ""
        self._trained = False

    def train_from_texts(self, texts: list[str], max_merges: int | None = None) -> None:
        """Train BPE merges from a list of transcripts."""
        max_merges = max_merges or (self._target_vocab_size - len(self._chars) - 1)

        # Tokenize all texts into characters
        corpus: list[list[str]] = []
        for text in texts:
            text = text.lower().strip()
            chars = [c for c in text if c in self._token_to_id]
            if chars:
                corpus.append(chars)

        # BPE merge loop
        merges: list[tuple[str, str]] = []
        for _ in range(max_merges):
            # Count adjacent pairs
            pair_counts: dict[tuple[str, str], int] = {}
            for tokens in corpus:
                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)  # type: ignore
            merged = best_pair[0] + best_pair[1]
            merges.append(best_pair)

            # Add to vocab
            new_id = len(self._id_to_token)
            self._token_to_id[merged] = new_id
            self._id_to_token[new_id] = merged

            # Apply merge to corpus
            for idx, tokens in enumerate(corpus):
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                corpus[idx] = new_tokens

        self._merges = merges
        self._trained = True
        print(f"  BPE trained: {len(merges)} merges, vocab size {self.vocab_size}", flush=True)

    def encode(self, text: str) -> torch.Tensor:
        text = text.lower().strip()
        tokens = [c for c in text if c in self._token_to_id]

        # Apply merges in order
        if self._trained:
            for a, b in self._merges:
                merged = a + b
                new_tokens = []
                i = 0
                while i < len(tokens):
                    if i < len(tokens) - 1 and tokens[i] == a and tokens[i + 1] == b:
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(tokens[i])
                        i += 1
                tokens = new_tokens

        indices = [self._token_to_id[t] for t in tokens if t in self._token_to_id]
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        pieces = []
        prev = -1
        for idx in tokens.tolist():
            if idx == self.blank_id:
                prev = idx
                continue
            if idx != prev:
                pieces.append(self._id_to_token.get(idx, ""))
            prev = idx
        return "".join(pieces)

    @property
    def vocab_size(self) -> int:
        return len(self._id_to_token)

    @property
    def blank_id(self) -> int:
        return 0


def create_tokenizer(config_type: str = "char", **kwargs) -> ITokenizer:
    """Factory function to create the right tokenizer."""
    if config_type == "bpe":
        return BPETokenizer(vocab_size=kwargs.get("vocab_size", 1000))
    return CharTokenizer()
