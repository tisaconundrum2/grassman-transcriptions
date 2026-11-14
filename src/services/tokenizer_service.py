import torch

from src.interfaces import ITokenizer


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
