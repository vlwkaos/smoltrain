import numpy as np
import torch
import torch.nn as nn

# UTF-8 byte encoding: vocab covers all 256 byte values (0-255), max sequence length 512
VOCAB_SIZE = 256
MAX_LEN = 512


def encode_text(text: str, max_len: int = MAX_LEN) -> np.ndarray:
    """Encode text as UTF-8 bytes, truncate/pad to max_len. Returns int64 array."""
    ids = list(text.encode("utf-8", errors="replace"))
    ids = ids[:max_len]
    ids += [0] * (max_len - len(ids))
    return np.array(ids, dtype=np.int64)


class CharCNN(nn.Module):
    """Char-level CNN classifier for routing: simple / reasoning / agentic."""

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        embed_dim: int = 32,
        num_filters: int = 128,
        kernel_sizes: list[int] | None = None,
        max_len: int = MAX_LEN,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 4, 5, 6]
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList(
            [nn.Conv1d(embed_dim, num_filters, k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def encode(self, text: str) -> torch.Tensor:
        """str → (1, max_len) int64 tensor of byte values."""
        ids = list(text.encode("utf-8", errors="replace"))
        ids = ids[: self.max_len]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor([ids], dtype=torch.long)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, max_len) int64 → (batch, num_classes) logits."""
        # (batch, max_len, embed_dim) → (batch, embed_dim, max_len)
        e = self.embedding(x).permute(0, 2, 1)
        pooled = [torch.amax(conv(e), dim=2) for conv in self.convs]
        out = torch.cat(pooled, dim=1)   # (batch, num_filters * n_kernels)
        out = self.dropout(out)
        return self.fc(out)
