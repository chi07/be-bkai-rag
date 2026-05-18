from __future__ import annotations

from collections.abc import Sequence
import os

import numpy as np
from sentence_transformers import SentenceTransformer

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


class EmbeddingModel:
    def __init__(self, model_name: str, max_seq_length: int | None = None) -> None:
        self.model_name = model_name
        if torch is not None:
            torch.set_num_threads(int(os.getenv("RAG_TORCH_NUM_THREADS", "1")))
        self.model = SentenceTransformer(model_name)
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length

    @property
    def dimension(self) -> int:
        if hasattr(self.model, "get_embedding_dimension"):
            return self.model.get_embedding_dimension()
        return self.model.get_sentence_embedding_dimension()

    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        return self.model.encode(
            list(texts),
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def encode_one(self, text: str) -> list[float]:
        vector = self.encode([text], batch_size=1)[0]
        return vector.astype(float).tolist()
