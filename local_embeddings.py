"""LangChain-compatible wrapper around Chroma's local ONNX MiniLM embeddings."""

from __future__ import annotations

import os

from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_core.embeddings import Embeddings


class LocalEmbeddings(Embeddings):
    def __init__(self):
        self._ef = DefaultEmbeddingFunction()
        self._count_tokenizer = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._ef(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._ef([text])[0]

    def model_name(self) -> str:
        # Chroma's DefaultEmbeddingFunction reports name "default" and hides the
        # ONNX object. The vectors are all-MiniLM-L6-v2.
        if type(self._ef).__name__ == "DefaultEmbeddingFunction":
            return "all-MiniLM-L6-v2"
        name = getattr(self._ef, "MODEL_NAME", None)
        if name:
            return str(name)
        namer = getattr(self._ef, "name", None)
        if callable(namer):
            return str(namer())
        return type(self._ef).__name__

    def max_tokens(self) -> int:
        """Embedder window. Fails closed when the wrapper does not report one."""
        reporter = getattr(self._ef, "max_tokens", None)
        if callable(reporter):
            return int(reporter())
        raw = (os.getenv("EMBED_MAX_TOKENS") or "").strip()
        if raw:
            return int(raw)
        raise RuntimeError(
            "Embedder does not report max_tokens(); set EMBED_MAX_TOKENS"
        )

    def _measuring_tokenizer(self):
        """Tokenizer with truncation and padding off, so long texts keep their real length."""
        if self._count_tokenizer is not None:
            return self._count_tokenizer
        # DefaultEmbeddingFunction builds a fresh ONNXMiniLM on each call and
        # does not expose it. Load that same tokenizer here, without truncation
        # or padding — the cached ONNX tokenizer caps every text at 256.
        if type(self._ef).__name__ != "DefaultEmbeddingFunction":
            raise RuntimeError(
                "Untruncated token counts are implemented for the MiniLM "
                "default embedder. Set a tokenizer on the new embedder before "
                "chunking against its window."
            )
        from chromadb.utils.embedding_functions.onnx_mini_lm_l6_v2 import (
            ONNXMiniLM_L6_V2,
        )

        onnx = ONNXMiniLM_L6_V2()
        onnx._download_model_if_not_exists()
        path = os.path.join(onnx.DOWNLOAD_PATH, onnx.EXTRACTED_FOLDER_NAME, "tokenizer.json")
        tokenizer = onnx.Tokenizer.from_file(path)
        tokenizer.no_truncation()
        tokenizer.no_padding()
        self._count_tokenizer = tokenizer
        return self._count_tokenizer

    def count_tokens(self, text: str) -> int:
        """Token count including special tokens, matching the embedder's window."""
        encoded = self._measuring_tokenizer().encode(text or "")
        return len(encoded.ids)

    def _content_spans(self, text: str) -> list[tuple[int, int]]:
        encoded = self._measuring_tokenizer().encode(text or "")
        return [(start, end) for start, end in encoded.offsets if end > start]

    def token_prefix_end(self, text: str, n: int) -> int:
        """Char offset just after the first n tokens (the embedder's visible cut)."""
        if n <= 0:
            return 0
        encoded = self._measuring_tokenizer().encode(text or "")
        if len(encoded.ids) <= n:
            return len(text or "")
        end = encoded.offsets[n - 1][1]
        if end > 0:
            return end
        for start, stop in encoded.offsets[n:]:
            if stop > 0:
                return stop
        return len(text or "")

    def token_tail(self, text: str, n: int) -> str:
        """Last n content tokens, sliced from the original string."""
        if n <= 0 or not text:
            return ""
        spans = self._content_spans(text)
        if not spans:
            return ""
        if len(spans) <= n:
            return text[spans[0][0] : spans[-1][1]]
        start = spans[-n][0]
        end = spans[-1][1]
        return text[start:end]
