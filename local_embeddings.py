"""LangChain-compatible wrapper around Chroma's local ONNX MiniLM embeddings."""

from langchain_core.embeddings import Embeddings
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


class LocalEmbeddings(Embeddings):
    def __init__(self):
        self._ef = DefaultEmbeddingFunction()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._ef(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._ef([text])[0]
