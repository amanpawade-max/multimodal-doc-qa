"""Embedding + persistent vector storage (ChromaDB).

We compute embeddings ourselves (local sentence-transformers or an API) and
pass them to Chroma directly, rather than relying on Chroma's built-in
embedding function — that keeps embedding provider and vector store fully
decoupled and avoids any implicit model downloads at query time.
"""
from __future__ import annotations

from typing import Iterable

import chromadb

from config import config


class Embedder:
    def __init__(self):
        self.provider = config.EMBEDDING_PROVIDER
        if self.provider == "local":
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(config.LOCAL_EMBEDDING_MODEL)
        elif self.provider == "openai":
            from openai import OpenAI

            self._client = OpenAI(api_key=config.OPENAI_API_KEY)
        else:
            raise ValueError(f"Unknown EMBEDDING_PROVIDER: {self.provider}")

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self.provider == "local":
            return self._model.encode(texts, normalize_embeddings=True).tolist()
        # openai
        resp = self._client.embeddings.create(model=config.OPENAI_EMBEDDING_MODEL, input=texts)
        return [d.embedding for d in resp.data]


class VectorStore:
    def __init__(self, embedder: Embedder | None = None):
        self.embedder = embedder or Embedder()
        self._client = chromadb.PersistentClient(path=config.INDEX_DIR)
        self._collection = self._client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, ids: list[str], texts: list[str], metadatas: list[dict]) -> None:
        """`texts` is what gets embedded and searched (prose chunks, or
        vision-generated captions for tables/figures). `metadatas` carries
        the pointer back to the source (page, bbox, image_path, type)."""
        if not ids:
            return
        embeddings = self.embedder.embed(texts)
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

    def query(self, question: str, top_k: int | None = None) -> dict:
        top_k = top_k or config.TOP_K
        q_emb = self.embedder.embed([question])[0]
        return self._collection.query(
            query_embeddings=[q_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def count(self) -> int:
        return self._collection.count()
