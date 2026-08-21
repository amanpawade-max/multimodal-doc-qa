"""Given a user question, retrieve the most relevant chunks and attach the
raw image crop for any table/figure hits, so generation can look at the
actual source region rather than only its caption."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.embed_store import VectorStore


@dataclass
class RetrievedItem:
    id: str
    type: str  # "text" | "table" | "figure"
    page: int
    doc_name: str
    text: str  # prose chunk, or the vision-generated caption for table/figure
    image_path: Optional[str]
    distance: float


def retrieve(store: VectorStore, question: str, top_k: int | None = None) -> list[RetrievedItem]:
    raw = store.query(question, top_k=top_k)
    items: list[RetrievedItem] = []
    if not raw["ids"] or not raw["ids"][0]:
        return items

    for i, item_id in enumerate(raw["ids"][0]):
        meta = raw["metadatas"][0][i]
        doc = raw["documents"][0][i]
        dist = raw["distances"][0][i]
        items.append(
            RetrievedItem(
                id=item_id,
                type=meta.get("type", "text"),
                page=meta.get("page", -1),
                doc_name=meta.get("doc_name", ""),
                text=doc,
                image_path=meta.get("image_path"),
                distance=dist,
            )
        )
    return items
