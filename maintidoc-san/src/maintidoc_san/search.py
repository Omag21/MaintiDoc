"""Recherche texte et visuelle filtrée par droits d'accès.

Étape 1 : deux recherches indépendantes sur les collections Qdrant.
La fusion hybride, le reranker et la sélection des preuves arrivent à l'étape 2.
"""
from __future__ import annotations

from collections.abc import Iterable

from .config import Settings
from .embeddings import Embedder, ImageEmbedder
from .schemas import SearchHit
from .vector_store import VectorStore


class Retriever:
    def __init__(self, store: VectorStore, text_embedder: Embedder, image_embedder: ImageEmbedder,
                 settings: Settings):
        self.store, self.text_embedder, self.image_embedder = store, text_embedder, image_embedder
        self.settings = settings

    def search_text(self, question: str, user_groups: Iterable[str], limit: int = 8,
                    equipment_type: str | None = None) -> list[SearchHit]:
        vector = self.text_embedder.embed_texts([question])[0]
        return self.store.search(self.settings.text_collection, vector, user_groups, limit, equipment_type)

    def search_visual(self, question: str, user_groups: Iterable[str], limit: int = 5,
                      equipment_type: str | None = None, schemas_only: bool = True) -> list[SearchHit]:
        vector = self.image_embedder.embed_queries([question])[0]
        modalities = ["schema"] if schemas_only else None
        return self.store.search(self.settings.visual_collection, vector, user_groups, limit,
                                 equipment_type, modalities)