"""Encodeurs : BGE-M3 (texte) et CLIP (images + requêtes multilingues).

Les modèles sont chargés à la demande et mis en cache. Une interface commune
(`Embedder`) permet de les remplacer par des encodeurs factices dans les tests.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Protocol, Sequence

import numpy as np
from PIL import Image


class Embedder(Protocol):
    dimension: int
    model_name: str

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray: ...


class ImageEmbedder(Protocol):
    dimension: int
    model_name: str

    def embed_images(self, paths: Sequence[str]) -> np.ndarray: ...
    def embed_queries(self, texts: Sequence[str]) -> np.ndarray: ...


@lru_cache(maxsize=4)
def _sentence_transformer(name: str, device: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name, device=device)


class BGEM3Embedder:
    """Embeddings denses BGE-M3 (1024 dimensions, multilingue, normalisés)."""

    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "cpu", batch_size: int = 8):
        self.model_name, self.device, self.batch_size = model_name, device, batch_size
        self._model = _sentence_transformer(model_name, device)
        self.dimension = self._model.get_embedding_dimension()
    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(list(texts), batch_size=self.batch_size,
                                     normalize_embeddings=True, show_progress_bar=len(texts) > 16)
        return np.asarray(vectors, dtype=np.float32)


class CLIPEmbedder:
    """Images encodées par CLIP ViT-B/32 ; requêtes encodées par sa variante multilingue.

    Les deux modèles partagent le même espace à 512 dimensions : une question en
    français peut donc être comparée directement aux schémas (plus de traduction manuelle).
    """

    def __init__(self, image_model: str = "sentence-transformers/clip-ViT-B-32",
                 query_model: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1",
                 device: str = "cpu", batch_size: int = 8):
        self.model_name = image_model
        self.query_model_name = query_model
        self.batch_size = batch_size
        self._image_model = _sentence_transformer(image_model, device)
        self._device = device
        self._query_model = None  # chargé seulement pour la recherche
        self.dimension = 512

    def embed_images(self, paths: Sequence[str]) -> np.ndarray:
        images = [Image.open(p).convert("RGB") for p in paths]
        try:
            vectors = self._image_model.encode(images, batch_size=self.batch_size,
                                               normalize_embeddings=True, show_progress_bar=False)
        finally:
            for im in images:
                im.close()
        return np.asarray(vectors, dtype=np.float32)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        if self._query_model is None:
            self._query_model = _sentence_transformer(self.query_model_name, self._device)
        vectors = self._query_model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)