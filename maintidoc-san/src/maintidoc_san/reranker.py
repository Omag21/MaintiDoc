"""Reranker multilingue (cross-encoder) : reclasse les candidats question par question.

Différence avec les embeddings :
- un encodeur (BGE-M3) calcule un vecteur par texte, séparément : rapide, mais approximatif ;
- un cross-encoder lit la question ET le passage ensemble : plus lent, mais bien plus précis.
On l'applique donc seulement aux ~30 meilleurs candidats déjà trouvés par Qdrant.

Le même modèle sert à vérifier qu'une citation soutient l'affirmation associée.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Protocol, Sequence

import numpy as np


class Reranker(Protocol):
    model_name: str

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        """Probabilité de pertinence (0-1) de chaque passage pour la requête."""
        ...


@lru_cache(maxsize=2)
def _cross_encoder(name: str, device: str, max_length: int):
    from sentence_transformers import CrossEncoder
    return CrossEncoder(name, device=device, max_length=max_length)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
                 device: str = "cpu", batch_size: int = 16, max_length: int = 512):
        self.model_name, self.batch_size = model_name, batch_size
        self._model = _cross_encoder(model_name, device, max_length)

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        logits = self._model.predict([(query, p) for p in passages], batch_size=self.batch_size,
                                     show_progress_bar=False)
        # Le modèle renvoie un score brut (logit) : la sigmoïde le ramène entre 0 et 1.
        return (1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float32)))).astype(np.float32)