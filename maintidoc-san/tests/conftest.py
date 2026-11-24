"""Outils de test partagés : faux modèles, Qdrant en mémoire, corpus temporaire.

Aucun téléchargement de modèle ni serveur n'est nécessaire pour lancer les tests.
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest

from maintidoc_san.config import PROJECT_ROOT, Settings
from maintidoc_san.ingest import Ingestor
from maintidoc_san.sparse import tokenize
from maintidoc_san.vector_store import VectorStore, make_client

PDF_NAME = "Manuel_Maintenance_SAN_Projet_Integrateur.pdf"


def find_pdf() -> Path:
    for candidate in (PROJECT_ROOT / "corpus" / PDF_NAME, PROJECT_ROOT.parent / PDF_NAME):
        if candidate.is_file():
            return candidate
    pytest.skip("Manuel PDF introuvable")


class FakeText:
    """Vecteur pseudo-aléatoire stable par texte : teste la mécanique, pas la qualité."""

    dimension, model_name = 32, "fake-text"

    def embed_texts(self, texts):
        out = []
        for t in texts:
            seed = int(hashlib.md5(t.encode()).hexdigest()[:8], 16)
            v = np.random.default_rng(seed).normal(size=self.dimension)
            out.append(v / np.linalg.norm(v))
        return np.asarray(out, dtype=np.float32)


class FakeImage(FakeText):
    dimension, model_name = 16, "fake-image"

    def embed_images(self, paths):
        return self.embed_texts([Path(p).name for p in paths])

    def embed_queries(self, texts):
        return self.embed_texts(texts)


class FakeOCR:
    """Renvoie un texte fixe pour chaque image : vérifie le circuit OCR sans le vrai moteur."""

    def __init__(self, lines=("Controleur A", "Disque B03")):
        self.lines = list(lines)
        self.calls = 0

    def read(self, image: bytes) -> list[str]:
        self.calls += 1
        return list(self.lines)


class FakeReranker:
    """Score = part des termes de la question présents dans le passage (entre 0 et 1)."""

    model_name = "fake-reranker"

    def score(self, query, passages):
        q = set(tokenize(query))
        return np.asarray([len(q & set(tokenize(p))) / (len(q) or 1) for p in passages], dtype=np.float32)


def write_corpus(tmp_path: Path, extra: str = "") -> Path:
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(exist_ok=True)
    shutil.copy(find_pdf(), corpus_dir / PDF_NAME)
    config = tmp_path / "config"
    config.mkdir(exist_ok=True)
    corpus = config / "corpus.yaml"
    corpus.write_text(
        "documents:\n"
        "  - document_id: SAN-MAN-001\n"
        f"    path: ../corpus/{PDF_NAME}\n"
        "    title: Manuel SAN\n"
        "    version: '1.1'\n"
        "    equipment_type: SAN-LAB-2000\n"
        "    equipment_family: stockage_san\n"
        "    access_groups: [Technicien, responsable_maintenance]\n"
        "    exclude_sections: ['7']\n" + extra, encoding="utf-8")
    return corpus


@pytest.fixture()
def env(tmp_path):
    settings = Settings(qdrant_path=":memory:", data_dir=tmp_path / "data",
                        corpus_file=write_corpus(tmp_path))
    store = VectorStore(make_client(settings), settings)
    ingestor = Ingestor(store, FakeText(), FakeImage(), settings, ocr=FakeOCR())
    return settings, store, ingestor