"""Tests de l'étape 1 : extraction, métadonnées, idempotence et filtrage d'accès.

Encodeurs factices + Qdrant en mémoire : aucun téléchargement de modèle ni serveur requis.
Lancement : uv run pytest -q
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np
import pytest

from maintidoc_san.config import PROJECT_ROOT, Settings
from maintidoc_san.extraction import extract_document
from maintidoc_san.ingest import Ingestor, load_corpus
from maintidoc_san.vector_store import VectorStore, make_client

PDF_NAME = "Manuel_Maintenance_SAN_Projet_Integrateur.pdf"


def _find_pdf() -> Path:
    for candidate in (PROJECT_ROOT / "corpus" / PDF_NAME, PROJECT_ROOT.parent / PDF_NAME):
        if candidate.is_file():
            return candidate
    pytest.skip("Manuel PDF introuvable")


class FakeText:
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


@pytest.fixture()
def env(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    shutil.copy(_find_pdf(), corpus_dir / PDF_NAME)
    config = tmp_path / "config"
    config.mkdir()
    (config / "corpus.yaml").write_text(
        "documents:\n"
        "  - document_id: SAN-MAN-001\n"
        f"    path: ../corpus/{PDF_NAME}\n"
        "    title: Manuel SAN\n"
        "    version: '1.1'\n"
        "    equipment_type: SAN-LAB-2000\n"
        "    equipment_family: stockage_san\n"
        "    access_groups: [Technicien, responsable_maintenance]\n", encoding="utf-8")
    settings = Settings(qdrant_path=":memory:", data_dir=tmp_path / "data",
                        corpus_file=config / "corpus.yaml")
    store = VectorStore(make_client(settings), settings)
    ingestor = Ingestor(store, FakeText(), FakeImage(), settings)
    return settings, store, ingestor


def test_extraction_metadata(env):
    settings, _, _ = env
    spec = load_corpus(settings.corpus_file)[0]
    doc = extract_document(spec, settings)
    assert doc.page_count == 9
    schemas = [v for v in doc.visuals if v.modality == "schema"]
    assert sorted(v.figure_id for v in schemas) == ["FIG-01", "FIG-02", "FIG-03", "FIG-04", "FIG-05"]
    assert {v.page for v in schemas} == {2, 3, 4, 5, 6}
    assert any(t.modality == "table" for t in doc.texts)
    assert any(t.modality == "figure_caption" for t in doc.texts)
    for el in [*doc.texts, *doc.visuals]:
        assert el.access_groups == ["responsable_maintenance", "technicien"]
        assert el.document_version == "1.1" and el.document_sha256 == doc.sha256
    for v in doc.visuals:
        assert (settings.data_dir / v.image_path).is_file()


def test_ingestion_is_idempotent(env):
    settings, store, ingestor = env
    specs = load_corpus(settings.corpus_file)
    first = ingestor.ingest_corpus(specs)[0]
    assert first.status == "indexed" and first.text_points > 0 and first.visual_points == 6
    n_text = store.count(settings.text_collection)
    second = ingestor.ingest_corpus(specs)[0]
    assert second.status == "unchanged"
    forced = ingestor.ingest_corpus(specs, force=True)[0]
    assert forced.status == "indexed"
    assert store.count(settings.text_collection) == n_text  # pas de doublons
    assert (settings.data_dir / "index_manifest.json").is_file()


def test_modified_document_replaces_old_points(env):
    settings, store, ingestor = env
    specs = load_corpus(settings.corpus_file)
    ingestor.ingest_corpus(specs)
    old_sha = next(iter(store.indexed_sha256(settings.text_collection, "SAN-MAN-001")))
    # Simule une nouvelle révision : on modifie une métadonnée PDF, ce qui change l'empreinte.
    import pymupdf
    with pymupdf.open(specs[0].path) as doc:
        doc.set_metadata({**doc.metadata, "subject": "revision 1.2"})
        data = doc.tobytes()
    specs[0].path.write_bytes(data)
    report = ingestor.ingest_corpus(specs)[0]
    assert report.status == "indexed"
    hashes = store.indexed_sha256(settings.text_collection, "SAN-MAN-001")
    assert hashes == {report.sha256} and old_sha not in hashes


def test_access_filter(env):
    settings, store, ingestor = env
    ingestor.ingest_corpus(load_corpus(settings.corpus_file))
    vec = FakeText().embed_texts(["voyant orange"])[0]
    assert store.search(settings.text_collection, vec, ["technicien"], limit=3)
    assert store.search(settings.text_collection, vec, ["invite"], limit=3) == []
    with pytest.raises(PermissionError):
        store.search(settings.text_collection, vec, [], limit=3)
    hits = store.search(settings.visual_collection, FakeImage().embed_queries(["x"])[0],
                        ["technicien"], limit=10, modalities=["schema"])
    assert len(hits) == 5 and all(h.payload["figure_id"] for h in hits)