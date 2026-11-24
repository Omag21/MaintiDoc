"""Pipeline d'ingestion : registre de corpus -> extraction -> embeddings -> Qdrant.

Idempotent : un document dont l'empreinte SHA-256 est déjà indexée est ignoré.
Un document modifié est réindexé, puis ses anciens points sont supprimés.
Étape 2 : chaque passage reçoit aussi un vecteur BM25 ; les schémas passent par l'OCR.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import yaml

from .config import Settings
from .embeddings import Embedder, ImageEmbedder
from .extraction import extract_document, sha256_bytes
from .ocr import OCREngine, RapidOCREngine
from .schemas import DocumentSpec
from .sparse import document_vector
from .vector_store import VectorStore

log = logging.getLogger("maintidoc.ingest")


@dataclass
class IngestReport:
    document_id: str
    version: str
    sha256: str
    status: str  # "indexed" | "unchanged" | "error"
    pages: int = 0
    text_points: int = 0
    visual_points: int = 0
    seconds: float = 0.0
    error: str | None = None


def load_corpus(corpus_file: Path) -> list[DocumentSpec]:
    raw = yaml.safe_load(corpus_file.read_text(encoding="utf-8")) or {}
    specs = []
    for entry in raw.get("documents", []):
        spec = DocumentSpec(**entry)
        if not spec.path.is_absolute():
            spec.path = (corpus_file.parent / spec.path).resolve()
        specs.append(spec)
    ids = [s.document_id for s in specs]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"document_id en double dans le corpus : {sorted(duplicates)}")
    return specs


class Ingestor:
    def __init__(self, store: VectorStore, text_embedder: Embedder, image_embedder: ImageEmbedder,
                 settings: Settings, ocr: OCREngine | None = None):
        self.store, self.text_embedder, self.image_embedder = store, text_embedder, image_embedder
        self.settings = settings
        # OCR désactivable (MAINTIDOC_OCR_ENABLED=false) ; un faux moteur peut être injecté (tests).
        self.ocr = ocr if ocr is not None else (RapidOCREngine() if settings.ocr_enabled else None)
        self.store.ensure_collection(settings.text_collection, text_embedder.dimension, hybrid=True)
        self.store.ensure_collection(settings.visual_collection, image_embedder.dimension)

    def is_up_to_date(self, spec: DocumentSpec, sha: str) -> bool:
        s = self.settings
        return (self.store.indexed_sha256(s.text_collection, spec.document_id) == {sha}
                and self.store.indexed_sha256(s.visual_collection, spec.document_id) in ({sha}, set()))

    def ingest(self, spec: DocumentSpec, force: bool = False) -> IngestReport:
        start = time.perf_counter()
        data = spec.path.read_bytes()
        sha = sha256_bytes(data)
        if not force and self.is_up_to_date(spec, sha):
            log.info("%s inchangé (%s), ignoré", spec.document_id, sha[:12])
            return IngestReport(spec.document_id, spec.version, sha, "unchanged")

        doc = extract_document(spec, self.settings, pdf_bytes=data, ocr=self.ocr)
        s = self.settings
        passages = [t.text for t in doc.texts]
        text_vectors = self.text_embedder.embed_texts(passages) if passages else []
        sparse_vectors = [document_vector(t, s.bm25_avg_doc_tokens) for t in passages]
        n_text = self.store.upsert(s.text_collection, doc.texts, text_vectors, sparse_vectors)

        image_paths = [str(s.data_dir / v.image_path) for v in doc.visuals]
        image_vectors = self.image_embedder.embed_images(image_paths) if image_paths else []
        n_visual = self.store.upsert(s.visual_collection, doc.visuals, image_vectors)

        # Les nouveaux points sont en place : on retire ceux d'une version précédente.
        self.store.delete_document(s.text_collection, spec.document_id, keep_sha256=sha)
        self.store.delete_document(s.visual_collection, spec.document_id, keep_sha256=sha)

        report = IngestReport(spec.document_id, spec.version, sha, "indexed", doc.page_count,
                              n_text, n_visual, round(time.perf_counter() - start, 2))
        log.info("%s indexé : %d passages, %d images", spec.document_id, n_text, n_visual)
        return report

    def ingest_corpus(self, specs: list[DocumentSpec], force: bool = False,
                      remove_missing: bool = False) -> list[IngestReport]:
        reports = []
        for spec in specs:
            try:
                reports.append(self.ingest(spec, force=force))
            except Exception as exc:  # un document en échec ne bloque pas le corpus
                log.exception("Échec d'ingestion de %s", spec.document_id)
                reports.append(IngestReport(spec.document_id, spec.version, "", "error", error=str(exc)))
        if remove_missing:
            self._remove_unlisted({s.document_id for s in specs})
        self._write_manifest(reports)
        return reports

    def _remove_unlisted(self, keep_ids: set[str]) -> None:
        s = self.settings
        for collection in (s.text_collection, s.visual_collection):
            indexed: set[str] = set()
            offset = None
            while True:
                points, offset = self.store.client.scroll(collection, limit=256, offset=offset,
                                                          with_payload=["document_id"], with_vectors=False)
                indexed.update(p.payload["document_id"] for p in points)
                if offset is None:
                    break
            for document_id in indexed - keep_ids:
                log.info("Suppression de %s (retiré du corpus)", document_id)
                self.store.delete_document(collection, document_id)

    def _write_manifest(self, reports: list[IngestReport]) -> None:
        s = self.settings
        s.data_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "index_version": s.index_version,
            "text_collection": s.text_collection,
            "visual_collection": s.visual_collection,
            "text_embedding_model": self.text_embedder.model_name,
            "image_embedding_model": self.image_embedder.model_name,
            "sparse": "bm25 (IDF calculé par Qdrant)",
            "ocr": type(self.ocr).__name__ if self.ocr else None,
            "documents": [asdict(r) for r in reports],
        }
        (s.data_dir / "index_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")