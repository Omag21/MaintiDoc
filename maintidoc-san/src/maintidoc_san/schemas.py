"""Schémas Pydantic des éléments indexés, des résultats de recherche et des citations."""
from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

# Espace de noms fixe : un même élément donne toujours le même identifiant Qdrant.
POINT_NAMESPACE = uuid.UUID("5b0c7a52-6f3e-4c1e-9a43-2d1f8f0c9e11")

# Référence de citation : DOCUMENT:pPAGE:ELEMENT, par ex. SAN-MAN-001:p5:p05_table_01_row_02
CITATION_RE = re.compile(r"([A-Za-z0-9_.-]+):p(\d+):([A-Za-z0-9_-]+)")


class Modality(StrEnum):
    TEXT = "text"
    TABLE = "table"              # tableau complet en Markdown (représentation structurée)
    TABLE_ROW = "table_row"      # une ligne de tableau, indexée seule (conseil du manuel)
    FIGURE_CAPTION = "figure_caption"
    IMAGE_OCR = "image_ocr"      # texte lu par OCR dans un schéma
    PAGE_OCR = "page_ocr"        # texte lu par OCR sur une page scannée
    SCHEMA = "schema"
    ILLUSTRATION = "illustration"


class DocumentSpec(BaseModel):
    """Une entrée du registre de corpus (config/corpus.yaml)."""

    document_id: str = Field(pattern=r"^[A-Za-z0-9_.-]+$")
    path: Path
    title: str
    version: str
    equipment_type: str
    equipment_family: str
    access_groups: list[str] = Field(min_length=1)
    language: str = "fr"
    # Sections à ne pas indexer (ex. le jeu de questions d'évaluation : éviter la fuite de données).
    exclude_sections: list[str] = Field(default_factory=list)

    @field_validator("access_groups")
    @classmethod
    def _normalize_groups(cls, groups: list[str]) -> list[str]:
        return sorted({g.strip().lower() for g in groups if g.strip()})


class Provenance(BaseModel):
    """Métadonnées communes à tout élément indexé (texte ou visuel)."""

    document_id: str
    document_title: str
    document_version: str
    document_sha256: str
    source_file: str
    page: int = Field(ge=1)
    section: str | None = None  # numéro de section du manuel (« 4 » pour « 4. Voyants... »)
    element_id: str
    modality: Modality
    equipment_type: str
    equipment_family: str
    access_groups: list[str]
    language: str = "fr"
    index_version: str
    embedding_model: str
    ingested_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def citation_ref(self) -> str:
        return f"{self.document_id}:p{self.page}:{self.element_id}"

    def point_id(self) -> str:
        key = f"{self.document_id}:{self.document_version}:{self.element_id}"
        return str(uuid.uuid5(POINT_NAMESPACE, key))


class TextElement(Provenance):
    text: str
    bbox: list[float] | None = None
    figure_id: str | None = None   # légendes et OCR de schémas
    chunk_index: int | None = None
    table_id: str | None = None    # tableaux et lignes de tableaux
    cells: dict[str, str] | None = None  # ligne de tableau : {en-tête: valeur}


class VisualElement(Provenance):
    figure_id: str | None = None
    caption: str | None = None
    ocr_text: str | None = None
    image_path: str  # relatif à data_dir
    image_sha256: str
    width: int
    height: int
    bbox: list[float] | None = None


class ExtractedDocument(BaseModel):
    spec: DocumentSpec
    sha256: str
    page_count: int
    texts: list[TextElement]
    visuals: list[VisualElement]


class SearchHit(BaseModel):
    rank: int
    score: float                       # score de la dernière étape (RRF ou reranker)
    ref: str
    payload: dict
    source: str = "text"               # "text" | "visual"
    retrieval_score: float | None = None  # score Qdrant d'origine
    rerank_score: float | None = None     # probabilité de pertinence (0-1) du reranker
    related_refs: list[str] = Field(default_factory=list)  # éléments fusionnés (légende, OCR, image)

    @property
    def evidence_text(self) -> str:
        """Texte présenté au reranker et au futur modèle génératif."""
        p = self.payload
        if self.source == "visual":
            parts = [f"{p.get('figure_id') or 'Image'} : {p.get('caption') or ''}", p.get("ocr_text") or ""]
            return "\n".join(x for x in parts if x.strip())
        return p.get("text", "")


# --------------------------------------------------------------------- citations
class Citation(BaseModel):
    """Format de citation du cahier des charges."""

    document_id: str
    page: int
    element_id: str

    @property
    def ref(self) -> str:
        return f"{self.document_id}:p{self.page}:{self.element_id}"

    @classmethod
    def parse(cls, ref: str) -> "Citation":
        match = CITATION_RE.fullmatch(ref.strip())
        if not match:
            raise ValueError(f"Référence de citation invalide : {ref!r}")
        return cls(document_id=match.group(1), page=int(match.group(2)), element_id=match.group(3))


class CitationCheck(BaseModel):
    sentence: str
    ref: str | None
    status: str  # "supported" | "unsupported" | "unknown_ref" | "uncited"
    support_score: float | None = None


class CitationReport(BaseModel):
    checks: list[CitationCheck]

    @property
    def cited(self) -> list[CitationCheck]:
        return [c for c in self.checks if c.ref is not None]

    @property
    def validity_rate(self) -> float:
        """Part des références qui pointent vers une preuve réellement récupérée."""
        cited = self.cited
        return sum(c.status != "unknown_ref" for c in cited) / len(cited) if cited else 0.0

    @property
    def support_rate(self) -> float:
        """Part des références qui soutiennent l'affirmation associée (objectif : >= 90 %)."""
        cited = self.cited
        return sum(c.status == "supported" for c in cited) / len(cited) if cited else 0.0

    @property
    def ok(self) -> bool:
        return bool(self.cited) and all(c.status == "supported" for c in self.checks)