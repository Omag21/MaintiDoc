"""Schémas Pydantic des éléments indexés et de leur provenance."""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

# Espace de noms fixe : un même élément donne toujours le même identifiant Qdrant.
POINT_NAMESPACE = uuid.UUID("5b0c7a52-6f3e-4c1e-9a43-2d1f8f0c9e11")


class Modality(StrEnum):
    TEXT = "text"
    TABLE = "table"
    FIGURE_CAPTION = "figure_caption"
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
    figure_id: str | None = None  # renseigné pour les légendes de figures
    chunk_index: int | None = None


class VisualElement(Provenance):
    figure_id: str | None = None
    caption: str | None = None
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
    score: float
    ref: str
    payload: dict