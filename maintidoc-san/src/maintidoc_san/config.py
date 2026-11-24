"""Configuration centralisée de MaintiDoc.

Les valeurs viennent des variables d'environnement (ou d'un fichier .env non versionné).
Aucun secret n'est écrit dans le code ni dans Git.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Charge un .env minimal (CLE=valeur) sans écraser l'environnement existant."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


class Settings(BaseModel):
    # --- Qdrant -----------------------------------------------------------
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_path: str | None = None

    # v2 : vecteurs nommés (dense + BM25), lignes de tableaux, OCR -> nouvelles collections.
    index_version: str = "v2"
    text_collection_prefix: str = "maintidoc_text"
    visual_collection_prefix: str = "maintidoc_visual"

    # --- Modèles ----------------------------------------------------------------
    text_embedding_model: str = "BAAI/bge-m3"
    image_embedding_model: str = "sentence-transformers/clip-ViT-B-32"
    visual_query_model: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    # Petit reranker multilingue (cross-encoder) : adaptable par fine-tuning à l'étape 6.
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 8

    # --- Corpus et données ------------------------------------------------------
    corpus_file: Path = PROJECT_ROOT / "config" / "corpus.yaml"
    eval_file: Path = PROJECT_ROOT / "config" / "eval_questions.yaml"
    data_dir: Path = PROJECT_ROOT / "data"

    # --- Extraction ---------------------------------------------------------------
    chunk_max_chars: int = Field(default=800, ge=200)
    min_image_side: int = Field(default=80, ge=1)
    ocr_enabled: bool = True
    ocr_min_page_chars: int = Field(default=30, ge=0)  # en dessous : page considérée comme scannée
    ocr_dpi: int = Field(default=200, ge=72)

    # --- Recherche (étape 2) ------------------------------------------------------
    bm25_avg_doc_tokens: float = Field(default=60.0, gt=0)
    candidates_per_source: int = Field(default=20, ge=1)  # profondeur de chaque recherche
    rrf_k: int = Field(default=60, ge=1)
    visual_weight: float = Field(default=0.5, ge=0)  # poids de la liste CLIP dans la fusion
    rerank_enabled: bool = True
    rerank_blend: float = Field(default=0.3, ge=0, le=1)  # part du score de fusion conservée
    min_rerank_score: float = Field(default=0.05, ge=0, le=1)  # sous ce seuil : pas une preuve
    relative_rerank_score: float = Field(default=0.2, ge=0, le=1)  # fraction du meilleur score
    max_evidence: int = Field(default=5, ge=1)
    max_evidence_per_page: int = Field(default=2, ge=1)
    evidence_char_budget: int = Field(default=4000, ge=500)
    citation_support_threshold: float = Field(default=0.3, ge=0, le=1)

    @property
    def text_collection(self) -> str:
        return f"{self.text_collection_prefix}_{self.index_version}"

    @property
    def visual_collection(self) -> str:
        return f"{self.visual_collection_prefix}_{self.index_version}"

    @property
    def assets_dir(self) -> Path:
        return self.data_dir / "assets"

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv(PROJECT_ROOT / ".env")
        values: dict[str, object] = {}
        for name in cls.model_fields:
            env_value = os.environ.get(f"MAINTIDOC_{name.upper()}")
            if env_value not in (None, ""):
                values[name] = env_value
        return cls(**values)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()