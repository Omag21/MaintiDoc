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
    # Serveur (Docker Compose) par défaut. QDRANT_PATH active le mode local
    # (dossier) ; QDRANT_PATH=":memory:" sert aux tests.
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_path: str | None = None

    # Version du schéma d'index : l'incrémenter si le découpage ou les modèles changent.
    index_version: str = "v1"
    text_collection_prefix: str = "maintidoc_text"
    visual_collection_prefix: str = "maintidoc_visual"

    # --- Modèles d'embeddings -----------------------------------------------
    text_embedding_model: str = "BAAI/bge-m3"
    # Encodeur d'images CLIP + encodeur de requêtes multilingue aligné sur le même espace.
    image_embedding_model: str = "sentence-transformers/clip-ViT-B-32"
    visual_query_model: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
    embedding_device: str = "cpu"
    embedding_batch_size: int = 8

    # --- Corpus et données ------------------------------------------------------
    corpus_file: Path = PROJECT_ROOT / "config" / "corpus.yaml"
    data_dir: Path = PROJECT_ROOT / "data"

    # --- Découpage ----------------------------------------------------------------
    chunk_max_chars: int = Field(default=1200, ge=200)
    min_image_side: int = Field(default=80, ge=1)

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