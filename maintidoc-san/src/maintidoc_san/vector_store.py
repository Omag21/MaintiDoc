"""Accès Qdrant : collections séparées texte / visuel, index de payload, mise à jour idempotente.

- Serveur Qdrant (Docker Compose) par défaut, mode local ou mémoire possibles.
- Identifiants de points déterministes (uuid5) : réindexer remplace au lieu de dupliquer.
- Index de payload sur les champs de filtrage : le contrôle d'accès s'applique
  AVANT la recherche (filtre Qdrant), conformément au cahier des charges.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

from qdrant_client import QdrantClient, models

from .config import Settings
from .schemas import Provenance, SearchHit

KEYWORD_FIELDS = ("document_id", "document_version", "document_sha256", "modality",
                  "equipment_type", "equipment_family", "access_groups", "figure_id", "language")
INTEGER_FIELDS = ("page",)


def make_client(settings: Settings) -> QdrantClient:
    if settings.qdrant_path:
        if settings.qdrant_path == ":memory:":
            return QdrantClient(location=":memory:")
        return QdrantClient(path=settings.qdrant_path)
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=30)


class VectorStore:
    def __init__(self, client: QdrantClient, settings: Settings):
        self.client = client
        self.settings = settings

    # ------------------------------------------------------------------ schéma
    def ensure_collection(self, name: str, dimension: int) -> None:
        if self.client.collection_exists(name):
            current = self.client.get_collection(name).config.params.vectors
            size = current.size if isinstance(current, models.VectorParams) else None
            if size is not None and size != dimension:
                raise ValueError(
                    f"La collection {name} a des vecteurs de taille {size}, attendu {dimension}. "
                    "Incrémentez MAINTIDOC_INDEX_VERSION pour créer un nouvel index.")
            return
        self.client.create_collection(
            name, vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE))
        if self.settings.qdrant_path:
            return  # le mode local ignore les index de payload (filtrage quand même correct)
        for field in KEYWORD_FIELDS:
            self.client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD)
        for field in INTEGER_FIELDS:
            self.client.create_payload_index(name, field, models.PayloadSchemaType.INTEGER)

    # ------------------------------------------------------------ mise à jour
    def indexed_sha256(self, collection: str, document_id: str) -> set[str]:
        """Empreintes du document actuellement présentes dans l'index."""
        if not self.client.collection_exists(collection):
            return set()
        hashes: set[str] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection, scroll_filter=_match("document_id", document_id), limit=256,
                offset=offset, with_payload=["document_sha256"], with_vectors=False)
            hashes.update(p.payload["document_sha256"] for p in points)
            if offset is None:
                return hashes

    def delete_document(self, collection: str, document_id: str, keep_sha256: str | None = None) -> None:
        """Supprime les points d'un document (sauf ceux de l'empreinte à conserver)."""
        if not self.client.collection_exists(collection):
            return
        must = [models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))]
        must_not = []
        if keep_sha256:
            must_not.append(models.FieldCondition(key="document_sha256",
                                                  match=models.MatchValue(value=keep_sha256)))
        self.client.delete(collection, points_selector=models.FilterSelector(
            filter=models.Filter(must=must, must_not=must_not)), wait=True)

    def upsert(self, collection: str, elements: Sequence[Provenance], vectors: Sequence[Sequence[float]],
               batch_size: int = 64) -> int:
        if len(elements) != len(vectors):
            raise ValueError("Autant de vecteurs que d'éléments sont attendus.")
        points = [
            models.PointStruct(id=el.point_id(), vector=list(map(float, vec)),
                               payload={**el.model_dump(mode="json"), "citation_ref": el.citation_ref})
            for el, vec in zip(elements, vectors)
        ]
        for start in range(0, len(points), batch_size):
            self.client.upsert(collection, points=points[start:start + batch_size], wait=True)
        return len(points)

    def count(self, collection: str, document_id: str | None = None) -> int:
        if not self.client.collection_exists(collection):
            return 0
        flt = _match("document_id", document_id) if document_id else None
        return self.client.count(collection, count_filter=flt, exact=True).count

    # --------------------------------------------------------------- recherche
    def search(self, collection: str, vector: Sequence[float], user_groups: Iterable[str],
               limit: int = 10, equipment_type: str | None = None,
               modalities: Iterable[str] | None = None) -> list[SearchHit]:
        """Recherche filtrée : seuls les éléments autorisés pour l'utilisateur sont candidats."""
        response = self.client.query_points(
            collection, query=list(map(float, vector)), limit=limit, with_payload=True,
            query_filter=access_filter(user_groups, equipment_type, modalities))
        return [SearchHit(rank=i, score=p.score, ref=p.payload.get("citation_ref", ""), payload=p.payload)
                for i, p in enumerate(response.points, 1)]


def _match(key: str, value: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))])


def access_filter(user_groups: Iterable[str], equipment_type: str | None = None,
                  modalities: Iterable[str] | None = None) -> models.Filter:
    groups = sorted({g.strip().lower() for g in user_groups if g and g.strip()})
    if not groups:
        # Aucun groupe => aucun document : refus par défaut.
        raise PermissionError("L'utilisateur n'appartient à aucun groupe d'accès.")
    must = [models.FieldCondition(key="access_groups", match=models.MatchAny(any=groups))]
    if equipment_type:
        must.append(models.FieldCondition(key="equipment_type", match=models.MatchValue(value=equipment_type)))
    if modalities:
        must.append(models.FieldCondition(key="modality", match=models.MatchAny(any=list(modalities))))
    return models.Filter(must=must)