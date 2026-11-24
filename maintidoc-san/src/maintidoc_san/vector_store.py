"""Accès Qdrant : collections texte (dense + BM25) et visuelle, mise à jour idempotente.

- Collection texte : deux vecteurs nommés par point, « dense » (BGE-M3) et « bm25 » (épars).
- Collection visuelle : un vecteur CLIP par image.
- Identifiants de points déterministes (uuid5) : réindexer remplace au lieu de dupliquer.
- Le filtre d'accès est appliqué DANS Qdrant, avant le classement : un document
  non autorisé n'est jamais candidat, même pour la recherche lexicale.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence

from qdrant_client import QdrantClient, models

from .config import Settings
from .schemas import Provenance, SearchHit

DENSE, SPARSE = "dense", "bm25"  # noms des vecteurs de la collection texte

KEYWORD_FIELDS = ("document_id", "document_version", "document_sha256", "modality", "section",
                  "equipment_type", "equipment_family", "access_groups", "figure_id", "table_id", "language")
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
    def ensure_collection(self, name: str, dimension: int, hybrid: bool = False) -> None:
        """Crée la collection si besoin ; refuse une collection existante incompatible."""
        if self.client.collection_exists(name):
            params = self.client.get_collection(name).config.params
            vectors = params.vectors
            if hybrid:
                ok = isinstance(vectors, dict) and DENSE in vectors and SPARSE in (params.sparse_vectors or {})
                size = vectors[DENSE].size if ok else None
            else:
                ok = isinstance(vectors, models.VectorParams)
                size = vectors.size if ok else None
            if not ok or size != dimension:
                raise ValueError(
                    f"La collection {name} n'a pas le schéma attendu (hybride={hybrid}, dimension {dimension}). "
                    "Changez MAINTIDOC_INDEX_VERSION pour créer un nouvel index.")
            return
        dense = models.VectorParams(size=dimension, distance=models.Distance.COSINE)
        if hybrid:
            self.client.create_collection(
                name, vectors_config={DENSE: dense},
                # IDF calculé par Qdrant sur la collection : la partie « rareté » de BM25.
                sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)})
        else:
            self.client.create_collection(name, vectors_config=dense)
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
               sparse_vectors: Sequence[models.SparseVector] | None = None, batch_size: int = 64) -> int:
        if len(elements) != len(vectors) or (sparse_vectors is not None and len(sparse_vectors) != len(vectors)):
            raise ValueError("Autant de vecteurs que d'éléments sont attendus.")
        points = []
        for i, (el, vec) in enumerate(zip(elements, vectors)):
            dense = list(map(float, vec))
            if sparse_vectors is None:
                vector = dense
            else:
                vector = {DENSE: dense}
                if sparse_vectors[i].indices:  # un passage sans terme utile n'a pas de vecteur BM25
                    vector[SPARSE] = sparse_vectors[i]
            points.append(models.PointStruct(
                id=el.point_id(), vector=vector,
                payload={**el.model_dump(mode="json"), "citation_ref": el.citation_ref}))
        for start in range(0, len(points), batch_size):
            self.client.upsert(collection, points=points[start:start + batch_size], wait=True)
        return len(points)

    def count(self, collection: str, document_id: str | None = None) -> int:
        if not self.client.collection_exists(collection):
            return 0
        flt = _match("document_id", document_id) if document_id else None
        return self.client.count(collection, count_filter=flt, exact=True).count

    # --------------------------------------------------------------- recherche
    def search(self, collection: str, vector: Sequence[float] | models.SparseVector, user_groups: Iterable[str],
               limit: int = 10, equipment_type: str | None = None, modalities: Iterable[str] | None = None,
               using: str | None = None, source: str = "text") -> list[SearchHit]:
        """Recherche sur un seul vecteur (dense, BM25 ou CLIP), filtrée par droits d'accès."""
        query = vector if isinstance(vector, models.SparseVector) else list(map(float, vector))
        response = self.client.query_points(
            collection, query=query, using=using, limit=limit, with_payload=True,
            query_filter=access_filter(user_groups, equipment_type, modalities))
        return _hits(response.points, source)

    def hybrid_search(self, collection: str, dense: Sequence[float], sparse: models.SparseVector,
                      user_groups: Iterable[str], limit: int = 10, equipment_type: str | None = None,
                      modalities: Iterable[str] | None = None, depth: int | None = None) -> list[SearchHit]:
        """Dense + BM25 dans Qdrant, fusionnés par Reciprocal Rank Fusion (RRF).

        Chaque sous-recherche (prefetch) reçoit le filtre d'accès : la fusion ne voit
        que des candidats autorisés.
        """
        flt = access_filter(user_groups, equipment_type, modalities)
        depth = depth or limit * 2
        response = self.client.query_points(
            collection,
            prefetch=[
                models.Prefetch(query=list(map(float, dense)), using=DENSE, limit=depth, filter=flt),
                models.Prefetch(query=sparse, using=SPARSE, limit=depth, filter=flt),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            query_filter=flt, limit=limit, with_payload=True)
        return _hits(response.points, "text")


def _hits(points, source: str) -> list[SearchHit]:
    return [SearchHit(rank=i, score=p.score, retrieval_score=p.score, ref=p.payload.get("citation_ref", ""),
                      payload=p.payload, source=source)
            for i, p in enumerate(points, 1)]


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