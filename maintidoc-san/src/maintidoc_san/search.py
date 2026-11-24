"""Recherche multimodale de l'étape 2.

Pipeline (cahier des charges, section 4.2) :
1. droits de l'utilisateur -> filtre Qdrant appliqué à chaque recherche ;
2. recherche textuelle hybride (dense BGE-M3 + BM25, fusion RRF dans Qdrant)
   et recherche visuelle (CLIP) sur le corpus autorisé ;
3. fusion texte + images par Reciprocal Rank Fusion ; une figure trouvée par sa légende,
   son texte OCR et son image ne compte qu'une fois ;
4. reclassement des candidats (passages, lignes de tableaux, légendes + OCR) par le reranker ;
5. sélection des preuves transmises au futur modèle génératif (étape 3).
"""
from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass, field

from qdrant_client import models

from .config import Settings
from .embeddings import Embedder, ImageEmbedder
from .evidence import select_evidence
from .reranker import Reranker
from .schemas import SearchHit
from .sparse import query_vector
from .vector_store import DENSE, SPARSE, VectorStore, access_filter

MODES = ("dense", "hybrid", "full")  # full = hybride + reranker
FIGURE_MODALITIES = {"figure_caption", "image_ocr", "schema", "illustration"}


@dataclass
class RetrievalResult:
    question: str
    mode: str
    candidates: list[SearchHit]   # tous les candidats classés
    evidence: list[SearchHit]     # preuves retenues
    timings: dict[str, float] = field(default_factory=dict)


def unit_key(hit: SearchHit) -> str:
    """Clé de regroupement : une figure = une seule unité, quels que soient ses éléments."""
    p = hit.payload
    if p.get("figure_id") and p.get("modality") in FIGURE_MODALITIES:
        return f"{p['document_id']}:{p['figure_id']}"
    return hit.ref


def rrf_fuse(result_lists: list[list[SearchHit]], k: int = 60,
             weights: list[float] | None = None) -> list[SearchHit]:
    """Reciprocal Rank Fusion : score = somme de poids / (k + rang) sur chaque liste.

    Seul le rang compte : on peut donc fusionner des scores non comparables
    (cosinus BGE-M3, BM25, cosinus CLIP). Le poids réduit l'influence d'une liste moins fiable.
    """
    weights = weights or [1.0] * len(result_lists)
    scores: dict[str, float] = {}
    best: dict[str, SearchHit] = {}
    related: dict[str, list[str]] = {}
    for hits, weight in zip(result_lists, weights):
        for hit in hits:
            key = unit_key(hit)
            scores[key] = scores.get(key, 0.0) + weight / (k + hit.rank)
            refs = related.setdefault(key, [])
            if hit.ref not in refs:
                refs.append(hit.ref)
            # Pour une figure, on garde de préférence l'élément visuel (image + légende + OCR).
            if key not in best or (hit.source == "visual" and best[key].source != "visual"):
                best[key] = hit
    fused = []
    for rank, key in enumerate(sorted(scores, key=scores.get, reverse=True), 1):
        hit = best[key]
        fused.append(hit.model_copy(update={
            "rank": rank, "score": scores[key],
            "related_refs": [r for r in related[key] if r != hit.ref]}))
    return fused


class Retriever:
    def __init__(self, store: VectorStore, text_embedder: Embedder, image_embedder: ImageEmbedder,
                 settings: Settings, reranker: Reranker | None = None):
        self.store, self.text_embedder, self.image_embedder = store, text_embedder, image_embedder
        self.settings, self.reranker = settings, reranker

    # ----------------------------------------------------------- briques de base
    def search_text(self, question: str, user_groups: Iterable[str], limit: int = 20,
                    equipment_type: str | None = None, hybrid: bool = True) -> list[SearchHit]:
        s = self.settings
        dense = self.text_embedder.embed_texts([question])[0]
        sparse = query_vector(question)
        if hybrid and sparse.indices:
            return self.store.hybrid_search(s.text_collection, dense, sparse, user_groups, limit, equipment_type,
                                            depth=s.candidates_per_source)
        return self.store.search(s.text_collection, dense, user_groups, limit, equipment_type, using=DENSE)

    def search_lexical(self, question: str, user_groups: Iterable[str], limit: int = 20) -> list[SearchHit]:
        sparse = query_vector(question)
        if not sparse.indices:
            return []
        return self.store.search(self.settings.text_collection, sparse, user_groups, limit, using=SPARSE)

    def search_visual(self, question: str, user_groups: Iterable[str], limit: int = 10,
                      equipment_type: str | None = None, schemas_only: bool = True) -> list[SearchHit]:
        vector = self.image_embedder.embed_queries([question])[0]
        modalities = ["schema"] if schemas_only else None
        return self.store.search(self.settings.visual_collection, vector, user_groups, limit,
                                 equipment_type, modalities, source="visual")

    # -------------------------------------------------------------- pipeline
    def retrieve(self, question: str, user_groups: Iterable[str], mode: str = "full",
                 equipment_type: str | None = None, use_visual: bool = True) -> RetrievalResult:
        if mode not in MODES:
            raise ValueError(f"Mode inconnu : {mode} (attendu : {', '.join(MODES)})")
        s, groups = self.settings, list(user_groups)
        access_filter(groups)  # refus immédiat si l'utilisateur n'a aucun groupe
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        lists = [self.search_text(question, groups, s.candidates_per_source, equipment_type,
                                  hybrid=mode != "dense")]
        weights = [1.0]
        if use_visual:
            lists.append(self.search_visual(question, groups, max(5, s.candidates_per_source // 2), equipment_type))
            weights.append(s.visual_weight)
        timings["retrieval_s"] = time.perf_counter() - t0

        candidates = rrf_fuse(lists, k=s.rrf_k, weights=weights)
        candidates = self._attach_figures(candidates, groups)

        if mode == "full" and s.rerank_enabled and self.reranker is not None and candidates:
            t0 = time.perf_counter()
            candidates = self.rerank(question, candidates)
            timings["rerank_s"] = time.perf_counter() - t0

        evidence = select_evidence(candidates, s)
        return RetrievalResult(question, mode, candidates, evidence, timings)

    def rerank(self, question: str, candidates: list[SearchHit]) -> list[SearchHit]:
        """Score final = reranker + part du score de fusion (normalisé entre 0 et 1).

        Le reranker ne lit que du texte : pour une figure, seulement légende + OCR.
        Garder une part du classement initial évite d'écarter un schéma bien trouvé par CLIP.
        """
        texts = [c.evidence_text[:1500] for c in candidates]
        scores = self.reranker.score(question, texts)
        top_fused = max(c.score for c in candidates) or 1.0
        blend = self.settings.rerank_blend
        rescored = [c.model_copy(update={"rerank_score": float(sc),
                                         "score": (1 - blend) * float(sc) + blend * c.score / top_fused})
                    for c, sc in zip(candidates, scores)]
        rescored.sort(key=lambda c: c.score, reverse=True)
        return [c.model_copy(update={"rank": i}) for i, c in enumerate(rescored, 1)]

    def _attach_figures(self, candidates: list[SearchHit], groups: list[str]) -> list[SearchHit]:
        """Figure trouvée seulement par sa légende ou son OCR : on y joint l'image (pour le VLM).

        La recherche passe par le même filtre d'accès : une image non autorisée n'est jamais jointe.
        """
        out = []
        for hit in candidates:
            p = hit.payload
            if hit.source != "visual" and p.get("figure_id") and p.get("modality") in FIGURE_MODALITIES:
                flt = access_filter(groups)
                flt.must.extend([
                    models.FieldCondition(key="document_id", match=models.MatchValue(value=p["document_id"])),
                    models.FieldCondition(key="figure_id", match=models.MatchValue(value=p["figure_id"]))])
                points, _ = self.store.client.scroll(self.settings.visual_collection, scroll_filter=flt,
                                                     limit=1, with_payload=True, with_vectors=False)
                if points:
                    visual = points[0].payload
                    hit = hit.model_copy(update={
                        "source": "visual", "payload": visual, "ref": visual["citation_ref"],
                        "related_refs": [r for r in [hit.ref, *hit.related_refs] if r != visual["citation_ref"]]})
            out.append(hit)
        return out