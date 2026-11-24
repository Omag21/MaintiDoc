"""Mesure de la recherche sur le jeu de questions du manuel : Recall@5, MRR@10, nDCG@5.

Sert à vérifier que l'étape 2 améliore réellement la recherche, en comparant :
dense (comme l'étape 1) -> hybride (dense + BM25) -> complet (hybride + reranker).
L'évaluation complète (réponses, citations, injections) viendra à l'étape 6.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

from .schemas import SearchHit
from .search import Retriever
from .sparse import normalize


@dataclass
class EvalQuestion:
    id: str
    question: str
    sources: list[dict]


@dataclass
class QuestionResult:
    id: str
    recall_at_5: float
    mrr_at_10: float
    ndcg_at_5: float
    top_refs: list[str]


def load_questions(path: Path) -> list[EvalQuestion]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return [EvalQuestion(q["id"], q["question"], q["sources"]) for q in raw.get("questions", [])]


def matches(hit: SearchHit, source: dict) -> bool:
    """Le résultat remplit-il toutes les conditions de la source attendue ?"""
    p = hit.payload
    for key, expected in source.items():
        if key == "figure_id" and p.get("figure_id") != expected:
            return False
        if key == "section" and str(p.get("section")) != str(expected):
            return False
        if key == "page" and p.get("page") != int(expected):
            return False
        if key == "contains" and normalize(str(expected)) not in normalize(hit.evidence_text):
            return False
    return True


def score_ranking(hits: list[SearchHit], sources: list[dict]) -> tuple[float, float, float]:
    # Rang (1-based) auquel chaque résultat trouve une source pas encore trouvée.
    found: set[int] = set()
    gains: list[int] = []
    first_relevant = None
    for rank, hit in enumerate(hits[:10], 1):
        new = [i for i, src in enumerate(sources) if i not in found and matches(hit, src)]
        relevant = any(matches(hit, src) for src in sources)
        if relevant and first_relevant is None:
            first_relevant = rank
        gains.append(1 if new else 0)
        found.update(new[:1])
    found_at_5 = set()
    for hit in hits[:5]:
        found_at_5.update(i for i, src in enumerate(sources) if matches(hit, src))
    recall = len(found_at_5) / len(sources)
    mrr = 1.0 / first_relevant if first_relevant else 0.0
    dcg = sum(g / math.log2(r + 1) for r, g in enumerate(gains[:5], 1))
    ideal = sum(1 / math.log2(r + 1) for r in range(1, min(len(sources), 5) + 1))
    return recall, mrr, dcg / ideal


def evaluate(retriever: Retriever, questions: list[EvalQuestion], groups: list[str],
             mode: str) -> list[QuestionResult]:
    results = []
    for q in questions:
        hits = retriever.retrieve(q.question, groups, mode=mode).candidates
        recall, mrr, ndcg = score_ranking(hits, q.sources)
        results.append(QuestionResult(q.id, recall, mrr, ndcg, [h.ref for h in hits[:3]]))
    return results


def averages(results: list[QuestionResult]) -> dict[str, float]:
    n = len(results) or 1
    return {"Recall@5": sum(r.recall_at_5 for r in results) / n,
            "MRR@10": sum(r.mrr_at_10 for r in results) / n,
            "nDCG@5": sum(r.ndcg_at_5 for r in results) / n}