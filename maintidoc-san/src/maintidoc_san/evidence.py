"""Sélection des preuves et validation des citations.

Sélection : on ne transmet au modèle génératif que des preuves pertinentes, variées et
de taille maîtrisée. S'il n'y en a aucune, l'assistant devra s'abstenir (étape 3).

Validation : une citation est acceptée si (1) elle pointe vers une preuve réellement
récupérée pour cet utilisateur et (2) cette preuve soutient la phrase qui la cite.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable

from .config import Settings
from .reranker import Reranker
from .schemas import CITATION_RE, CitationCheck, CitationReport, SearchHit
from .sparse import tokenize

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-Ý0-9«\"(\[])")
MIN_CLAIM_WORDS = 4  # phrases plus courtes (« Oui. ») : pas d'affirmation à vérifier


def select_evidence(candidates: list[SearchHit], settings: Settings) -> list[SearchHit]:
    """Garde les meilleures preuves : seuil de pertinence, diversité des pages, budget de texte."""
    if not candidates:
        return []
    reranked = candidates[0].rerank_score is not None
    best = max((c.rerank_score or 0.0) for c in candidates)
    selected: list[SearchHit] = []
    per_page: Counter = Counter()
    tables: set[tuple] = set()
    chars = 0
    for hit in candidates:
        if reranked and (hit.rerank_score < settings.min_rerank_score
                         or hit.rerank_score < best * settings.relative_rerank_score):
            continue  # jugé peu pertinent par le reranker : pas une preuve
        p = hit.payload
        page_key = (p.get("document_id"), p.get("page"))
        if per_page[page_key] >= settings.max_evidence_per_page:
            continue
        table_id = p.get("table_id")
        if table_id and (p.get("document_id"), table_id) in tables:
            continue  # ligne d'un tableau déjà retenu (ou l'inverse) : redondant
        size = len(hit.evidence_text)
        if selected and chars + size > settings.evidence_char_budget:
            continue
        selected.append(hit)
        per_page[page_key] += 1
        if table_id:
            tables.add((p.get("document_id"), table_id))
        chars += size
        if len(selected) >= settings.max_evidence:
            break
    return [h.model_copy(update={"rank": i}) for i, h in enumerate(selected, 1)]


def lexical_support(claim: str, evidence: str) -> float:
    """Part des termes de l'affirmation présents dans la preuve (repli sans reranker)."""
    claim_terms = set(tokenize(claim))
    if not claim_terms:
        return 0.0
    return len(claim_terms & set(tokenize(evidence))) / len(claim_terms)


def validate_citations(answer: str, evidence: list[SearchHit], reranker: Reranker | None = None,
                       threshold: float = 0.3) -> CitationReport:
    """Vérifie chaque référence [DOC:pN:ELEMENT] de la réponse."""
    known: dict[str, str] = {}
    for hit in evidence:
        for ref in (hit.ref, *hit.related_refs):
            known[ref] = hit.evidence_text
    if reranker is not None:
        def support(claim: str, text: str) -> float:
            return float(reranker.score(claim, [text[:1500]])[0])
    else:
        support: Callable[[str, str], float] = lexical_support

    checks: list[CitationCheck] = []
    for sentence in (s.strip() for s in SENTENCE_SPLIT.split(answer.strip())):
        if not sentence:
            continue
        refs = ["{}:p{}:{}".format(*m) for m in CITATION_RE.findall(sentence)]
        claim = re.sub(r"\[[^\]]*\]", "", CITATION_RE.sub("", sentence)).strip(" ,;")
        if not refs:
            if len(claim.split()) >= MIN_CLAIM_WORDS:
                checks.append(CitationCheck(sentence=sentence, ref=None, status="uncited"))
            continue
        for ref in refs:
            if ref not in known:
                checks.append(CitationCheck(sentence=sentence, ref=ref, status="unknown_ref"))
                continue
            score = support(claim, known[ref])
            checks.append(CitationCheck(sentence=sentence, ref=ref, support_score=round(score, 3),
                                        status="supported" if score >= threshold else "unsupported"))
    return CitationReport(checks=checks)