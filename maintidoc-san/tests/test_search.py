"""Tests de l'étape 2 : BM25, recherche hybride, fusion RRF, reranking, preuves, citations."""
from __future__ import annotations

import pytest

from maintidoc_san.evaluation import load_questions, score_ranking
from maintidoc_san.evidence import select_evidence, validate_citations
from maintidoc_san.ingest import load_corpus
from maintidoc_san.schemas import Citation, SearchHit
from maintidoc_san.search import Retriever, rrf_fuse
from maintidoc_san.sparse import document_vector, query_vector, tokenize

from conftest import FakeImage, FakeReranker, FakeText


@pytest.fixture()
def retriever(env):
    settings, store, ingestor = env
    ingestor.ingest_corpus(load_corpus(settings.corpus_file))
    return Retriever(store, FakeText(), FakeImage(), settings, FakeReranker())


def hit(ref, rank, source="text", **payload):
    payload = {"document_id": "D", "page": 1, "citation_ref": ref, **payload}
    return SearchHit(rank=rank, score=1.0, ref=ref, payload=payload, source=source)


# ---------------------------------------------------------------------- BM25
def test_tokenize_keeps_codes_and_ignores_accents():
    tokens = tokenize("Vérifier DIAG-FC-01 sur le Contrôleur")
    assert {"verifier", "diag-fc-01", "diag", "fc", "01", "controleur"} <= set(tokens)
    assert "le" not in tokens and "sur" not in tokens


def test_sparse_vectors_are_stable():
    assert query_vector("AL-04 capacite") == query_vector("capacité AL-04")
    vec = document_vector("AL-04 AL-04 capacite")
    assert len(vec.indices) == len(set(vec.indices)) and all(v > 0 for v in vec.values)


def test_bm25_finds_exact_code(retriever):
    # Les faux vecteurs denses sont aléatoires : seul BM25 peut trouver le code exact.
    hits = retriever.search_lexical("AL-04", ["technicien"], limit=3)
    assert "AL-04" in hits[0].payload["text"]
    hybrid = retriever.search_text("AL-04", ["technicien"], limit=5)
    assert any("AL-04" in h.payload["text"] for h in hybrid)


def test_hybrid_search_respects_access(retriever):
    assert retriever.search_text("AL-04", ["invite"], limit=5) == []
    with pytest.raises(PermissionError):
        retriever.retrieve("AL-04", [])


# ----------------------------------------------------------------------- RRF
def test_rrf_rewards_agreement_and_merges_figures():
    a = [hit("D:p1:x", 1), hit("D:p2:cap", 2, figure_id="FIG-01", modality="figure_caption")]
    b = [hit("D:p2:img", 1, source="visual", figure_id="FIG-01", modality="schema"), hit("D:p1:x", 2)]
    fused = rrf_fuse([a, b], k=60)
    assert len(fused) == 2  # la légende et l'image de FIG-01 ne forment qu'une unité
    figure = next(h for h in fused if h.payload.get("figure_id") == "FIG-01")
    assert figure.source == "visual" and figure.related_refs == ["D:p2:cap"]


def test_rrf_weight_reduces_a_list():
    a, b = [hit("D:p1:a", 1)], [hit("D:p1:b", 1, source="visual")]
    fused = rrf_fuse([a, b], weights=[1.0, 0.5])
    assert fused[0].ref == "D:p1:a"


# ------------------------------------------------------------ pipeline complet
def test_full_pipeline_returns_evidence_with_images(retriever):
    result = retriever.retrieve("Quel est le sens de AL-01 ?", ["technicien"], mode="full")
    assert result.candidates[0].rerank_score is not None
    assert result.evidence and len(result.evidence) <= retriever.settings.max_evidence
    assert any("AL-01" in h.evidence_text for h in result.evidence)
    # Toute figure candidate porte l'image d'origine (pour le futur modèle vision-langage).
    for h in result.candidates:
        if h.payload.get("figure_id"):
            assert h.source == "visual" and h.payload["image_path"]


def test_irrelevant_question_gives_no_evidence(retriever):
    result = retriever.retrieve("recette de la tarte aux pommes", ["technicien"], mode="full")
    assert result.evidence == []  # l'assistant devra s'abstenir


# ------------------------------------------------------------------- preuves
def test_evidence_selection_limits(env):
    settings, _, _ = env
    s = settings.model_copy(update={"max_evidence": 3, "max_evidence_per_page": 1})
    cands = [hit(f"D:p{p}:e{i}", i, page=p, text="x" * 50) for i, p in enumerate([1, 1, 2, 3, 4], 1)]
    for c, score in zip(cands, [0.9, 0.8, 0.7, 0.01, 0.6]):
        c.rerank_score = score
    selected = select_evidence(cands, s)
    assert [h.ref for h in selected] == ["D:p1:e1", "D:p2:e3", "D:p4:e5"]  # 1 par page, seuil, max 3


# ----------------------------------------------------------------- citations
def test_citation_validation():
    evidence = [hit("D:p5:row1", 1, text="Code : AL-01 | Signification : Chemin FC degrade")]
    good = validate_citations("AL-01 signale un chemin FC degrade [D:p5:row1].", evidence, threshold=0.5)
    assert good.ok and good.support_rate == 1.0
    wrong = validate_citations("Le ventilateur surchauffe pendant la nuit [D:p5:row1].", evidence, threshold=0.5)
    assert wrong.checks[0].status == "unsupported"
    invented = validate_citations("AL-01 signale un chemin FC degrade [D:p9:inconnu].", evidence)
    assert invented.checks[0].status == "unknown_ref" and invented.validity_rate == 0.0
    uncited = validate_citations("Le disque sera remplace automatiquement par le systeme.", evidence)
    assert uncited.checks[0].status == "uncited" and not uncited.ok


def test_citation_format():
    c = Citation.parse("SAN-MAN-001:p5:p05_table_01_row_01")
    assert (c.document_id, c.page, c.element_id) == ("SAN-MAN-001", 5, "p05_table_01_row_01")
    with pytest.raises(ValueError):
        Citation.parse("page 5")


# ---------------------------------------------------------------- évaluation
def test_ranking_metrics():
    hits = [hit("D:p1:a", 1, text="rien"), hit("D:p2:b", 2, text="voir AL-04")]
    recall, mrr, ndcg = score_ranking(hits, [{"contains": "AL-04"}])
    assert recall == 1.0 and mrr == 0.5 and 0 < ndcg < 1


def test_eval_questions_file_is_valid():
    from maintidoc_san.config import PROJECT_ROOT
    questions = load_questions(PROJECT_ROOT / "config" / "eval_questions.yaml")
    assert len(questions) == 12 and all(q.sources for q in questions)