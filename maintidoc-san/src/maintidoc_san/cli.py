"""Interface en ligne de commande : ingest | search | evaluate | verify | status.

Lancement : python -m maintidoc_san <commande> (ou maintidoc-san <commande>).
"""
from __future__ import annotations

import logging
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .config import get_settings

app = typer.Typer(help="MaintiDoc SAN — recherche multimodale (étape 2).", no_args_is_help=True)
console = Console()

GroupsOpt = Annotated[str, typer.Option(help="Groupes de l'utilisateur, séparés par des virgules.")]


def _components(with_reranker: bool = False):
    from .embeddings import BGEM3Embedder, CLIPEmbedder
    from .vector_store import VectorStore, make_client

    s = get_settings()
    store = VectorStore(make_client(s), s)
    text = BGEM3Embedder(s.text_embedding_model, s.embedding_device, s.embedding_batch_size)
    image = CLIPEmbedder(s.image_embedding_model, s.visual_query_model, s.embedding_device,
                         s.embedding_batch_size)
    reranker = None
    if with_reranker and s.rerank_enabled:
        from .reranker import CrossEncoderReranker
        reranker = CrossEncoderReranker(s.reranker_model, s.embedding_device)
    return s, store, text, image, reranker


def _groups(groups: str) -> list[str]:
    return [g for g in groups.split(",") if g.strip()]


def _retriever():
    from .search import Retriever

    s, store, text, image, reranker = _components(with_reranker=True)
    return s, Retriever(store, text, image, s, reranker), reranker


@app.command()
def ingest(force: Annotated[bool, typer.Option(help="Réindexer même si rien n'a changé.")] = False,
           prune: Annotated[bool, typer.Option(help="Supprimer les documents retirés du corpus.")] = False):
    """Indexe (ou met à jour) les documents déclarés dans config/corpus.yaml."""
    from .ingest import Ingestor, load_corpus

    s, store, text, image, _ = _components()
    specs = load_corpus(s.corpus_file)
    reports = Ingestor(store, text, image, s).ingest_corpus(specs, force=force, remove_missing=prune)
    table = Table("Document", "Version", "Statut", "Pages", "Passages", "Images", "Durée (s)")
    for r in reports:
        table.add_row(r.document_id, r.version, r.status, str(r.pages), str(r.text_points),
                      str(r.visual_points), str(r.seconds))
    console.print(table)
    if any(r.status == "error" for r in reports):
        raise typer.Exit(code=1)


@app.command()
def search(question: str, groups: GroupsOpt = "technicien",
           mode: Annotated[str, typer.Option(help="dense | hybrid | full (hybride + reranker).")] = "full",
           equipment: Annotated[str | None, typer.Option(help="Filtrer par type d'équipement.")] = None,
           candidates: Annotated[int, typer.Option(help="Nombre de candidats affichés.")] = 8):
    """Recherche multimodale : candidats classés puis preuves retenues."""
    s, retriever, _ = _retriever()
    result = retriever.retrieve(question, _groups(groups), mode=mode, equipment_type=equipment)

    table = Table("Rang", "Score", "Rerank", "Référence", "Type", "Figure", "Extrait",
                  title=f"Candidats (mode {mode})")
    for h in result.candidates[:candidates]:
        p = h.payload
        rerank = f"{h.rerank_score:.3f}" if h.rerank_score is not None else "-"
        excerpt = h.evidence_text.replace("\n", " ")[:80]
        table.add_row(str(h.rank), f"{h.score:.3f}", rerank, h.ref, p.get("modality", ""),
                      p.get("figure_id") or "", excerpt)
    console.print(table)

    if not result.evidence:
        console.print("[yellow]Aucune preuve suffisante : l'assistant devra s'abstenir ou demander des précisions.")
        return
    console.print("[bold]Preuves retenues :")
    for h in result.evidence:
        p = h.payload
        image = f"  [image : {p['image_path']}]" if p.get("image_path") else ""
        console.print(f"  {h.rank}. [cyan]{h.ref}[/cyan] (p.{p.get('page')}, section {p.get('section')}){image}")
    console.print(f"[dim]Temps : " + ", ".join(f"{k}={v:.2f}s" for k, v in result.timings.items()))


@app.command()
def evaluate(groups: GroupsOpt = "technicien",
             details: Annotated[bool, typer.Option(help="Afficher le détail par question.")] = False):
    """Compare dense / hybride / hybride + reranker sur le jeu de questions du manuel."""
    from .evaluation import averages, evaluate as run_eval, load_questions

    s, retriever, reranker = _retriever()
    questions = load_questions(s.eval_file)
    modes = ["dense", "hybrid"] + (["full"] if reranker else [])
    summary = Table("Mode", "Recall@5", "MRR@10", "nDCG@5", title=f"{len(questions)} questions")
    for mode in modes:
        results = run_eval(retriever, questions, _groups(groups), mode)
        avg = averages(results)
        summary.add_row(mode, *(f"{avg[k]:.3f}" for k in ("Recall@5", "MRR@10", "nDCG@5")))
        if details:
            detail = Table("Question", "Recall@5", "MRR@10", "Top 3", title=f"Détail — {mode}")
            for r in results:
                detail.add_row(r.id, f"{r.recall_at_5:.2f}", f"{r.mrr_at_10:.2f}", "\n".join(r.top_refs))
            console.print(detail)
    console.print(summary)


@app.command()
def verify(question: str, answer: str, groups: GroupsOpt = "technicien"):
    """Vérifie les citations [DOC:pN:ELEMENT] d'une réponse par rapport aux preuves récupérées."""
    from .evidence import validate_citations

    s, retriever, reranker = _retriever()
    result = retriever.retrieve(question, _groups(groups))
    report = validate_citations(answer, result.evidence, reranker, s.citation_support_threshold)
    table = Table("Phrase", "Référence", "Statut", "Soutien")
    for c in report.checks:
        score = f"{c.support_score:.2f}" if c.support_score is not None else "-"
        table.add_row(c.sentence[:70], c.ref or "-", c.status, score)
    console.print(table)
    console.print(f"Validité : {report.validity_rate:.0%} — soutien : {report.support_rate:.0%} — "
                  f"{'[green]OK' if report.ok else '[red]à corriger'}")


@app.command()
def status():
    """Affiche l'état des collections Qdrant."""
    from .vector_store import VectorStore, make_client

    s = get_settings()
    store = VectorStore(make_client(s), s)
    table = Table("Collection", "Points")
    for name in (s.text_collection, s.visual_collection):
        table.add_row(name, str(store.count(name)))
    console.print(table)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    app()