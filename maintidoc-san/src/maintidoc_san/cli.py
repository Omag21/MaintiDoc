"""Interface en ligne de commande : maintidoc-san ingest | search | status."""
from __future__ import annotations

import logging
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from .config import get_settings

app = typer.Typer(help="MaintiDoc SAN — index vectoriel Qdrant (étape 1).", no_args_is_help=True)
console = Console()


def _components():
    from .embeddings import BGEM3Embedder, CLIPEmbedder
    from .vector_store import VectorStore, make_client

    s = get_settings()
    store = VectorStore(make_client(s), s)
    text = BGEM3Embedder(s.text_embedding_model, s.embedding_device, s.embedding_batch_size)
    image = CLIPEmbedder(s.image_embedding_model, s.visual_query_model, s.embedding_device,
                         s.embedding_batch_size)
    return s, store, text, image


@app.command()
def ingest(force: Annotated[bool, typer.Option(help="Réindexer même si rien n'a changé.")] = False,
           prune: Annotated[bool, typer.Option(help="Supprimer les documents retirés du corpus.")] = False):
    """Indexe (ou met à jour) les documents déclarés dans config/corpus.yaml."""
    from .ingest import Ingestor, load_corpus

    s, store, text, image = _components()
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
def search(question: str,
           groups: Annotated[str, typer.Option(help="Groupes de l'utilisateur, séparés par des virgules.")] = "technicien",
           limit: int = 5,
           equipment: Annotated[str | None, typer.Option(help="Filtrer par type d'équipement.")] = None):
    """Recherche texte + visuelle (sans fusion) pour contrôler l'index."""
    from .search import Retriever

    s, store, text, image = _components()
    retriever = Retriever(store, text, image, s)
    user_groups = [g for g in groups.split(",") if g.strip()]
    for title, hits in (("Texte (BGE-M3)", retriever.search_text(question, user_groups, limit, equipment)),
                        ("Schémas (CLIP)", retriever.search_visual(question, user_groups, limit, equipment))):
        table = Table("Rang", "Score", "Référence", "Type", "Figure", "Extrait", title=title)
        for h in hits:
            p = h.payload
            excerpt = (p.get("text") or p.get("caption") or "").replace("\n", " ")[:90]
            table.add_row(str(h.rank), f"{h.score:.3f}", h.ref, p.get("modality", ""),
                          p.get("figure_id") or "", excerpt)
        console.print(table)


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
    app()