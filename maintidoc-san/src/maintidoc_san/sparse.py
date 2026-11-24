"""Vecteurs épars de type BM25 pour la recherche lexicale dans Qdrant.

Pourquoi : les embeddings denses captent le sens mais ratent souvent les codes exacts
(AL-04, DIAG-FC-01, B03, INC-117). BM25 retrouve ces correspondances mot pour mot.

Répartition du calcul BM25 :
- ici, côté client : la partie « fréquence du terme » saturée et normalisée par la longueur ;
- dans Qdrant (modifier=IDF) : la partie « rareté du terme dans la collection » (IDF),
  recalculée automatiquement quand des documents sont ajoutés ou supprimés.
"""
from __future__ import annotations

import re
import unicodedata
import zlib
from collections import Counter

from qdrant_client import models

K1, B = 1.2, 0.75  # paramètres BM25 classiques

STOPWORDS = frozenset("""
a au aux avec ce ces dans de des du elle en et eux il ils je la le les leur lui ma mais me meme mes
moi mon ne nos notre nous on ou par pas pour qu que qui sa se ses son sur ta te tes toi ton tu un une
vos votre vous c d j l m n s t y est sont ete etre avoir a-t-il quel quelle quels quelles comment
quoi dont cette cet faut peut doit
""".split())

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")


def normalize(text: str) -> str:
    """Minuscules sans accents : « Contrôleur » et « controleur » deviennent identiques."""
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def tokenize(text: str) -> list[str]:
    """Découpe en termes ; un code composé est gardé entier ET découpé.

    « DIAG-FC-01 » -> diag-fc-01, diag, fc, 01 : la requête « DIAG-FC-01 » trouve le code
    exact (terme rare, fort IDF) et la requête « diag » trouve aussi le passage.
    """
    tokens: list[str] = []
    for tok in TOKEN_RE.findall(normalize(text)):
        if tok in STOPWORDS:
            continue
        tokens.append(tok)
        if "-" in tok or "_" in tok:
            tokens.extend(p for p in re.split(r"[-_]", tok) if p and p not in STOPWORDS)
    return tokens


def term_index(term: str) -> int:
    """Indice stable du terme (CRC32) : identique d'un processus et d'une machine à l'autre.

    (hash() de Python change à chaque lancement : inutilisable pour un index persistant.)
    """
    return zlib.crc32(term.encode("utf-8")) & 0x7FFFFFFF


def _to_sparse(weights: dict[int, float]) -> models.SparseVector:
    indices = sorted(weights)
    return models.SparseVector(indices=indices, values=[float(weights[i]) for i in indices])


def document_vector(text: str, avg_doc_tokens: float = 60.0) -> models.SparseVector:
    """Poids BM25 (sans IDF) de chaque terme du passage."""
    tokens = tokenize(text)
    if not tokens:
        return models.SparseVector(indices=[], values=[])
    length_norm = K1 * (1 - B + B * len(tokens) / avg_doc_tokens)
    weights: dict[int, float] = {}
    for term, tf in Counter(tokens).items():
        idx = term_index(term)
        weights[idx] = weights.get(idx, 0.0) + tf * (K1 + 1) / (tf + length_norm)
    return _to_sparse(weights)


def query_vector(text: str) -> models.SparseVector:
    """Requête : chaque terme distinct vaut 1 ; Qdrant multiplie par son IDF."""
    return _to_sparse({term_index(t): 1.0 for t in set(tokenize(text))})