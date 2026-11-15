"""Extraction PyMuPDF : passages textuels, tableaux, légendes et figures avec provenance.

Différences avec le notebook :
- découpage en passages (blocs PyMuPDF regroupés) au lieu d'un vecteur par page ;
- figure_id détecté depuis la légende « FIG-xx : ... » la plus proche, et non par position de page ;
- chaque élément porte la version du document, le type d'équipement et les groupes d'accès.
"""
from __future__ import annotations

import hashlib
import io
import re

import pymupdf
from PIL import Image

from .config import Settings
from .schemas import DocumentSpec, ExtractedDocument, Modality, TextElement, VisualElement

CAPTION_RE = re.compile(r"^\s*(FIG-\d{2})\s*:\s*(.+)", re.DOTALL)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _clean(text: str) -> str:
    text = text.replace("\u00b7", "'")  # apostrophe mal encodée dans le manuel (D·UN)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def _chunk_blocks(blocks: list[tuple], max_chars: int) -> list[tuple[str, list[float]]]:
    """Regroupe les blocs texte consécutifs d'une page en passages de taille bornée."""
    chunks: list[tuple[str, list[float]]] = []
    buffer: list[str] = []
    bbox: list[float] | None = None
    for x0, y0, x1, y1, text, *_ in blocks:
        text = _clean(text)
        if not text:
            continue
        if buffer and sum(len(b) for b in buffer) + len(text) > max_chars:
            chunks.append(("\n".join(buffer), bbox))
            buffer, bbox = [], None
        buffer.append(text)
        bbox = [x0, y0, x1, y1] if bbox is None else [
            min(bbox[0], x0), min(bbox[1], y0), max(bbox[2], x1), max(bbox[3], y1)]
    if buffer:
        chunks.append(("\n".join(buffer), bbox))
    return chunks


def _table_to_markdown(rows: list[list[str]]) -> str:
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header, *body = rows
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


def _captions(page: pymupdf.Page) -> list[tuple[str, str, pymupdf.Rect]]:
    found = []
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks", sort=True):
        match = CAPTION_RE.match(text)
        if match:
            found.append((match.group(1), _clean(match.group(2)), pymupdf.Rect(x0, y0, x1, y1)))
    return found


def _nearest_caption(rect: pymupdf.Rect, captions):
    if not captions:
        return None
    def distance(c):
        crect = c[2]
        return min(abs(crect.y0 - rect.y1), abs(rect.y0 - crect.y1))
    return min(captions, key=distance)


def extract_document(spec: DocumentSpec, settings: Settings, pdf_bytes: bytes | None = None) -> ExtractedDocument:
    pdf_path = spec.path  # déjà résolu par load_corpus()
    data = pdf_bytes if pdf_bytes is not None else pdf_path.read_bytes()
    digest = sha256_bytes(data)
    assets_dir = settings.assets_dir / spec.document_id / spec.version
    assets_dir.mkdir(parents=True, exist_ok=True)

    common = dict(
        document_id=spec.document_id,
        document_title=spec.title,
        document_version=spec.version,
        document_sha256=digest,
        source_file=pdf_path.name,
        equipment_type=spec.equipment_type,
        equipment_family=spec.equipment_family,
        access_groups=spec.access_groups,
        language=spec.language,
        index_version=settings.index_version,
    )
    texts: list[TextElement] = []
    visuals: list[VisualElement] = []

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        page_count = len(doc)
        for page in doc:
            p = page.number + 1

            try:
                tables = page.find_tables().tables
            except Exception:  # le détecteur peut échouer sur certains PDF
                tables = []

            # 1. Passages textuels. Les zones de tableaux sont conservées ici car leur texte
            #    de bloc est mieux espacé que les cellules de find_tables() (nettoyage : étape 2).
            blocks = [b for b in page.get_text("blocks", sort=True) if b[6] == 0]  # 0 = bloc texte
            for i, (chunk, bbox) in enumerate(_chunk_blocks(blocks, settings.chunk_max_chars), 1):
                texts.append(TextElement(**common, page=p, element_id=f"p{p:02d}_chunk_{i:02d}",
                                         modality=Modality.TEXT, text=chunk, bbox=bbox, chunk_index=i,
                                         embedding_model=settings.text_embedding_model))

            # 2. Tableaux en Markdown (structure préservée pour le contexte RAG)
            for k, table in enumerate(tables, 1):
                rows = [[_clean(str(c or "")) for c in (row or [])] for row in table.extract()]
                rows = [r for r in rows if any(r)]
                if len(rows) < 2:
                    continue
                texts.append(TextElement(**common, page=p, element_id=f"p{p:02d}_table_{k:02d}",
                                         modality=Modality.TABLE, text=_table_to_markdown(rows),
                                         bbox=list(table.bbox), embedding_model=settings.text_embedding_model))

            # 3. Figures et légendes
            captions = _captions(page)
            for fig_id, caption, crect in captions:
                texts.append(TextElement(**common, page=p, element_id=f"p{p:02d}_caption_{fig_id}",
                                         modality=Modality.FIGURE_CAPTION, text=f"{fig_id} : {caption}",
                                         bbox=list(crect), figure_id=fig_id,
                                         embedding_model=settings.text_embedding_model))

            for k, info in enumerate(page.get_images(full=True), 1):
                xref = info[0]
                payload = doc.extract_image(xref)
                if not payload:
                    continue
                raw = payload["image"]
                with Image.open(io.BytesIO(raw)) as im:
                    width, height = im.size
                    if min(width, height) < settings.min_image_side:
                        continue  # icônes et décorations
                    filename = f"p{p:02d}_image_{k:02d}.png"
                    im.convert("RGB").save(assets_dir / filename, format="PNG")
                rects = page.get_image_rects(xref)
                rect = rects[0] if rects else None
                nearest = _nearest_caption(rect, captions) if rect is not None else None
                fig_id, caption = (nearest[0], nearest[1]) if nearest else (None, None)
                visuals.append(VisualElement(
                    **common, page=p, element_id=f"p{p:02d}_image_{k:02d}",
                    modality=Modality.SCHEMA if fig_id else Modality.ILLUSTRATION,
                    figure_id=fig_id, caption=caption,
                    image_path=(assets_dir / filename).relative_to(settings.data_dir).as_posix(),
                    image_sha256=sha256_bytes(raw), width=width, height=height,
                    bbox=list(rect) if rect is not None else None,
                    embedding_model=settings.image_embedding_model))

    return ExtractedDocument(spec=spec, sha256=digest, page_count=page_count, texts=texts, visuals=visuals)