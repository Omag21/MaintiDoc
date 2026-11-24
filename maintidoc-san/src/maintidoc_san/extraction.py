"""Extraction PyMuPDF : passages, tableaux (complets + ligne par ligne), figures et OCR.

Nouveautés de l'étape 2 :
- en-têtes / pieds de page répétés retirés des passages ;
- numéro de section détecté (« 4. Voyants... ») et ajouté aux métadonnées ;
- tableaux reconstruits à partir des mots (plus de mots collés) et indexés ligne par ligne ;
- OCR local du texte des schémas et des pages scannées ;
- sections exclues (ex. jeu de questions d'évaluation) jamais indexées.
"""
from __future__ import annotations

import hashlib
import io
import re
from collections import defaultdict
from dataclasses import dataclass

import pymupdf
from PIL import Image

from .config import Settings
from .ocr import OCREngine
from .schemas import DocumentSpec, ExtractedDocument, Modality, TextElement, VisualElement

CAPTION_RE = re.compile(r"^\s*(FIG-\d{2})\s*:\s*(.+)", re.DOTALL)
HEADING_RE = re.compile(r"^(\d{1,2})\.\s+(\S.{0,70})$")  # titre de section sur une seule ligne
MIN_OCR_CHARS = 10  # texte OCR plus court : ignoré


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _clean(text: str) -> str:
    text = text.replace("·", "'")  # apostrophe mal encodée dans le manuel (D·UN)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def _center_in(bbox, rect: pymupdf.Rect) -> bool:
    x0, y0, x1, y1 = bbox[:4]
    return rect.contains(pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2))


# ------------------------------------------------------------- en-têtes / pieds
def _signature(text: str) -> str:
    return re.sub(r"\d+", "#", _clean(text))


def _boilerplate(doc: pymupdf.Document) -> set[str]:
    """Blocs répétés sur au moins la moitié des pages (« ... | Page 3 ») : pas du contenu."""
    pages_by_sig: dict[str, set[int]] = defaultdict(set)
    for page in doc:
        for block in page.get_text("blocks"):
            if block[6] == 0 and (sig := _signature(block[4])):
                pages_by_sig[sig].add(page.number)
    threshold = max(2, len(doc) // 2)
    return {sig for sig, pages in pages_by_sig.items() if len(pages) >= threshold}


# ------------------------------------------------------------------- sections
@dataclass
class _Sections:
    """Suit la section courante au fil des pages (les titres sont numérotés 1, 2, 3...)."""

    current: str | None = None
    titles: dict[str, str] | None = None

    def __post_init__(self):
        self.titles = {}

    def heading(self, text: str) -> str | None:
        match = HEADING_RE.match(text)
        if not match or "\n" in text:
            return None
        number = match.group(1)
        # Croissant : « 1. Identifier le serveur » dans une procédure n'est pas un titre.
        if self.current is not None and int(number) <= int(self.current):
            return None
        self.current = number
        self.titles[number] = text
        return number

    def title(self, section: str | None) -> str | None:
        return self.titles.get(section) if section else None


def _section_at(y: float, events: list[tuple[float, str]], start: str | None) -> str | None:
    """Section en vigueur à la hauteur y de la page (dernier titre placé au-dessus)."""
    section = start
    for y0, number in events:
        if y0 <= y:
            section = number
    return section


# ------------------------------------------------------------------ passages
def _chunk_items(items: list[tuple[str, list[float], str | None, bool]], max_chars: int):
    """Regroupe les blocs consécutifs d'une même section en passages de taille bornée.

    Un nouveau passage commence à chaque titre de section ou quand la taille est dépassée.
    """
    chunks: list[tuple[str, list[float], str | None]] = []
    buffer: list[str] = []
    bbox: list[float] | None = None
    section: str | None = None
    for text, rect, sec, is_heading in items:
        size = sum(len(b) for b in buffer)
        if buffer and (is_heading or sec != section or size + len(text) > max_chars):
            chunks.append(("\n".join(buffer), bbox, section))
            buffer, bbox = [], None
        buffer.append(text)
        section = sec
        bbox = list(rect) if bbox is None else [
            min(bbox[0], rect[0]), min(bbox[1], rect[1]), max(bbox[2], rect[2]), max(bbox[3], rect[3])]
    if buffer:
        chunks.append(("\n".join(buffer), bbox, section))
    return chunks


# ------------------------------------------------------------------- tableaux
def _table_rows(page: pymupdf.Page, table, words) -> list[list[str]]:
    """Texte de chaque cellule reconstruit à partir des mots qu'elle contient.

    table.extract() colle les mots (« CheminFCdegrade ») ; get_text("words") garde les espaces.
    """
    rows = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            if cell is None:
                cells.append("")
                continue
            rect = pymupdf.Rect(cell)
            inside = sorted((w for w in words if _center_in(w, rect)), key=lambda w: (w[5], w[6], w[7]))
            cells.append(_clean(" ".join(w[4] for w in inside)))
        rows.append(cells)
    return [r for r in rows if any(r)]


def _table_to_markdown(rows: list[list[str]]) -> str:
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    header, *body = rows
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


# -------------------------------------------------------------------- figures
def _captions(blocks, boilerplate: set[str]) -> list[tuple[str, str, pymupdf.Rect]]:
    found = []
    for x0, y0, x1, y1, text, *_ in blocks:
        match = CAPTION_RE.match(text)
        if match and _signature(text) not in boilerplate:
            found.append((match.group(1), _clean(match.group(2)), pymupdf.Rect(x0, y0, x1, y1)))
    return found


def _nearest_caption(rect: pymupdf.Rect, captions):
    if not captions:
        return None

    def distance(c):
        crect = c[2]
        return min(abs(crect.y0 - rect.y1), abs(rect.y0 - crect.y1))

    return min(captions, key=distance)


# -------------------------------------------------------------- extraction
def extract_document(spec: DocumentSpec, settings: Settings, pdf_bytes: bytes | None = None,
                     ocr: OCREngine | None = None) -> ExtractedDocument:
    pdf_path = spec.path  # déjà résolu par load_corpus()
    data = pdf_bytes if pdf_bytes is not None else pdf_path.read_bytes()
    digest = sha256_bytes(data)
    assets_dir = settings.assets_dir / spec.document_id / spec.version
    assets_dir.mkdir(parents=True, exist_ok=True)
    excluded = set(spec.exclude_sections)

    common = dict(
        document_id=spec.document_id, document_title=spec.title, document_version=spec.version,
        document_sha256=digest, source_file=pdf_path.name, equipment_type=spec.equipment_type,
        equipment_family=spec.equipment_family, access_groups=spec.access_groups,
        language=spec.language, index_version=settings.index_version,
    )
    text_model = settings.text_embedding_model
    texts: list[TextElement] = []
    visuals: list[VisualElement] = []

    def add_text(**fields):
        if fields.get("section") not in excluded:
            texts.append(TextElement(**common, embedding_model=text_model, **fields))

    with pymupdf.open(stream=data, filetype="pdf") as doc:
        page_count = len(doc)
        boilerplate = _boilerplate(doc)
        sections = _Sections()

        for page in doc:
            p = page.number + 1
            start_section = sections.current
            blocks = [b for b in page.get_text("blocks", sort=True)
                      if b[6] == 0 and _clean(b[4]) and _signature(b[4]) not in boilerplate]
            try:
                tables = page.find_tables().tables
            except Exception:  # le détecteur peut échouer sur certains PDF
                tables = []
            table_rects = [pymupdf.Rect(t.bbox) for t in tables]

            # 1. Passages textuels (hors tableaux, indexés à part)
            items, events = [], []
            for x0, y0, x1, y1, raw, *_ in blocks:
                text = _clean(raw)
                number = sections.heading(text)
                if number:
                    events.append((y0, number))
                if any(_center_in((x0, y0, x1, y1), r) for r in table_rects):
                    continue
                items.append((text, [x0, y0, x1, y1], sections.current, number is not None))
            for i, (chunk, bbox, sec) in enumerate(_chunk_items(items, settings.chunk_max_chars), 1):
                title = sections.title(sec)
                if title and not chunk.startswith(title):
                    chunk = f"{title}\n{chunk}"  # contexte de section en tête de passage
                add_text(page=p, section=sec, element_id=f"p{p:02d}_chunk_{i:02d}", modality=Modality.TEXT,
                         text=chunk, bbox=bbox, chunk_index=i)

            # 2. Tableaux : version complète (Markdown) + une entrée par ligne
            words = page.get_text("words") if tables else []
            for k, table in enumerate(tables, 1):
                rows = _table_rows(page, table, words)
                if len(rows) < 2:
                    continue
                table_id = f"p{p:02d}_table_{k:02d}"
                sec = _section_at(table.bbox[1], events, start_section)
                title = sections.title(sec)
                markdown = _table_to_markdown(rows)
                add_text(page=p, section=sec, element_id=table_id, modality=Modality.TABLE,
                         text=f"{title}\n{markdown}" if title else markdown, bbox=list(table.bbox),
                         table_id=table_id)
                header, *body = rows
                for r, row in enumerate(body, 1):
                    cells = {(h or f"col{c}"): v for c, (h, v) in enumerate(zip(header, row), 1)}
                    line = " | ".join(f"{h} : {v}" for h, v in cells.items() if v)
                    add_text(page=p, section=sec, element_id=f"{table_id}_row_{r:02d}",
                             modality=Modality.TABLE_ROW, text=f"{title}\n{line}" if title else line,
                             table_id=table_id, cells=cells)

            # 3. Légendes de figures
            captions = _captions(blocks, boilerplate)
            for fig_id, caption, crect in captions:
                add_text(page=p, section=_section_at(crect.y0, events, start_section),
                         element_id=f"p{p:02d}_caption_{fig_id}", modality=Modality.FIGURE_CAPTION,
                         text=f"{fig_id} : {caption}", bbox=list(crect), figure_id=fig_id)

            # 4. Images (+ OCR du texte qu'elles contiennent)
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
                sec = _section_at(rect.y0, events, start_section) if rect is not None else sections.current
                if sec in excluded:
                    continue
                nearest = _nearest_caption(rect, captions) if rect is not None else None
                fig_id, caption = (nearest[0], nearest[1]) if nearest else (None, None)
                ocr_lines = ocr.read((assets_dir / filename).read_bytes()) if ocr else []
                ocr_text = " ; ".join(ocr_lines)
                if len(re.sub(r"\W", "", ocr_text)) < MIN_OCR_CHARS:
                    ocr_text = None  # une lettre isolée (photo de couverture) n'est pas du contenu
                element_id = f"p{p:02d}_image_{k:02d}"
                visuals.append(VisualElement(
                    **common, page=p, section=sec, element_id=element_id,
                    modality=Modality.SCHEMA if fig_id else Modality.ILLUSTRATION,
                    figure_id=fig_id, caption=caption, ocr_text=ocr_text,
                    image_path=(assets_dir / filename).relative_to(settings.data_dir).as_posix(),
                    image_sha256=sha256_bytes(raw), width=width, height=height,
                    bbox=list(rect) if rect is not None else None,
                    embedding_model=settings.image_embedding_model))
                if ocr_text:
                    label = fig_id or "Image"
                    add_text(page=p, section=sec, element_id=f"{element_id}_ocr", modality=Modality.IMAGE_OCR,
                             text=f"{label} - texte lu dans le schéma : {ocr_text}", figure_id=fig_id,
                             bbox=list(rect) if rect is not None else None)

            # 5. Page scannée (presque pas de texte natif mais des images) : OCR de la page entière
            native_chars = sum(len(_clean(b[4])) for b in blocks)
            if ocr and native_chars < settings.ocr_min_page_chars and page.get_images():
                lines = ocr.read(page.get_pixmap(dpi=settings.ocr_dpi).tobytes("png"))
                ocr_items = [(line, list(page.rect), sections.current, False) for line in lines]
                for i, (chunk, bbox, sec) in enumerate(_chunk_items(ocr_items, settings.chunk_max_chars), 1):
                    add_text(page=p, section=sec, element_id=f"p{p:02d}_ocr_{i:02d}", modality=Modality.PAGE_OCR,
                             text=chunk, bbox=bbox, chunk_index=i)

    return ExtractedDocument(spec=spec, sha256=digest, page_count=page_count, texts=texts, visuals=visuals)