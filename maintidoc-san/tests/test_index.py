"""Tests de l'index (étapes 1 et 2) : extraction, métadonnées, idempotence, accès.

Lancement : uv run pytest -q
"""
from __future__ import annotations

import pymupdf
import pytest

from maintidoc_san.extraction import extract_document
from maintidoc_san.ingest import load_corpus

from conftest import FakeOCR, FakeText


def test_extraction_metadata(env):
    settings, _, _ = env
    spec = load_corpus(settings.corpus_file)[0]
    doc = extract_document(spec, settings, ocr=FakeOCR())
    assert doc.page_count == 9
    schemas = [v for v in doc.visuals if v.modality == "schema"]
    assert sorted(v.figure_id for v in schemas) == ["FIG-01", "FIG-02", "FIG-03", "FIG-04", "FIG-05"]
    assert {v.page for v in schemas} == {2, 3, 4, 5, 6}
    modalities = {t.modality for t in doc.texts}
    assert {"text", "table", "table_row", "figure_caption", "image_ocr"} <= modalities
    for el in [*doc.texts, *doc.visuals]:
        assert el.access_groups == ["responsable_maintenance", "technicien"]
        assert el.document_version == "1.1" and el.document_sha256 == doc.sha256
    for v in doc.visuals:
        assert (settings.data_dir / v.image_path).is_file()


def test_sections_and_boilerplate(env):
    settings, _, _ = env
    doc = extract_document(load_corpus(settings.corpus_file)[0], settings)
    by_page = {t.page: t.section for t in doc.texts if t.modality == "text"}
    assert by_page[3] == "2" and by_page[5] == "4" and by_page[6] == "5"
    assert not any("Manuel pedagogique fictif" in t.text for t in doc.texts)  # pied de page retiré


def test_excluded_section_is_not_indexed(env):
    settings, _, _ = env
    doc = extract_document(load_corpus(settings.corpus_file)[0], settings)
    assert all(t.section != "7" for t in doc.texts)
    assert not any("Quel est le role des deux fabrics" in t.text for t in doc.texts)


def test_table_rows_are_readable(env):
    settings, _, _ = env
    doc = extract_document(load_corpus(settings.corpus_file)[0], settings)
    row = next(t for t in doc.texts if t.modality == "table_row" and t.cells and t.cells.get("Code") == "AL-01")
    assert row.page == 5 and row.table_id == "p05_table_01"
    assert row.cells["Signification pour le simulateur"] == "Chemin FC degrade"  # plus de mots collés
    assert "Code : AL-01" in row.text


def test_image_ocr_creates_searchable_text(env):
    settings, _, _ = env
    ocr = FakeOCR(["Controleur A", "Disque B03"])
    doc = extract_document(load_corpus(settings.corpus_file)[0], settings, ocr=ocr)
    fig3 = next(v for v in doc.visuals if v.figure_id == "FIG-03")
    assert fig3.ocr_text == "Controleur A ; Disque B03"
    ocr_text = next(t for t in doc.texts if t.modality == "image_ocr" and t.figure_id == "FIG-03")
    assert "Disque B03" in ocr_text.text


def test_scanned_page_is_ocred(env, tmp_path):
    settings, _, _ = env
    # PDF « scanné » : une page qui ne contient qu'une image, sans texte natif.
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 400, 300), False)
        pix.clear_with(200)
        page.insert_image(page.rect, pixmap=pix)
        data = pdf.tobytes()
    spec = load_corpus(settings.corpus_file)[0]
    doc = extract_document(spec, settings, pdf_bytes=data, ocr=FakeOCR(["Procedure SCAN-01 : verifier le cable"]))
    page_ocr = [t for t in doc.texts if t.modality == "page_ocr"]
    assert page_ocr and "SCAN-01" in page_ocr[0].text


def test_ingestion_is_idempotent(env):
    settings, store, ingestor = env
    specs = load_corpus(settings.corpus_file)
    first = ingestor.ingest_corpus(specs)[0]
    assert first.status == "indexed" and first.text_points > 0 and first.visual_points == 6
    n_text = store.count(settings.text_collection)
    assert ingestor.ingest_corpus(specs)[0].status == "unchanged"
    assert ingestor.ingest_corpus(specs, force=True)[0].status == "indexed"
    assert store.count(settings.text_collection) == n_text  # pas de doublons
    assert (settings.data_dir / "index_manifest.json").is_file()


def test_modified_document_replaces_old_points(env):
    settings, store, ingestor = env
    specs = load_corpus(settings.corpus_file)
    ingestor.ingest_corpus(specs)
    old_sha = next(iter(store.indexed_sha256(settings.text_collection, "SAN-MAN-001")))
    with pymupdf.open(specs[0].path) as doc:
        doc.set_metadata({**doc.metadata, "subject": "revision 1.2"})
        data = doc.tobytes()
    specs[0].path.write_bytes(data)
    report = ingestor.ingest_corpus(specs)[0]
    assert report.status == "indexed"
    hashes = store.indexed_sha256(settings.text_collection, "SAN-MAN-001")
    assert hashes == {report.sha256} and old_sha not in hashes


def test_incompatible_collection_is_refused(env):
    settings, store, _ = env
    store.ensure_collection("ancienne_collection", 32)  # schéma de l'étape 1 (un seul vecteur)
    with pytest.raises(ValueError, match="INDEX_VERSION"):
        store.ensure_collection("ancienne_collection", 32, hybrid=True)


def test_access_filter(env):
    settings, store, ingestor = env
    ingestor.ingest_corpus(load_corpus(settings.corpus_file))
    vec = FakeText().embed_texts(["voyant orange"])[0]
    assert store.search(settings.text_collection, vec, ["technicien"], limit=3, using="dense")
    assert store.search(settings.text_collection, vec, ["invite"], limit=3, using="dense") == []
    with pytest.raises(PermissionError):
        store.search(settings.text_collection, vec, [], limit=3, using="dense")