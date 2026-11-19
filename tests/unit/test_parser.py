
import pymupdf
import pytest

from evidence_rag.ingestion.parser import parse_pdf


def create_pdf(path, texts):
    pdf = pymupdf.open()

    for text in texts:
        page = pdf.new_page()
        if text:
            page.insert_text((72, 72), text)

    pdf.save(path)
    pdf.close()


def test_parse_pdf_extracts_pages(tmp_path):
    path = tmp_path / "paper.pdf"
    create_pdf(path, ["First page content", "Second page content"])

    document = parse_pdf(path)

    assert document.source_name == "paper.pdf"
    assert len(document.pages) == 2
    assert document.pages[0].page_number == 1
    assert document.pages[1].page_number == 2
    assert "First page content" in document.pages[0].text
    assert "Second page content" in document.pages[1].text


def test_document_id_is_deterministic(tmp_path):
    path = tmp_path / "paper.pdf"
    create_pdf(path, ["Reproducible document"])

    first = parse_pdf(path)
    second = parse_pdf(path)

    assert first.document_id == second.document_id
    assert len(first.document_id) == 64


def test_missing_pdf_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        parse_pdf(tmp_path / "missing.pdf")


def test_empty_text_pdf_raises_error(tmp_path):
    path = tmp_path / "empty.pdf"
    create_pdf(path, [""])

    with pytest.raises(ValueError, match="no extractable text"):
        parse_pdf(path)


def test_invalid_pdf_raises_error(tmp_path):
    path = tmp_path / "invalid.pdf"
    path.write_bytes(b"This is not a PDF")

    with pytest.raises(ValueError, match="Cannot parse PDF"):
        parse_pdf(path)
