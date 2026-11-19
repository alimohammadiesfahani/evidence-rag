
import hashlib
from pathlib import Path

import pymupdf

from evidence_rag.ingestion.models import Document, Page


def parse_pdf(path: str | Path) -> Document:
    """Extract page-level text from a PDF without OCR.

    Raises:
        FileNotFoundError: If the source file does not exist.
        ValueError: If the input is invalid or contains no usable text.
    """
    source = Path(path)

    if not source.is_file():
        raise FileNotFoundError(f"PDF not found: {source}")

    if source.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file: {source}")

    file_bytes = source.read_bytes()
    document_id = hashlib.sha256(file_bytes).hexdigest()

    try:
        with pymupdf.open(stream=file_bytes, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise ValueError("Password-protected PDFs are not supported")

            pages = []

            for index, pdf_page in enumerate(pdf):
                text = pdf_page.get_text("text", sort=True).strip()

                pages.append(
                    Page(
                        page_number=index + 1,
                        text=text,
                    )
                )

    except (pymupdf.FileDataError, pymupdf.EmptyFileError) as exc:
        raise ValueError(f"Cannot parse PDF: {source}") from exc

    if not pages or not any(page.text for page in pages):
        raise ValueError(f"PDF contains no extractable text: {source}")

    return Document(
        document_id=document_id,
        source_name=source.name,
        pages=pages,
    )
