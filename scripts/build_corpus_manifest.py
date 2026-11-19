
"""Generate a reproducibility manifest for a PDF retrieval corpus."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from transformers import AutoTokenizer

from evidence_rag.ingestion.chunker import chunk_document
from evidence_rag.ingestion.parser import parse_pdf


MODEL_NAME = "BAAI/bge-small-en-v1.5"
MAX_TOKENS = 384
OVERLAP_TOKENS = 64


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def build_manifest(corpus_dir: Path) -> dict:
    if not corpus_dir.is_dir():
        raise ValueError(f"Corpus directory does not exist: {corpus_dir}")

    paths = sorted(
        (
            path for path in corpus_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.name,
    )

    if not paths:
        raise ValueError("Corpus contains no PDF files")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    tokenizer_fingerprint = hashlib.sha256(
        tokenizer.backend_tokenizer.to_str().encode("utf-8")
    ).hexdigest()

    documents = []
    all_chunk_ids = []

    for path in paths:
        document = parse_pdf(path)

        chunks = chunk_document(
            document,
            tokenizer,
            max_tokens=MAX_TOKENS,
            overlap_tokens=OVERLAP_TOKENS,
        )

        documents.append(
            {
                "source_name": path.name,
                "document_id": document.document_id,
                "sha256": sha256_file(path),
                "page_count": len(document.pages),
                "chunk_count": len(chunks),
                "chunk_ids": [chunk.chunk_id for chunk in chunks],
            }
        )

        all_chunk_ids.extend(chunk.chunk_id for chunk in chunks)

    if len(all_chunk_ids) != len(set(all_chunk_ids)):
        raise ValueError("Duplicate chunk IDs detected")

    source_files = [
        Path("src/evidence_rag/ingestion/parser.py"),
        Path("src/evidence_rag/ingestion/chunker.py"),
        Path("src/evidence_rag/ingestion/models.py"),
    ]

    implementation_hashes = {
        path.as_posix(): sha256_file(path)
        for path in source_files
    }

    return {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tokenizer": {
            "model": MODEL_NAME,
            "fingerprint_sha256": tokenizer_fingerprint,
        },
        "chunking": {
            "max_tokens": MAX_TOKENS,
            "overlap_tokens": OVERLAP_TOKENS,
            "implementation_hashes": implementation_hashes,
        },
        "document_count": len(documents),
        "total_chunks": len(all_chunk_ids),
        "documents": documents,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest = build_manifest(args.corpus)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Documents: {manifest['document_count']}")
    print(f"Chunks: {manifest['total_chunks']}")
    print(f"Manifest saved: {args.output}")


if __name__ == "__main__":
    main()
