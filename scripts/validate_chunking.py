
import argparse
from pathlib import Path

from transformers import AutoTokenizer

from evidence_rag.ingestion.chunker import chunk_document
from evidence_rag.ingestion.parser import parse_pdf

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        use_fast=True,
    )

    document = parse_pdf(args.pdf)

    chunks = chunk_document(
        document=document,
        tokenizer=tokenizer,
        max_tokens=384,
        overlap_tokens=64,
    )

    assert chunks, "No chunks were produced"

    for chunk in chunks:
        source_page = document.pages[chunk.page_number - 1]

        assert (
            source_page.text[chunk.start_char:chunk.end_char]
            == chunk.text
        )

        assert chunk.token_count <= 384

        assert chunk.token_count == len(
            tokenizer.encode(
                chunk.text,
                add_special_tokens=True,
            )
        )

    print(f"Document: {document.source_name}")
    print(f"Pages: {len(document.pages)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Maximum tokens: {max(c.token_count for c in chunks)}")
    print(f"Minimum tokens: {min(c.token_count for c in chunks)}")
    print(f"Unique chunk IDs: {len({c.chunk_id for c in chunks})}")

    print("\nFirst chunk preview:")
    print(chunks[0].text[:500])

    print("\nValidation passed.")


if __name__ == "__main__":
    main()
