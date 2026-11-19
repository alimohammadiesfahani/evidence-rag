
import pytest

from evidence_rag.ingestion.chunker import chunk_document
from evidence_rag.ingestion.models import Document, Page


class FakeTokenizer:
    """Offline tokenizer for deterministic unit tests."""

    def encode(
        self,
        text,
        *,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
    ):
        count = len(text.split())

        if add_special_tokens:
            count += 2

        if truncation and max_length is not None:
            count = min(count, max_length)

        return list(range(count))


@pytest.fixture
def document():
    return Document(
        document_id="a" * 64,
        source_name="research.pdf",
        pages=[
            Page(
                page_number=1,
                text=" ".join(f"word{i}" for i in range(30)),
            ),
            Page(
                page_number=2,
                text="Additional research findings",
            ),
        ],
    )


def test_chunks_respect_token_budget(document):
    chunks = chunk_document(
        document,
        FakeTokenizer(),
        max_tokens=12,
        overlap_tokens=4,
    )

    assert len(chunks) > 2
    assert all(chunk.token_count <= 12 for chunk in chunks)


def test_chunks_preserve_source_metadata(document):
    chunks = chunk_document(
        document,
        FakeTokenizer(),
        max_tokens=12,
        overlap_tokens=4,
    )

    assert {chunk.page_number for chunk in chunks} == {1, 2}
    assert all(chunk.document_id == document.document_id for chunk in chunks)
    assert all(chunk.source_name == "research.pdf" for chunk in chunks)


def test_chunks_have_deterministic_ids(document):
    first = chunk_document(document, FakeTokenizer(), 12, 4)
    second = chunk_document(document, FakeTokenizer(), 12, 4)

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]


def test_character_offsets_match_source(document):
    chunks = chunk_document(document, FakeTokenizer(), 12, 4)

    for chunk in chunks:
        page = document.pages[chunk.page_number - 1]
        assert page.text[chunk.start_char:chunk.end_char] == chunk.text


def test_adjacent_chunks_overlap(document):
    chunks = chunk_document(document, FakeTokenizer(), 12, 4)
    first_page = [c for c in chunks if c.page_number == 1]

    assert first_page[0].end_char > first_page[1].start_char


def test_invalid_chunk_configuration(document):
    with pytest.raises(ValueError):
        chunk_document(document, FakeTokenizer(), 0, 0)

    with pytest.raises(ValueError):
        chunk_document(document, FakeTokenizer(), 12, 12)


def test_empty_page_is_skipped():
    document = Document(
        document_id="b" * 64,
        source_name="empty-page.pdf",
        pages=[
            Page(page_number=1, text=""),
            Page(page_number=2, text="Actual content here"),
        ],
    )

    chunks = chunk_document(document, FakeTokenizer(), 12, 0)

    assert len(chunks) == 1
    assert chunks[0].page_number == 2


def test_single_oversized_word_raises():
    document = Document(
        document_id="c" * 64,
        source_name="long.pdf",
        pages=[Page(page_number=1, text="supercalifragilistic")],
    )

    class OversizedTokenizer:
        def encode(
            self,
            text,
            *,
            add_special_tokens=True,
            truncation=False,
            max_length=None,
        ):
            count = 100

            if truncation and max_length is not None:
                count = min(count, max_length)

            return list(range(count))

    with pytest.raises(ValueError, match="single word"):
        chunk_document(document, OversizedTokenizer(), 12, 0)
