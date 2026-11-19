
import hashlib
import re
from typing import Protocol

from evidence_rag.ingestion.models import Chunk, Document


class Tokenizer(Protocol):
    """Minimal interface required from a tokenizer."""

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
    ) -> list[int]: ...


def chunk_document(
    document: Document,
    tokenizer: Tokenizer,
    max_tokens: int = 384,
    overlap_tokens: int = 64,
) -> list[Chunk]:
    """Split document pages into token-bounded passages.

    Chunks never cross page boundaries. Character offsets refer
    to the text stored in each original Page object.

    Overlap is an approximate target measured using the tokenizer.
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError(
            "overlap_tokens must be non-negative "
            "and smaller than max_tokens"
        )

    
    def count_tokens(text: str) -> int:
        return len(
            tokenizer.encode(
                text,
                add_special_tokens=True,
                truncation=True,
                max_length=max_tokens + 1,
            )
        )


    chunks = []

    for page in document.pages:
        text = page.text
        words = list(re.finditer(r"\S+", text))

        if not words:
            continue

        start = 0

        while start < len(words):
            start_char = words[start].start()

            def passage(
                end: int,
                *,
                page_text: str = text,
                offset: int = start_char,
                word_matches: list[re.Match[str]] = words,
            ) -> str:
                return page_text[offset:word_matches[end - 1].end()]

            # Find the longest passage that fits the token budget.
            low = start + 1
            high = len(words)
            best_end = None

            while low <= high:
                mid = (low + high) // 2

                if count_tokens(passage(mid)) <= max_tokens:
                    best_end = mid
                    low = mid + 1
                else:
                    high = mid - 1

            if best_end is None:
                raise ValueError(
                    f"A single word exceeds the token budget "
                    f"on page {page.page_number}"
                )

            end_char = words[best_end - 1].end()
            chunk_text = text[start_char:end_char]
            token_count = count_tokens(chunk_text)

            identity = (
                f"{document.document_id}:"
                f"{page.page_number}:"
                f"{start_char}:{end_char}:"
                f"{chunk_text}"
            )

            chunk_id = hashlib.sha256(
                identity.encode("utf-8")
            ).hexdigest()

            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    source_name=document.source_name,
                    page_number=page.page_number,
                    start_char=start_char,
                    end_char=end_char,
                    text=chunk_text,
                    token_count=token_count,
                )
            )

            if best_end == len(words):
                break

            if overlap_tokens == 0:
                start = best_end
                continue

            # Find the longest trailing overlap within budget.
            next_start = best_end

            for candidate in range(best_end - 1, start, -1):
                overlap_text = text[
                    words[candidate].start():
                    words[best_end - 1].end()
                ]

                if count_tokens(overlap_text) > overlap_tokens:
                    break

                next_start = candidate

            # Always advance to avoid an infinite loop.
            start = min(best_end, next_start)

    return chunks
